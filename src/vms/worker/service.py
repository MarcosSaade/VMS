"""The inference worker loop: lease cameras, claim frames, detect, write results (D3, D20).

One iteration (`step`):

1. Leases: renew this worker's camera leases every `renew_every_s`, and lease more cameras
   (up to `max_cameras`) on the same cadence or whenever there was nothing to claim.
2. Claim up to `claim_batch` pending frames of leased cameras, oldest first.
3. For each frame, in order: fetch the JPEG, then decode, input statistics and detection in
   the inference thread (CPU bound, so off the event loop), then one transaction that writes
   the detections and marks the frame done, fenced on this worker's claim.

Failures: an undecodable JPEG fails the frame for good (retrying cannot fix it); an object
store, model or database error puts it back to pending until `max_attempts`. A frame whose
claim was lost (this worker stalled past its lease and another took the camera) is dropped
without writing. If even reporting a failure to the database fails, the worker requeues its
own in-flight frames at the start of the next step.

Tracking and zones (task 1.6). Each leased camera has a `CameraState` in this worker: a
ByteTrack tracker, the zone state machine and the camera's zones (reloaded every
`zones.refresh_s`). After detection, still in the inference thread, the frame's detections go
through the tracker, tracked anchors through the zones, and the result carries track ids on
detections, one track sighting per tracked object and any zone events, all written in the
frame's fenced transaction. A camera's state is dropped when its lease is lost, so a camera
coming back starts a new tracker session (new track ids).

A pause longer than `tracker.buffer_s` between two tracked frames of a camera starts a new
tracker session: ByteTrack counts its buffer in frames, so it would otherwise join objects
across the pause.

The tracker must see frames in capture order. A frame captured no later than the last one
tracked for its camera (it reached ingest late, or is a retry of a frame whose write failed)
is detected and stored without tracking, and counted as late. If a frame's write fails after
the tracker has seen it, that frame's track sightings and events are lost with it.

Low-confidence detections (below `tracker.store_min_conf`) feed the tracker's second round
but are stored only when they belong to a track.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Iterator
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, TypeVar, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from opentelemetry import context as otel_context
from opentelemetry import trace

from vms.common.clock import Clock
from vms.common.ids import UlidGenerator
from vms.common.logs import get_logger
from vms.common.metrics import Metrics
from vms.storage.frames import FrameStoreError
from vms.storage.objects import ObjectStoreError
from vms.storage.work import (
    ClaimedFrame,
    DetectionRow,
    EventRow,
    FailOutcome,
    FrameResult,
    TrackUpdate,
    WorkQueue,
)
from vms.worker.detector import Detection, Detector
from vms.worker.pipeline import CameraState, TrackedFrame, Tracker, track_frame
from vms.worker.settings import QueueSettings
from vms.worker.stats import input_stats
from vms.worker.tracking import CameraTracker, TrackerSettings
from vms.worker.zones import ZoneDef, ZoneEvent, ZoneSettings, ZoneTracker

log = get_logger(__name__)

T = TypeVar("T")

STAGE_BUCKETS_S = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
FRESHNESS_BUCKETS_S = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 300.0)


class FrameFetcher(Protocol):
    async def get(self, key: str) -> bytes: ...


class Outcome(StrEnum):
    DONE = "done"
    RETRY = "retry"
    FAILED = "failed"
    LOST = "lost"


class DecodeError(ValueError):
    """The object is not a decodable image: a permanent failure."""


@dataclass(frozen=True)
class Analysis:
    width: int
    height: int
    brightness: float
    contrast: float
    blur_var: float
    detections: tuple[DetectionRow, ...]
    tracks: tuple[TrackUpdate, ...]
    zone_events: tuple[ZoneEvent, ...]
    tracked: bool
    timings_s: dict[str, float]


class Worker:
    def __init__(
        self,
        queue: WorkQueue,
        fetcher: FrameFetcher,
        detector: Detector,
        settings: QueueSettings,
        metrics: Metrics,
        *,
        worker_id: str,
        clock: Clock,
        ids: UlidGenerator,
        tracker_settings: TrackerSettings | None = None,
        zone_settings: ZoneSettings | None = None,
        tracker_factory: Callable[[TrackerSettings], Tracker] = CameraTracker,
        executor: Executor | None = None,
        tracer: trace.Tracer | None = None,
    ) -> None:
        self.worker_id = worker_id
        self._queue = queue
        self._fetcher = fetcher
        self._detector = detector
        self._settings = settings
        self._clock = clock
        self._ids = ids
        self._tracking = tracker_settings or TrackerSettings()
        self._zone_settings = zone_settings or ZoneSettings()
        self._tracker_factory = tracker_factory
        self._cameras: dict[str, CameraState] = {}
        # One inference thread: batch size 1, frames in order, and PyTorch already uses
        # several cores inside one call.
        self._executor = executor or ThreadPoolExecutor(1, thread_name_prefix="infer")
        self._owns_executor = executor is None
        self._tracer = tracer or trace.get_tracer(__name__)

        self._held: set[str] = set()
        self._last_leases = float("-inf")
        self._last_backlog = float("-inf")
        self._requeue_pending = False

        self._frames = metrics.counter("frames", "total", "Frames by outcome", ["outcome"])
        self._detections = metrics.counter("detections", "total", "Detections written")
        self._late = metrics.counter(
            "late_frames",
            "total",
            "Frames not tracked: captured no later than the last tracked frame of the camera",
        )
        self._zone_events = metrics.counter(
            "zone_events", "total", "Zone events written", ["event_type"]
        )
        self._resets = metrics.counter(
            "tracker_resets", "total", "New tracker sessions after a pause longer than the buffer"
        )
        self._track_sightings = metrics.counter(
            "track_sightings", "total", "Tracked objects written (one per track per frame)"
        )
        self._errors = metrics.counter(
            "loop_errors", "total", "Queue operations that failed (database)", ["op"]
        )
        self._stage = metrics.histogram(
            "stage_duration", "seconds", "Time per worker stage", ["stage"], STAGE_BUCKETS_S
        )
        self._freshness = metrics.histogram(
            "capture_to_done",
            "seconds",
            "Detections stored minus frame capture time (freshness)",
            buckets=FRESHNESS_BUCKETS_S,
        )
        self._wait = metrics.histogram(
            "queue_wait",
            "seconds",
            "Frame claimed minus frame received by ingest",
            buckets=FRESHNESS_BUCKETS_S,
        )
        self._leased = metrics.gauge("leased", "cameras", "Cameras leased by this worker")
        self._backlog = metrics.gauge("backlog", "frames", "Pending frames across all cameras")

    @property
    def leased_cameras(self) -> frozenset[str]:
        return frozenset(self._held)

    # --- Loop ---------------------------------------------------------------------------

    async def run(self, stop: asyncio.Event) -> None:
        """Process frames until `stop` is set, then hand back leases and unfinished frames."""
        idle_s = self._settings.poll_min_s
        try:
            while not stop.is_set():
                try:
                    claimed = await self.step(stop)
                except FrameStoreError as exc:
                    log.warning("queue unavailable", extra={"error": str(exc)})
                    claimed = 0
                if claimed:
                    idle_s = self._settings.poll_min_s
                    continue
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(stop.wait(), timeout=idle_s)
                idle_s = min(idle_s * 2, self._settings.poll_max_s)
        finally:
            await self.release()

    async def step(self, stop: asyncio.Event | None = None) -> int:
        """One iteration; returns how many frames were claimed. Raises FrameStoreError if the
        queue cannot be reached (the caller backs off)."""
        if self._requeue_pending:
            await self._queue.release_frames(self.worker_id)
            self._requeue_pending = False
        now = self._clock.monotonic()
        if now - self._last_leases >= self._settings.renew_every_s:
            await self._renew_and_acquire()
            self._last_leases = now
        if now - self._last_backlog >= self._settings.backlog_every_s:
            self._backlog.set(await self._queue.backlog())
            self._last_backlog = now

        with self._timed("claim"):
            frames = await self._queue.claim(self.worker_id, self._settings.claim_batch)
        if not frames:
            # Nothing to do with the cameras held: look for new ones right away.
            await self._acquire()
            return 0
        for i, frame in enumerate(frames):
            if stop is not None and stop.is_set():
                log.info("stopping; handing back claimed frames", extra={"n": len(frames) - i})
                break
            await self.process(frame)
        return len(frames)

    async def _renew_and_acquire(self) -> None:
        self._held = set(await self._queue.renew(self.worker_id, self._settings.lease_ttl_s))
        lost = set(self._cameras) - self._held
        for camera_id in lost:  # another worker has it now; its history is not ours to continue
            del self._cameras[camera_id]
        if lost:
            log.warning("lost camera leases", extra={"cameras": sorted(lost)})
        await self._acquire()

    async def _acquire(self) -> None:
        room = self._settings.max_cameras - len(self._held)
        if room > 0:
            new = await self._queue.acquire_cameras(
                self.worker_id, room, self._settings.lease_ttl_s, self._settings.max_attempts
            )
            if new:
                log.info("leased cameras", extra={"cameras": new})
            self._held.update(new)
        self._leased.set(len(self._held))

    async def release(self) -> None:
        """Hand back leases and in-flight frames so another worker can continue at once."""
        try:
            await self._queue.release(self.worker_id)
        except FrameStoreError as exc:
            # The leases expire on their own after lease_ttl_s.
            log.warning("release failed", extra={"error": str(exc)})
        self._held.clear()
        self._cameras.clear()
        self._leased.set(0)
        if self._owns_executor:
            self._executor.shutdown(wait=True)

    # --- One frame ----------------------------------------------------------------------

    async def process(self, frame: ClaimedFrame) -> Outcome:
        with self._tracer.start_as_current_span(
            "worker.frame",
            attributes={"camera_id": frame.camera_id, "frame_id": frame.frame_id},
        ):
            self._wait.observe(max(0.0, (self._clock.now() - frame.received_at).total_seconds()))
            state = await self._camera_state(frame.camera_id)
            key = (frame.ts, frame.seq)
            track = state.last_key is None or key > state.last_key
            if not track:
                self._late.inc()
            elif state.last_key is not None and (
                (frame.ts - state.last_key[0]).total_seconds() > self._tracking.buffer_s
            ):
                self._reset_tracker(state)
            try:
                with self._timed("fetch"):
                    jpeg = await self._fetcher.get(frame.object_key)
                analysis = await self._in_thread(
                    lambda data: self._analyze(data, frame, state if track else None), jpeg
                )
            except DecodeError as exc:
                return await self._fail(frame, f"decode: {exc}", retry=False)
            except ObjectStoreError as exc:
                return await self._fail(frame, f"fetch: {exc}", retry=True)
            except Exception as exc:  # the model: out of memory, a CUDA error, a bug
                log.exception("inference failed", extra={"frame_id": frame.frame_id})
                return await self._fail(frame, f"inference: {type(exc).__name__}: {exc}", True)
            for stage, seconds in analysis.timings_s.items():
                self._stage.labels(stage=stage).observe(seconds)

            result = FrameResult(
                frame_id=frame.frame_id,
                camera_id=frame.camera_id,
                ts=frame.ts,
                width=analysis.width,
                height=analysis.height,
                brightness=analysis.brightness,
                contrast=analysis.contrast,
                blur_var=analysis.blur_var,
                model_version=self._detector.model_version,
                detections=analysis.detections,
                tracks=analysis.tracks,
                events=tuple(
                    EventRow(self._ids.new(), e.ts, e.track_id, e.zone_id, e.event_type)
                    for e in analysis.zone_events
                ),
            )
            try:
                with self._timed("write"):
                    won = await self._queue.complete(self.worker_id, result)
            except FrameStoreError as exc:
                return await self._fail(frame, f"write: {exc}", retry=True)
            if not won:
                self._frames.labels(outcome=Outcome.LOST).inc()
                log.warning("claim lost before completion", extra={"frame_id": frame.frame_id})
                return Outcome.LOST
            self._frames.labels(outcome=Outcome.DONE).inc()
            self._detections.inc(len(result.detections))
            self._track_sightings.inc(len(result.tracks))
            for event in result.events:
                self._zone_events.labels(event_type=event.event_type).inc()
            self._freshness.observe(max(0.0, (self._clock.now() - frame.ts).total_seconds()))
            return Outcome.DONE

    def _analyze(self, jpeg: bytes, frame: ClaimedFrame, state: CameraState | None) -> Analysis:
        """Decode, input statistics, detection and (if `state`) tracking and zones; runs in the
        inference thread. `state` is None for frames that must not be tracked (late)."""
        timings: dict[str, float] = {}
        with self._stage_timer(timings, "decode"):
            image = decode_jpeg(jpeg)
        with self._stage_timer(timings, "stats"):
            stats = input_stats(image)
        with self._stage_timer(timings, "infer"):
            found = self._detector.detect(image)
        height, width = image.shape[:2]

        if state is not None:
            with self._stage_timer(timings, "track"):
                tracked = self._track(found, frame, state, width, height)
        else:
            tracked = self._track(found, frame, None, width, height)
        return Analysis(
            width,
            height,
            stats.brightness,
            stats.contrast,
            stats.blur_var,
            tracked.detections,
            tracked.tracks,
            tracked.zone_events,
            state is not None,
            timings,
        )

    def _track(
        self,
        found: list[Detection],
        frame: ClaimedFrame,
        state: CameraState | None,
        width: int,
        height: int,
    ) -> TrackedFrame:
        return track_frame(
            found,
            camera_id=frame.camera_id,
            ts=frame.ts,
            seq=frame.seq,
            state=state,
            width=width,
            height=height,
            store_min_conf=self._tracking.store_min_conf,
        )

    def _reset_tracker(self, state: CameraState) -> None:
        """Start a new tracker session after a pause longer than the tracker's buffer.

        ByteTrack's buffer counts frames, not seconds, so across a pause (camera offline, a
        backlog of old frames) it would join objects seen before it to whatever is there after.
        The zone tracker needs no reset: it expires tracks by capture time and closes their
        visits at their last sighting.
        """
        state.session = self._ids.new()[-8:].lower()
        state.tracker = self._tracker_factory(self._tracking)
        self._resets.inc()

    async def _camera_state(self, camera_id: str) -> CameraState:
        """This worker's state for a camera, created on its first frame; zones are (re)loaded
        when stale. A failed zone load keeps the zones already known and retries next frame."""
        state = self._cameras.get(camera_id)
        if state is None:
            state = CameraState(
                session=self._ids.new()[-8:].lower(),
                tracker=self._tracker_factory(self._tracking),
                zones=ZoneTracker((), self._zone_settings, self._tracking.buffer_s),
                zones_loaded_at=float("-inf"),
            )
            self._cameras[camera_id] = state
        now = self._clock.monotonic()
        if now - state.zones_loaded_at >= self._zone_settings.refresh_s:
            try:
                rows = await self._queue.zones(camera_id)
            except FrameStoreError as exc:
                self._errors.labels(op="zones").inc()
                log.warning("zone load failed", extra={"camera_id": camera_id, "error": str(exc)})
            else:
                state.zones.set_zones(ZoneDef(r.zone_id, r.name, r.polygon) for r in rows)
                state.zones_loaded_at = now
        return state

    async def _fail(self, frame: ClaimedFrame, error: str, retry: bool) -> Outcome:
        try:
            result = await self._queue.fail(
                self.worker_id,
                frame.frame_id,
                error,
                retry=retry,
                max_attempts=self._settings.max_attempts,
            )
        except FrameStoreError as exc:
            # The frame stays claimed by us; requeue our in-flight frames next step.
            self._errors.labels(op="fail").inc()
            self._requeue_pending = True
            log.warning("could not record failure", extra={"error": str(exc)})
            result = FailOutcome.RETRY
        outcome = {
            FailOutcome.RETRY: Outcome.RETRY,
            FailOutcome.FAILED: Outcome.FAILED,
            FailOutcome.LOST: Outcome.LOST,
        }[result]
        self._frames.labels(outcome=outcome).inc()
        log.warning(
            "frame not processed",
            extra={"frame_id": frame.frame_id, "outcome": outcome, "error": error},
        )
        return outcome

    async def _in_thread(self, fn: Callable[[bytes], T], arg: bytes) -> T:
        ctx = otel_context.get_current()

        def call() -> T:
            token = otel_context.attach(ctx)  # spans in the thread join the frame's trace
            try:
                return fn(arg)
            finally:
                otel_context.detach(token)

        return await asyncio.get_running_loop().run_in_executor(self._executor, call)

    @contextmanager
    def _timed(self, stage: str) -> Iterator[None]:
        with self._tracer.start_as_current_span(f"worker.{stage}"):
            t0 = self._clock.monotonic()
            try:
                yield
            finally:
                self._stage.labels(stage=stage).observe(self._clock.monotonic() - t0)

    @contextmanager
    def _stage_timer(self, timings: dict[str, float], stage: str) -> Iterator[None]:
        """Like `_timed`, but records into `timings` (histograms are observed on the loop)."""
        with self._tracer.start_as_current_span(f"worker.{stage}"):
            t0 = self._clock.monotonic()
            try:
                yield
            finally:
                timings[stage] = self._clock.monotonic() - t0


def decode_jpeg(data: bytes) -> NDArray[np.uint8]:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise DecodeError(f"cannot decode {len(data)} bytes as an image")
    return cast(NDArray[np.uint8], image)
