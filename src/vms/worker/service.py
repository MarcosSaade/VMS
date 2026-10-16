"""The inference worker loop: lease cameras, claim frames, detect, write results (D3, D20).

One iteration (`step`):

1. Leases: renew this worker's camera leases every `renew_every_s`, hand back cameras with
   nothing claimed for `idle_release_s` and no pending frames (a camera that stopped sending
   must not hold a slot forever), and lease more cameras (up to `max_cameras`) on the same
   cadence or whenever there was nothing to claim.
2. Claim up to `claim_batch` pending frames of leased cameras, oldest first.
3. In batches of up to `infer_batch` frames (task 5.5; 1 is one frame at a time): fetch the
   JPEGs concurrently, then in the inference thread (CPU bound, so off the event loop) decode
   and input statistics per frame and one `detect_batch` call per image shape, then tracking
   frame by frame in claim order; then for each frame one transaction that writes its
   detections and marks it done, fenced on this worker's claim. A batch is whatever is
   claimed: the worker never waits for more frames to fill one.

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
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol, TypeVar, cast

import cv2
import numpy as np
from numpy.typing import NDArray
from opentelemetry import context as otel_context
from opentelemetry import trace

from vms.common.clock import Clock
from vms.common.ids import UlidGenerator
from vms.common.logs import get_logger
from vms.common.metrics import Metrics
from vms.common.telemetry import context_from_traceparent
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
A = TypeVar("A")

# Dense from 25 to 300 ms, where CPU inference lives (E-013): with only 0.1 and 0.25 there, the
# inference p95 read as a flat 100 ms bucket edge (E-026).
STAGE_BUCKETS_S = (
    0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.2, 0.25, 0.3, 0.5,
    1.0, 2.5, 5.0,
)  # fmt: skip
# Dense around the 2 s freshness SLO (ARCHITECTURE 8), with an edge at exactly 2 s so the
# share of frames within the SLO is read from a bucket, not interpolated.
# Past the SLO it keeps enough edges for minutes of overload: with only 60 and 300 s, a p95
# inside that bucket was interpolated to 4.8 min when no frame had waited much over 2 min (E-026).
FRESHNESS_BUCKETS_S = (
    0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 5.0, 10.0, 20.0, 30.0, 45.0, 60.0,
    90.0, 120.0, 180.0, 240.0, 300.0, 600.0,
)  # fmt: skip


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
    noise: float
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
        # One inference thread: frames in order, and the backends already use several cores
        # inside one call.
        self._executor = executor or ThreadPoolExecutor(1, thread_name_prefix="infer")
        self._owns_executor = executor is None
        self._tracer = tracer or trace.get_tracer(__name__)

        self._held: set[str] = set()
        self._active: dict[str, float] = {}  # camera -> monotonic time leased or last claimed
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
        self._batch_size = metrics.histogram(
            "infer_batch",
            "frames",
            "Frames per detect_batch call",
            buckets=(1, 2, 3, 4, 6, 8, 12, 16, 24, 32),
        )
        self._leased = metrics.gauge("leased", "cameras", "Cameras leased by this worker")
        # Both are global (every worker reports the same value): aggregate with max, not sum.
        self._backlog = metrics.gauge("backlog", "frames", "Pending frames across all cameras")
        self._backlog_age = metrics.gauge(
            "backlog_age",
            "seconds",
            "Wait so far of the oldest pending frame (database clock)",
        )

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
            backlog = await self._queue.backlog()
            self._backlog.set(backlog.frames)
            self._backlog_age.set(backlog.oldest_age_s)
            self._last_backlog = now

        # Histogram only: a span per poll would be a one-span trace, mostly of idle polls.
        with self._timed("claim", span=False):
            frames = await self._queue.claim(self.worker_id, self._settings.claim_batch)
        if not frames:
            # Nothing to do with the cameras held: look for new ones right away.
            await self._acquire()
            return 0
        for frame in frames:
            self._active[frame.camera_id] = now
        size = self._settings.infer_batch
        for i in range(0, len(frames), size):
            if stop is not None and stop.is_set():
                log.info("stopping; handing back claimed frames", extra={"n": len(frames) - i})
                break
            await self.process_batch(frames[i : i + size])
        return len(frames)

    async def _renew_and_acquire(self) -> None:
        self._held = set(await self._queue.renew(self.worker_id, self._settings.lease_ttl_s))
        lost = set(self._cameras) - self._held
        for camera_id in lost:  # another worker has it now; its history is not ours to continue
            del self._cameras[camera_id]
        if lost:
            log.warning("lost camera leases", extra={"cameras": sorted(lost)})
        now = self._clock.monotonic()
        idle = sorted(
            c for c in self._held if now - self._active.get(c, now) >= self._settings.idle_release_s
        )
        if idle:
            released = set(await self._queue.release_idle(self.worker_id, idle))
            if released:
                log.info("released idle cameras", extra={"cameras": sorted(released)})
            self._held -= released
        for camera_id in set(self._cameras) - self._held:
            del self._cameras[camera_id]
        for camera_id in set(self._active) - self._held:
            del self._active[camera_id]
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
            now = self._clock.monotonic()
            for camera_id in new:
                self._active[camera_id] = now
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
        self._active.clear()
        self._leased.set(0)
        if self._owns_executor:
            self._executor.shutdown(wait=True)

    # --- Frames -------------------------------------------------------------------------

    async def process(self, frame: ClaimedFrame) -> Outcome:
        """One frame on its own (a batch of one)."""
        return (await self.process_batch([frame]))[0]

    async def process_batch(self, frames: list[ClaimedFrame]) -> list[Outcome]:
        """Fetch the frames concurrently, detect them in one `detect_batch` call per image
        shape, track them in claim order (capture order per camera), then write each in its
        own fenced transaction. Outcomes are per frame: one bad JPEG or failed write does not
        fail the others; a model error fails (and retries) the whole batch."""
        now = self._clock.now()
        spans = [self._frame_span(frame, now) for frame in frames]
        outcomes: list[Outcome | None] = [None] * len(frames)
        try:
            for frame in frames:
                self._wait.observe(max(0.0, (now - frame.received_at).total_seconds()))
            states = {f.camera_id: await self._camera_state(f.camera_id) for f in frames}

            async def fetch(i: int) -> bytes:
                with trace.use_span(spans[i], end_on_exit=False), self._timed("fetch"):
                    return await self._fetcher.get(frames[i].object_key)

            fetched = await asyncio.gather(
                *(fetch(i) for i in range(len(frames))), return_exceptions=True
            )
            todo: list[tuple[int, bytes]] = []
            for i, got in enumerate(fetched):
                if isinstance(got, bytes):
                    todo.append((i, got))
                    continue
                with trace.use_span(spans[i], end_on_exit=False):
                    if isinstance(got, ObjectStoreError):
                        outcomes[i] = await self._fail(frames[i], f"fetch: {got}", retry=True)
                    else:
                        outcomes[i] = await self._fail(
                            frames[i], f"fetch: {type(got).__name__}: {got}", retry=True
                        )
            if todo:
                try:
                    analyses, infer_s = await self._in_thread(
                        lambda items: self._analyze_batch(items, frames, states, spans), todo
                    )
                except Exception as exc:  # the model: out of memory, a CUDA error, a bug
                    log.exception("inference failed", extra={"frames": len(todo)})
                    for i, _ in todo:
                        with trace.use_span(spans[i], end_on_exit=False):
                            outcomes[i] = await self._fail(
                                frames[i], f"inference: {type(exc).__name__}: {exc}", True
                            )
                else:
                    for seconds, size in infer_s:
                        self._stage.labels(stage="infer").observe(seconds)
                        self._batch_size.observe(size)
                    for i, analysis in analyses:
                        with trace.use_span(spans[i], end_on_exit=False):
                            outcomes[i] = await self._complete(frames[i], analysis)
        finally:
            for span, outcome in zip(spans, outcomes, strict=True):
                if outcome is not None:
                    span.set_attribute("outcome", outcome.value)
                span.end()
        return [o if o is not None else Outcome.RETRY for o in outcomes]

    def _frame_span(self, frame: ClaimedFrame, now: datetime) -> trace.Span:
        # Continue the frame's trace from ingest (D25): the time in the queue becomes a span of
        # its own, so a trace shows waiting apart from working.
        parent = context_from_traceparent(frame.traceparent)
        if parent is not None:
            wait = self._tracer.start_span(
                "worker.queue_wait",
                context=parent,
                start_time=_unix_ns(frame.received_at),
                attributes={"camera_id": frame.camera_id, "frame_id": frame.frame_id},
            )
            wait.end(end_time=_unix_ns(now))
        return self._tracer.start_span(
            "worker.frame",
            context=parent,
            kind=trace.SpanKind.CONSUMER,
            attributes={
                "camera_id": frame.camera_id,
                "frame_id": frame.frame_id,
                "attempt": frame.attempts,
            },
        )

    async def _complete(self, frame: ClaimedFrame, analysis: Analysis | DecodeError) -> Outcome:
        if isinstance(analysis, DecodeError):
            return await self._fail(frame, f"decode: {analysis}", retry=False)
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
            noise=analysis.noise,
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
        freshness_s = max(0.0, (self._clock.now() - frame.ts).total_seconds())
        self._freshness.observe(freshness_s)
        # On the span too, so slow frames can be found in Tempo: {span.freshness_s > 2.0}.
        trace.get_current_span().set_attribute("freshness_s", freshness_s)
        return Outcome.DONE

    def _analyze_batch(
        self,
        items: list[tuple[int, bytes]],
        frames: list[ClaimedFrame],
        states: dict[str, CameraState],
        spans: list[trace.Span],
    ) -> tuple[list[tuple[int, Analysis | DecodeError]], list[tuple[float, int]]]:
        """Runs in the inference thread: per frame decode and input statistics, then one
        `detect_batch` per image shape (mixed shapes would be letterboxed to squares, changing
        the detections), then tracking and zones frame by frame in claim order. Returns each
        frame's analysis (or its decode error) and (seconds, size) per detect_batch call."""
        out: dict[int, Analysis | DecodeError] = {}
        timings: dict[int, dict[str, float]] = {}
        decoded: dict[int, tuple[NDArray[np.uint8], Any]] = {}
        for i, jpeg in items:
            timings[i] = {}
            with self._frame_context(spans[i]):
                try:
                    with self._stage_timer(timings[i], "decode"):
                        image = decode_jpeg(jpeg)
                except DecodeError as exc:
                    out[i] = exc
                    continue
                with self._stage_timer(timings[i], "stats"):
                    decoded[i] = (image, input_stats(image))

        found: dict[int, list[Detection]] = {}
        infer_s: list[tuple[float, int]] = []
        by_shape: dict[tuple[int, ...], list[int]] = {}
        for i, (image, _) in decoded.items():
            by_shape.setdefault(image.shape, []).append(i)
        for group in by_shape.values():
            start_ns = time.time_ns()
            t0 = self._clock.monotonic()
            results = self._detector.detect_batch([decoded[i][0] for i in group])
            infer_s.append((self._clock.monotonic() - t0, len(group)))
            end_ns = time.time_ns()
            for i, dets in zip(group, results, strict=True):
                found[i] = dets
                # The shared forward pass appears in each frame's trace, with its batch size.
                self._tracer.start_span(
                    "worker.infer",
                    context=trace.set_span_in_context(spans[i]),
                    start_time=start_ns,
                    attributes={"batch_size": len(group)},
                ).end(end_time=end_ns)

        for i, _ in items:  # claim order: capture order within each camera
            if i not in decoded:
                continue
            frame, (image, stats) = frames[i], decoded[i]
            state = states[frame.camera_id]
            height, width = image.shape[:2]
            key = (frame.ts, frame.seq)
            track = state.last_key is None or key > state.last_key
            if not track:
                self._late.inc()
            elif state.last_key is not None and (
                (frame.ts - state.last_key[0]).total_seconds() > self._tracking.buffer_s
            ):
                self._reset_tracker(state)
            with self._frame_context(spans[i]):
                if track:
                    with self._stage_timer(timings[i], "track"):
                        tracked = self._track(found[i], frame, state, width, height)
                else:
                    tracked = self._track(found[i], frame, None, width, height)
            out[i] = Analysis(
                width,
                height,
                stats.brightness,
                stats.contrast,
                stats.blur_var,
                stats.noise,
                tracked.detections,
                tracked.tracks,
                tracked.zone_events,
                track,
                timings[i],
            )
        return [(i, out[i]) for i, _ in items], infer_s

    @contextmanager
    def _frame_context(self, span: trace.Span) -> Iterator[None]:
        token = otel_context.attach(trace.set_span_in_context(span))
        try:
            yield
        finally:
            otel_context.detach(token)

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

    async def _in_thread(self, fn: Callable[[A], T], arg: A) -> T:
        ctx = otel_context.get_current()

        def call() -> T:
            token = otel_context.attach(ctx)  # spans in the thread join the frame's trace
            try:
                return fn(arg)
            finally:
                otel_context.detach(token)

        return await asyncio.get_running_loop().run_in_executor(self._executor, call)

    @contextmanager
    def _timed(self, stage: str, *, span: bool = True) -> Iterator[None]:
        with (
            self._tracer.start_as_current_span(f"worker.{stage}")
            if span
            else contextlib.nullcontext()
        ):
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


def _unix_ns(ts: datetime) -> int:
    return int(ts.timestamp() * 1_000_000_000)


def decode_jpeg(data: bytes) -> NDArray[np.uint8]:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise DecodeError(f"cannot decode {len(data)} bytes as an image")
    return cast(NDArray[np.uint8], image)
