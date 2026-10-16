"""The drift monitor loop: per camera, the latest window against its reference (task 9.1).

Every `update_every_s`, `update`:

1. Finds cameras (every `discover_every_s`) and reads the reference ranges.
2. Reads the window [now - settle_s - window_s, now - settle_s) of done frames and their
   detections for every known camera, in one query each.
3. Gives a camera seen for the first time a reference: its first `reference_s` of capture
   time from its first frame in this window (`reason: auto`). While that period is still
   running, or settling, the camera has no statistics (`reference_frames` 0).
4. Loads a finished reference's frames and detections once and keeps them in memory, until
   its range changes (`python -m vms.drift recalibrate`) or the process restarts; the range
   lives in Postgres, so a restart recomputes the same reference.
5. Compares window and reference (vms.drift.signals) and sets the gauges.
6. If a detector and its clean-run calibration are configured (task 9.3), standardizes each
   channel, updates the camera's detector (vms.drift.detect) and exports each channel's z,
   its CUSUM statistic and whether it alarms; Prometheus alert rules fire on the alarm gauge
   (deploy/compose/prometheus/rules/drift.yml). A camera's detector restarts when its
   reference changes. Windows whose frames span less than `detect_min_coverage` of
   `window_s` (a camera starting, resuming or stopping) are not fed to the detector.

Gauges carry `camera_id` (and `signal`); an undefined statistic (too few frames or
detections) is exported as NaN, which Prometheus stores and Grafana draws as a gap, so "no
data" never looks like "no drift". Alerting on these gauges is task 9.3.

The monitor is one process for every camera: it is independent of which worker holds which
camera, and it only reads aggregates' raw inputs that the worker already stores. Its cost is
two indexed range reads per update (about window_s x FPS frames and their detections per
camera).
"""

from __future__ import annotations

import asyncio
import contextlib
import math
from datetime import datetime, timedelta

from vms.common.clock import Clock
from vms.common.logs import get_logger
from vms.common.metrics import Metrics
from vms.drift.detect import Calibration, CameraDetector, DetectorSettings, channel_values
from vms.drift.settings import MonitorSettings
from vms.drift.signals import (
    SIGNALS,
    DetectionRow,
    DriftStatistics,
    FrameRow,
    Sample,
    SignalSettings,
    compare,
    coverage,
)
from vms.drift.store import DriftStore, ReferenceRange
from vms.storage.frames import FrameStoreError

log = get_logger(__name__)

UPDATE_BUCKETS_S = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


class DriftMonitor:
    def __init__(
        self,
        store: DriftStore,
        settings: MonitorSettings,
        signals: SignalSettings,
        metrics: Metrics,
        *,
        clock: Clock,
        detector: DetectorSettings | None = None,
        calibration: Calibration | None = None,
    ) -> None:
        self._store = store
        self._detector = detector if calibration is not None else None
        self._calibration = calibration
        self._detectors: dict[str, CameraDetector] = {}
        self._alarming: set[tuple[str, str]] = set()
        self._settings = settings
        self._signals = signals
        self._clock = clock
        self._cameras: set[str] = set()
        self._last_discover = float("-inf")
        # Loaded reference samples, keyed by camera, with the range they were loaded for.
        self._references: dict[str, tuple[ReferenceRange, Sample]] = {}

        labels = ["camera_id", "signal"]
        self._window_mean = metrics.gauge(
            "window_mean",
            "value",
            "Mean of the signal over the current window, in the signal's units",
            labels,
        )
        self._reference_mean = metrics.gauge(
            "reference_mean", "value", "Mean of the signal over the reference period", labels
        )
        self._reference_std = metrics.gauge(
            "reference_std",
            "value",
            "Per-frame (per-detection for confidence) std of the signal in the reference",
            labels,
        )
        self._shift = metrics.gauge(
            "shift",
            "value",
            "(window mean - reference mean) / reference std, std floored at min_std",
            labels,
        )
        self._ks = metrics.gauge(
            "confidence_ks", "value", "KS statistic, window vs reference confidences", ["camera_id"]
        )
        self._w1 = metrics.gauge(
            "confidence_wasserstein",
            "value",
            "Wasserstein-1 distance, window vs reference confidences",
            ["camera_id"],
        )
        self._psi = metrics.gauge(
            "class_psi", "value", "PSI of the class mix, window vs reference", ["camera_id"]
        )
        self._window_frames = metrics.gauge(
            "window", "frames", "Done frames in the current window", ["camera_id"]
        )
        self._window_detections = metrics.gauge(
            "window", "detections", "Detections (>= min_conf) in the current window", ["camera_id"]
        )
        self._reference_frames = metrics.gauge(
            "reference",
            "frames",
            "Frames in the camera's reference; 0 while the calibration period runs",
            ["camera_id"],
        )
        channel = ["camera_id", "channel"]
        self._z = metrics.gauge(
            "z", "value", "Channel statistic standardized by its clean-run calibration", channel
        )
        self._cusum = metrics.gauge(
            "cusum",
            "value",
            "Detector statistic (CUSUM S, larger side); 0 for a threshold",
            channel,
        )
        self._alarm = metrics.gauge(
            "alarm", "value", "1 while the channel's detector alarms, else 0", channel
        )
        self._alarms = metrics.counter(
            "alarms", "total", "Alarm onsets (0 to 1 transitions) per camera and channel", channel
        )
        self._updates = metrics.counter(
            "updates", "total", "Monitor updates by outcome", ["outcome"]
        )
        self._created = metrics.counter(
            "references_created", "total", "References started for new cameras"
        )
        self._duration = metrics.histogram(
            "update_duration", "seconds", "Time per monitor update", buckets=UPDATE_BUCKETS_S
        )

    async def run(self, stop: asyncio.Event) -> None:
        while not stop.is_set():
            t0 = self._clock.monotonic()
            try:
                await self.update()
            except FrameStoreError as exc:
                self._updates.labels(outcome="error").inc()
                log.warning("drift update failed", extra={"error": str(exc)})
            else:
                self._updates.labels(outcome="ok").inc()
            self._duration.observe(self._clock.monotonic() - t0)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(stop.wait(), timeout=self._settings.update_every_s)

    async def update(self) -> None:
        now = self._clock.now()
        if self._clock.monotonic() - self._last_discover >= self._settings.discover_every_s:
            self._cameras.update(await self._store.cameras())
            self._last_discover = self._clock.monotonic()
        refs = await self._store.references()
        self._cameras.update(refs)
        if not self._cameras:
            return
        cameras = sorted(self._cameras)
        end = now - timedelta(seconds=self._settings.settle_s)
        start = end - timedelta(seconds=self._settings.window_s)
        frames, detections = await self._read(cameras, start, end)

        for camera_id in cameras:
            window = Sample.build(
                frames.get(camera_id, []), detections.get(camera_id, []), self._signals
            )
            self._window_frames.labels(camera_id=camera_id).set(window.frames)
            self._window_detections.labels(camera_id=camera_id).set(window.detections)
            ref = refs.get(camera_id)
            if ref is None:
                if not window.frames:
                    continue
                ref = await self._start_reference(camera_id, frames[camera_id][0].ts)
            reference = await self._reference(ref, now)
            if reference is None:
                self._reference_frames.labels(camera_id=camera_id).set(0)
                self._export_window_only(camera_id, window)
                continue
            self._reference_frames.labels(camera_id=camera_id).set(reference.frames)
            full = (
                coverage(frames.get(camera_id, []), self._settings.window_s)
                >= self._settings.detect_min_coverage
            )
            self._export(camera_id, window, reference, full=full)

        for camera_id in set(self._references) - set(refs):  # deleted: start over next time
            del self._references[camera_id]

    async def _read(
        self, cameras: list[str], start: datetime, end: datetime
    ) -> tuple[dict[str, list[FrameRow]], dict[str, list[DetectionRow]]]:
        frames: dict[str, list[FrameRow]] = {}
        for f in await self._store.frames(cameras, start, end):
            frames.setdefault(f.camera_id, []).append(f)
        detections: dict[str, list[DetectionRow]] = {}
        for d in await self._store.detections(cameras, start, end, self._signals.min_conf):
            detections.setdefault(d.camera_id, []).append(d)
        return frames, detections

    async def _start_reference(self, camera_id: str, first_ts: datetime) -> ReferenceRange:
        ref = await self._store.create_reference(
            ReferenceRange(
                camera_id,
                first_ts,
                first_ts + timedelta(seconds=self._settings.reference_s),
                "auto",
            )
        )
        self._created.inc()
        log.info(
            "drift reference started",
            extra={
                "camera_id": camera_id,
                "start": ref.start.isoformat(),
                "end": ref.end.isoformat(),
            },
        )
        return ref

    async def _reference(self, ref: ReferenceRange, now: datetime) -> Sample | None:
        """The reference sample, loaded once its period has ended and settled; None before."""
        cached = self._references.get(ref.camera_id)
        if cached is not None and cached[0] == ref:
            return cached[1]
        if ref.end + timedelta(seconds=self._settings.settle_s) > now:
            return None
        frames, detections = await self._read([ref.camera_id], ref.start, ref.end)
        sample = Sample.build(
            frames.get(ref.camera_id, []), detections.get(ref.camera_id, []), self._signals
        )
        if sample.frames < self._signals.min_frames:
            log.warning(
                "drift reference has too few frames; statistics stay undefined until recalibrated",
                extra={"camera_id": ref.camera_id, "frames": sample.frames},
            )
        self._references[ref.camera_id] = (ref, sample)
        self._detectors.pop(ref.camera_id, None)  # a new reference: detection starts over
        log.info(
            "drift reference loaded",
            extra={
                "camera_id": ref.camera_id,
                "frames": sample.frames,
                "detections": sample.detections,
                "reason": ref.reason,
            },
        )
        return sample

    def _export_window_only(self, camera_id: str, window: Sample) -> None:
        for signal in SIGNALS:
            self._window_mean.labels(camera_id=camera_id, signal=signal).set(window.mean(signal))

    def _export(self, camera_id: str, window: Sample, reference: Sample, *, full: bool) -> None:
        stats = compare(window, reference, self._signals)
        for signal in SIGNALS:
            labels = {"camera_id": camera_id, "signal": signal}
            self._window_mean.labels(**labels).set(stats.window_mean.get(signal, math.nan))
            self._reference_mean.labels(**labels).set(reference.mean(signal))
            self._reference_std.labels(**labels).set(reference.std(signal))
            self._shift.labels(**labels).set(stats.shift.get(signal, math.nan))
        self._ks.labels(camera_id=camera_id).set(stats.confidence_ks)
        self._w1.labels(camera_id=camera_id).set(stats.confidence_wasserstein)
        self._psi.labels(camera_id=camera_id).set(stats.class_psi)
        self._detect(camera_id, stats, full)

    def _detect(self, camera_id: str, stats: DriftStatistics, full: bool) -> None:
        """A partly filled window (`full` false) counts as undefined: the state is kept and
        nothing alarms."""
        if self._detector is None or self._calibration is None:
            return
        det = self._detectors.get(camera_id)
        if det is None:
            det = self._detectors[camera_id] = CameraDetector(self._detector, self._calibration)
        values = channel_values(stats)
        if not full:
            values = dict.fromkeys(values, math.nan)
        firing = set(det.update(values))
        for channel, cd in det.channels.items():
            labels = {"camera_id": camera_id, "channel": channel}
            self._z.labels(**labels).set(self._calibration.z(channel, values[channel]))
            self._cusum.labels(**labels).set(cd.statistic)
            self._alarm.labels(**labels).set(1 if channel in firing else 0)
            key = (camera_id, channel)
            was = key in self._alarming
            if channel in firing:
                self._alarming.add(key)
            else:
                self._alarming.discard(key)
            if channel in firing and not was:
                self._alarms.labels(**labels).inc()
                log.warning(
                    "drift alarm",
                    extra={"camera_id": camera_id, "channel": channel, "z": values[channel]},
                )
