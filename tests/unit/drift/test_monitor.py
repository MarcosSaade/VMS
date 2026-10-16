"""The drift monitor: references are started, loaded once, replaced on recalibration, and the
gauges follow an injected shift."""

from __future__ import annotations

import asyncio
import math
from dataclasses import replace
from datetime import timedelta

import numpy as np
import pytest

from tests.unit.drift.conftest import T0, FakeDriftStore, Scene, stream
from vms.common.clock import FakeClock
from vms.common.metrics import Metrics
from vms.drift.cli import recalibrate
from vms.drift.monitor import DriftMonitor
from vms.drift.settings import DriftSettings, MonitorSettings
from vms.drift.signals import SignalSettings
from vms.drift.store import ReferenceRange
from vms.storage.frames import FrameStoreError

SETTINGS = MonitorSettings(window_s=30, reference_s=60, update_every_s=5, settle_s=3)


class Harness:
    def __init__(self, store: FakeDriftStore | None = None) -> None:
        self.store = store or FakeDriftStore()
        self.clock = FakeClock(T0)
        self.metrics = Metrics("drift")
        self.monitor = DriftMonitor(
            self.store, SETTINGS, SignalSettings(), self.metrics, clock=self.clock
        )

    def at(self, seconds: float) -> None:
        self.clock.set(T0 + timedelta(seconds=seconds))

    def update(self) -> None:
        asyncio.run(self.monitor.update())

    def sample(self, name: str, **labels: str) -> float | None:
        return self.metrics.registry.get_sample_value(f"vms_drift_{name}", labels)


def clean_then_dark(seed: int = 7) -> FakeDriftStore:
    """cam-1: clean for 120 s, then 40 levels darker for 60 s."""
    rng = np.random.default_rng(seed)
    store = FakeDriftStore()
    store.add(stream("cam-1", T0, 120, Scene(), rng))
    store.add(stream("cam-1", T0 + timedelta(seconds=120), 60, Scene(brightness=80.0), rng))
    return store


def test_new_camera_gets_a_reference_from_its_first_frames() -> None:
    h = Harness(clean_then_dark())
    h.at(20)  # window [-13 s, 17 s): frames from T0
    h.update()
    ref = h.store.refs["cam-1"]
    assert (ref.start, ref.end, ref.reason) == (T0, T0 + timedelta(seconds=60), "auto")
    assert h.sample("references_created_total") == 1
    assert h.sample("reference_frames", camera_id="cam-1") == 0  # still calibrating
    assert h.sample("window_frames", camera_id="cam-1") == 85
    assert h.sample("window_mean_value", camera_id="cam-1", signal="brightness") == (
        pytest.approx(120, abs=1)
    )
    assert h.sample("shift_value", camera_id="cam-1", signal="brightness") is None


def test_statistics_appear_after_the_reference_settles_and_follow_the_drift() -> None:
    h = Harness(clean_then_dark())
    h.at(20)
    h.update()
    h.at(62)  # reference ends at 60 s, plus settle_s 3: not yet
    h.update()
    assert h.sample("reference_frames", camera_id="cam-1") == 0

    h.at(100)  # window [67 s, 97 s): clean
    h.update()
    assert h.sample("reference_frames", camera_id="cam-1") == 300
    assert h.sample("reference_mean_value", camera_id="cam-1", signal="brightness") == (
        pytest.approx(120, abs=0.5)
    )
    clean = h.sample("shift_value", camera_id="cam-1", signal="brightness")
    assert clean is not None
    assert abs(clean) < 0.5
    psi = h.sample("class_psi_value", camera_id="cam-1")
    assert psi is not None
    assert psi < 0.01

    h.at(170)  # window [137 s, 167 s): dark since 120 s
    h.update()
    dark = h.sample("shift_value", camera_id="cam-1", signal="brightness")
    assert dark == pytest.approx(-20, abs=2)  # 40 levels, reference std 2
    noise = h.sample("shift_value", camera_id="cam-1", signal="noise")
    assert noise is not None
    assert abs(noise) < 0.5


def test_reference_is_loaded_once_and_reloaded_after_recalibration() -> None:
    h = Harness(clean_then_dark())
    h.at(20)
    h.update()
    h.at(100)
    h.update()
    reads = h.store.reads
    h.at(105)
    h.update()
    assert h.store.reads == reads + 1  # the window only

    # Recalibrate onto the dark period: it becomes the new normal.
    settings = DriftSettings()
    asyncio.run(
        recalibrate(
            h.store,
            ["cam-1"],
            settings,
            h.clock,
            start=T0 + timedelta(seconds=120),
            end=T0 + timedelta(seconds=180),
        )
    )
    assert h.store.refs["cam-1"].reason == "recalibrate"
    h.at(185)
    h.update()
    assert h.store.reads == reads + 3  # window and the new reference
    assert h.sample("reference_mean_value", camera_id="cam-1", signal="brightness") == (
        pytest.approx(80, abs=0.5)
    )
    shift = h.sample("shift_value", camera_id="cam-1", signal="brightness")
    assert shift is not None
    assert abs(shift) < 0.5


def test_recalibrate_now_starts_a_new_period() -> None:
    store = FakeDriftStore()
    clock = FakeClock(T0)
    refs = asyncio.run(recalibrate(store, ["a", "b"], DriftSettings(), clock))
    assert [r.camera_id for r in refs] == ["a", "b"]
    assert store.refs["a"] == ReferenceRange("a", T0, T0 + timedelta(seconds=60), "recalibrate")
    with pytest.raises(ValueError, match="end after"):
        asyncio.run(recalibrate(store, ["a"], DriftSettings(), clock, start=T0, end=T0))


def test_camera_without_frames_in_the_window_reports_undefined_statistics() -> None:
    h = Harness(clean_then_dark())
    h.at(20)
    h.update()
    h.at(100)
    h.update()
    h.at(400)  # the camera stopped at 180 s
    h.update()
    assert h.sample("window_frames", camera_id="cam-1") == 0
    shift = h.sample("shift_value", camera_id="cam-1", signal="brightness")
    assert shift is not None
    assert math.isnan(shift)


def test_a_short_reference_leaves_statistics_undefined() -> None:
    store = FakeDriftStore()
    store.add(stream("cam-1", T0, 2, Scene(), np.random.default_rng(1)))  # 10 frames
    store.add(stream("cam-1", T0 + timedelta(seconds=70), 30, Scene(), np.random.default_rng(2)))
    h = Harness(store)
    h.at(5)
    h.update()
    h.at(100)
    h.update()
    assert h.sample("reference_frames", camera_id="cam-1") == 10
    shift = h.sample("shift_value", camera_id="cam-1", signal="brightness")
    assert shift is not None
    assert math.isnan(shift)


def test_cameras_are_independent() -> None:
    rng = np.random.default_rng(3)
    store = FakeDriftStore()
    store.add(stream("cam-1", T0, 180, Scene(), rng))
    store.add(stream("cam-2", T0, 120, Scene(), rng))
    store.add(
        stream("cam-2", T0 + timedelta(seconds=120), 60, replace(Scene(), blur_var=20.0), rng)
    )
    h = Harness(store)
    for t in (20, 100, 170):
        h.at(t)
        h.update()
    one = h.sample("shift_value", camera_id="cam-1", signal="log_blur")
    two = h.sample("shift_value", camera_id="cam-2", signal="log_blur")
    assert one is not None
    assert abs(one) < 0.5
    assert two is not None
    assert two < -10


def test_run_counts_failed_updates() -> None:
    h = Harness(FakeDriftStore(fail=True))
    stop = asyncio.Event()

    async def failing_cameras() -> list[str]:
        stop.set()  # one iteration
        raise FrameStoreError("down")

    h.store.cameras = failing_cameras  # type: ignore[method-assign]
    asyncio.run(h.monitor.run(stop))
    assert h.sample("updates_total", outcome="error") == 1
    assert h.sample("update_duration_seconds_count") == 1


def test_update_every_must_fit_in_the_window() -> None:
    with pytest.raises(ValueError, match="update_every_s"):
        MonitorSettings(window_s=5, update_every_s=10)


def test_detector_alarms_on_the_drift_and_counts_one_onset() -> None:
    from vms.drift.detect import Calibration, ChannelCalibration, DetectorSettings

    store = clean_then_dark()
    clock = FakeClock(T0)
    metrics = Metrics("drift")
    cal = Calibration(channels={"shift:brightness": ChannelCalibration(mean=0.0, std=0.5)})
    monitor = DriftMonitor(
        store,
        SETTINGS,
        SignalSettings(),
        metrics,
        clock=clock,
        detector=DetectorSettings(kind="cusum", k=1, h=10),
        calibration=cal,
    )

    def at(seconds: float) -> None:
        clock.set(T0 + timedelta(seconds=seconds))
        asyncio.run(monitor.update())

    def sample(name: str) -> float | None:
        return metrics.registry.get_sample_value(
            f"vms_drift_{name}", {"camera_id": "cam-1", "channel": "shift:brightness"}
        )

    at(20)
    for t in range(100, 125, 5):  # clean windows
        at(t)
    assert sample("alarm_value") == 0
    assert sample("cusum_value") == 0
    for t in range(155, 180, 5):  # dark from 120 s: z about -40 per update
        at(t)
    assert sample("alarm_value") == 1
    assert sample("alarms_total") == 1  # one onset, however many updates alarm
    z = sample("z_value")
    assert z is not None
    assert z < -20
