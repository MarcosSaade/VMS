"""Each drift signal against a reference: the statistic that should move moves, the others stay
near zero, and undefined statistics are NaN."""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from tests.unit.drift.conftest import T0, Scene, stream
from vms.drift.signals import (
    INPUT_SIGNALS,
    DetectionRow,
    DriftStatistics,
    FrameRow,
    Sample,
    SignalSettings,
    compare,
)

SETTINGS = SignalSettings()
CLEAN = Scene()


def sample(scene: Scene, seed: int, seconds: float = 30.0) -> Sample:
    frames, detections = stream("cam", T0, seconds, scene, np.random.default_rng(seed))
    return Sample.build(frames, detections, SETTINGS)


def stats_for(scene: Scene, seed: int = 2) -> DriftStatistics:
    return compare(sample(scene, seed), sample(CLEAN, 1, seconds=60), SETTINGS)


def test_clean_window_shows_no_drift() -> None:
    stats = stats_for(CLEAN)
    assert stats.window_frames == 150
    assert stats.window_detections == 600
    for signal, value in stats.shift.items():
        assert abs(value) < 0.5, signal
    assert stats.confidence_ks < 0.1
    assert stats.confidence_wasserstein < 0.02
    assert stats.class_psi < 0.01


def test_darkening_moves_brightness_only() -> None:
    stats = stats_for(replace(CLEAN, brightness=100.0))  # 20 levels darker, reference std 2
    assert stats.shift["brightness"] == pytest.approx(-10.0, abs=1.0)
    assert stats.window_mean["brightness"] == pytest.approx(100.0, abs=0.5)
    for signal in ("contrast", "log_blur", "noise", "detections_per_frame", "confidence"):
        assert abs(stats.shift[signal]) < 0.5, signal


@pytest.mark.parametrize(
    ("change", "signal", "direction"),
    [
        ({"contrast": 40.0}, "contrast", -1),
        ({"blur_var": 40.0}, "log_blur", -1),  # ln(10) / 0.1 std: about -23
        ({"noise": 6.0}, "noise", 1),
        ({"detections": 1}, "detections_per_frame", -1),  # occlusion: fewer boxes
    ],
)
def test_each_input_drift_moves_its_own_signal(
    change: dict[str, float], signal: str, direction: int
) -> None:
    stats = stats_for(replace(CLEAN, **change))  # type: ignore[arg-type]
    assert direction * stats.shift[signal] > 3.0
    others = set(INPUT_SIGNALS) - {signal}
    assert all(abs(stats.shift[s]) < 0.5 for s in others)


def test_confidence_drop_moves_its_distribution_statistics() -> None:
    stats = stats_for(replace(CLEAN, confidence=0.5))
    assert stats.shift["confidence"] == pytest.approx(-2.5, abs=0.4)  # 0.2 / 0.08
    assert stats.confidence_ks > 0.7
    assert stats.confidence_wasserstein == pytest.approx(0.2, abs=0.02)
    assert stats.class_psi < 0.01


def test_class_mix_change_moves_psi_not_inputs() -> None:
    stats = stats_for(replace(CLEAN, classes=("car", "truck", "bus")))
    assert stats.class_psi > 1.0
    assert all(abs(stats.shift[s]) < 0.5 for s in INPUT_SIGNALS)


def test_too_few_frames_leaves_statistics_undefined() -> None:
    stats = compare(sample(CLEAN, 2, seconds=2), sample(CLEAN, 1, seconds=60), SETTINGS)
    assert stats.window_frames == 10
    assert all(math.isnan(v) for v in stats.shift.values())
    assert math.isnan(stats.class_psi)
    assert stats.window_mean["brightness"] == pytest.approx(120.0, abs=2.0)  # still reported


def test_too_few_detections_leaves_detection_statistics_undefined() -> None:
    stats = stats_for(replace(CLEAN, detections=0))
    assert stats.shift["detections_per_frame"] < -3.0  # 0 boxes per frame is a real signal
    assert math.isnan(stats.shift["confidence"])
    assert math.isnan(stats.confidence_ks)
    assert math.isnan(stats.class_psi)
    assert abs(stats.shift["brightness"]) < 0.5


def test_build_filters_detections_and_bins_classes() -> None:
    frames = [
        FrameRow("cam", "f1", T0, 100.0, 10.0, 0.0, None),
        FrameRow("cam", "f2", T0, 110.0, 10.0, math.e - 1, None),
    ]
    detections = [
        DetectionRow("cam", "f1", "person", 0.9),
        DetectionRow("cam", "f1", "dog", 0.8),  # not a bin: "other"
        DetectionRow("cam", "f2", "car", 0.1),  # below min_conf
        DetectionRow("cam", "gone", "car", 0.9),  # frame not in the sample
    ]
    s = Sample.build(frames, detections, SETTINGS)
    assert s.frames == 2
    assert s.detections == 2
    assert list(s.values["detections_per_frame"]) == [2.0, 0.0]
    bins = dict(zip(SETTINGS.bins, s.class_counts, strict=True))
    assert bins["person"] == 1
    assert bins["other"] == 1
    assert bins["car"] == 0
    assert s.mean("brightness") == pytest.approx(105.0)
    assert s.mean("log_blur") == pytest.approx(0.5)  # ln(1 + 0) and ln(1 + e - 1)
    assert math.isnan(s.mean("noise"))  # frames from before migration 0007


def test_settings_need_a_floor_for_every_signal() -> None:
    with pytest.raises(ValueError, match="min_std"):
        SignalSettings(min_std={"brightness": 1.0})
    with pytest.raises(ValueError, match="other"):
        SignalSettings(classes=["person", "other"])
