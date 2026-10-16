"""Change detectors on synthetic streams with known change points."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vms.drift.detect import (
    CHANNELS,
    Calibration,
    CameraDetector,
    ChannelCalibration,
    ChannelDetector,
    DetectorSettings,
)


def first_alarm(det: ChannelDetector, zs: list[float]) -> int | None:
    for i, z in enumerate(zs):
        if det.update(z):
            return i
    return None


def test_threshold_is_memoryless_and_two_sided() -> None:
    det = ChannelDetector(DetectorSettings(kind="threshold", h=3), two_sided=True)
    assert first_alarm(det, [1, 2, 2.9, -2.9]) is None
    assert det.update(-3.1)
    one = ChannelDetector(DetectorSettings(kind="threshold", h=3), two_sided=False)
    assert not one.update(-10)  # a drop of KS or PSI is not drift
    assert one.update(10)


def test_cusum_catches_a_small_persistent_shift_long_before_a_threshold() -> None:
    rng = np.random.default_rng(0)
    clean = list(rng.normal(0, 1, 200))
    shifted = list(rng.normal(1.5, 1, 200))  # change point at 200
    cusum = ChannelDetector(DetectorSettings(kind="cusum", k=0.5, h=8), two_sided=True)
    threshold = ChannelDetector(DetectorSettings(kind="threshold", h=4), two_sided=True)
    t_cusum = first_alarm(cusum, clean + shifted)
    assert t_cusum is not None
    assert 200 <= t_cusum < 200 + 30  # about h / (d - k) = 8 updates, plus noise
    # The threshold (no false alarm on the clean part either) waits for one lucky window.
    t_threshold = first_alarm(threshold, clean + shifted)
    assert t_threshold is None or t_threshold - 200 > 2 * (t_cusum - 200)


def test_cusum_delay_follows_h_over_d_minus_k() -> None:
    det = ChannelDetector(DetectorSettings(kind="cusum", k=1, h=10, cap=100), two_sided=False)
    assert first_alarm(det, [3.0] * 20) == 5  # S = 2 per update; > 10 after 6 updates


def test_cusum_drains_on_clean_data_and_is_capped() -> None:
    det = ChannelDetector(DetectorSettings(kind="cusum", k=1, h=5, cap=2), two_sided=True)
    for _ in range(100):
        det.update(-20.0)
    assert det.statistic == 10  # capped at cap * h
    alarms = [det.update(0.0) for _ in range(10)]
    # Drains by k per update: alarm while S > h, then clears.
    assert alarms == [True] * 4 + [False] * 6
    det.reset()
    assert det.statistic == 0


def test_nan_keeps_the_state() -> None:
    det = ChannelDetector(DetectorSettings(kind="cusum", k=0, h=5), two_sided=False)
    det.update(4.0)
    assert not det.update(math.nan)
    assert det.statistic == 4.0
    assert det.update(2.0)


def test_calibration_standardizes_and_floors_std() -> None:
    cal = Calibration.fit(
        {"shift:brightness": [0.0, 0.0, 0.0], "class_psi": [0.01, 0.03], "empty": [math.nan]},
        {"shift:brightness": 0.1},
    )
    assert cal.channels["shift:brightness"].std == 0.1
    assert cal.z("shift:brightness", 0.5) == pytest.approx(5.0)
    assert cal.z("class_psi", 0.02) == pytest.approx(0.0)
    assert "empty" not in cal.channels
    assert math.isnan(cal.z("missing", 1.0))


def test_camera_alarms_when_any_channel_does() -> None:
    cal = Calibration(
        channels={
            "shift:brightness": ChannelCalibration(mean=0, std=1),
            "class_psi": ChannelCalibration(mean=0.01, std=0.01),
        }
    )
    cam = CameraDetector(DetectorSettings(kind="threshold", h=3), cal)
    assert cam.update({"shift:brightness": 1.0, "class_psi": 0.02}) == []
    assert cam.update({"shift:brightness": -5.0, "class_psi": 0.5}) == [
        "shift:brightness",
        "class_psi",
    ]
    # A channel without calibration is not watched.
    only = CameraDetector(DetectorSettings(kind="threshold", h=3, channels=["shift:noise"]), cal)
    assert only.channels == {}


def test_shipped_config_calibrates_every_channel_of_the_deployed_detector() -> None:
    from vms.common.settings import load_settings
    from vms.drift.settings import DriftSettings

    settings = load_settings(DriftSettings, "drift")
    assert settings.calibration is not None  # alarms on (task 9.3, D34)
    assert set(settings.detector.channels) == set(CHANNELS)
    assert set(settings.calibration.channels) == set(CHANNELS)
