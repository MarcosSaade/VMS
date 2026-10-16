"""Offline drift scoring: replay matches the monitor's windows, delays and false alarms are
counted against known onsets, and operating points are chosen out of sample."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from tests.unit.drift.conftest import T0, Scene, stream
from vms.drift.detect import Calibration, ChannelCalibration, DetectorSettings
from vms.drift.score import (
    ScoreScenario,
    Stream,
    SweepSettings,
    camera_series,
    choose,
    plot,
    score,
    score_stream,
    summarize,
)
from vms.drift.settings import MonitorSettings
from vms.drift.signals import SignalSettings

MONITOR = MonitorSettings(window_s=30, reference_s=60, update_every_s=5, settle_s=3)
CAL = Calibration(channels={"shift:brightness": ChannelCalibration(mean=0.0, std=1.0)})
ONLY_BRIGHTNESS = ["shift:brightness"]


def synthetic(
    rep: int, kind: str | None, onset: float | None, rng: np.random.Generator, step: float = 6.0
) -> Stream:
    s = Stream("s", rep, 100 + rep, f"cam-{kind}-{rep}", "clip", kind, 0.0, onset)
    for i in range(60):
        t = 90.0 + 5 * i
        z = float(rng.normal(0, 1)) + (step if onset is not None and t >= onset else 0.0)
        s.times_s.append(t)
        s.values.append({"shift:brightness": z})
    return s


def test_score_stream_counts_false_alarms_delay_and_misses() -> None:
    cfg = DetectorSettings(kind="threshold", h=3, channels=ONLY_BRIGHTNESS)
    s = Stream("s", 0, 1, "c", "x", "darken", 0.0, 120.0)
    s.times_s = [90, 95, 100, 105, 110, 115, 120, 125, 130]
    s.values = [{"shift:brightness": v} for v in (0, 4, 0, 0, 0, 0, 1, 5, 9)]
    out = score_stream(s, cfg, CAL)
    assert out.false_alarms == 1  # at 95 s, before the onset
    assert out.delay_s == pytest.approx(5.0)  # first alarm at 125 s
    assert out.channels == ["shift:brightness"]
    assert out.clean_s == pytest.approx(30.0)
    missed = score_stream(replace(s, values=[{"shift:brightness": 0}] * 9), cfg, CAL)
    assert missed.missed
    assert missed.delay_s is None
    clean = score_stream(replace(s, kind=None, onset_s=None), cfg, CAL)
    assert clean.false_alarms == 3  # 95, 125, 130: reset after each
    assert clean.clean_s == pytest.approx(40.0)


def test_summary_rate_and_rule_of_three() -> None:
    cfg = DetectorSettings(kind="threshold", h=100, channels=ONLY_BRIGHTNESS)
    s = Stream("s", 0, 1, "c", "x", None, 0.0, None, [0.0, 3600.0], [{}, {}])
    summary = summarize([score_stream(s, cfg, CAL)])
    assert summary.fa_per_hour == 0
    assert summary.fa_upper_95 == pytest.approx(3.0)


def test_choose_prefers_no_false_alarm_then_fewest_misses_then_speed() -> None:
    rng = np.random.default_rng(1)
    streams = [synthetic(0, None, None, rng) for _ in range(4)] + [
        synthetic(0, "darken", 150.0, rng) for _ in range(4)
    ]
    configs = [
        DetectorSettings(kind="threshold", h=h, channels=ONLY_BRIGHTNESS) for h in (2, 4.5, 20)
    ]
    chosen = choose(streams, configs, CAL)
    assert chosen is not None
    assert chosen.h == 4.5  # h=2 false-alarms on the noise; h=20 misses the step of 6


def test_score_holds_each_repetition_out() -> None:
    rng = np.random.default_rng(2)
    streams = []
    for rep in range(3):
        streams += [synthetic(rep, None, None, rng) for _ in range(3)]
        streams += [synthetic(rep, "darken", 150.0 + 10 * rep, rng) for _ in range(2)]
    scenario = ScoreScenario(
        name="t",
        runs=[Path("x.json")],
        sweep=SweepSettings(threshold_h=[3, 5, 8], cusum_k=[1], cusum_h=[5, 10, 20]),
        min_std={"shift:brightness": 0.1},
    )
    metrics = score(streams, scenario)
    assert [f["held_out_rep"] for f in metrics["folds"]] == [0, 1, 2]
    oos = metrics["out_of_sample"]
    assert set(oos) == {"threshold", "cusum"}
    assert oos["cusum"]["drifted"] == 6
    assert oos["cusum"]["missed"] == 0
    assert oos["cusum"]["by_drift"]["darken/abrupt"]["streams"] == 6
    assert len(metrics["sweep"]) == 6
    assert metrics["streams"]["drifted"] == 6
    json.dumps(metrics)  # serializable


def test_camera_series_replays_the_monitor_windows() -> None:
    rng = np.random.default_rng(3)
    frames, dets = stream("cam", T0, 120, Scene(), rng)
    f2, d2 = stream("cam", T0 + timedelta(seconds=120), 60, replace(Scene(), brightness=80.0), rng)
    times, values = camera_series(frames + f2, dets + d2, T0, MONITOR, SignalSettings())
    assert times[0] == pytest.approx(90.0)  # reference [0, 60) s, first window [60, 90)
    assert times[1] - times[0] == pytest.approx(5.0)
    assert times[-1] <= 180.0
    before = values[times.index(110.0)]["shift:brightness"]
    after = values[times.index(155.0)]["shift:brightness"]
    assert abs(before) < 0.5
    assert after < -15
    assert camera_series([], [], T0, MONITOR, SignalSettings()) == ([], [])


def test_plot_writes_a_png(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    streams = [synthetic(r, None, None, rng) for r in range(2)] + [
        synthetic(r, "darken", 150.0, rng) for r in range(2)
    ]
    scenario = ScoreScenario(
        name="t",
        runs=[Path("x.json")],
        sweep=SweepSettings(threshold_h=[4, 8], cusum_k=[1], cusum_h=[10]),
    )
    out = tmp_path / "p.png"
    plot(score(streams, scenario), out)
    assert out.read_bytes()[:4] == b"\x89PNG"
