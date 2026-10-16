"""Drift injection: event validation, seeded onsets, intensity ramps, and the runner sending
transformed or switched frames from the onset on (virtual time, inline render)."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from concurrent.futures import Executor, Future
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from tests.unit.loadgen.conftest import ScriptedTransport, fake_clip
from vms.common.clock import FakeClock
from vms.datasets.clips import decode_jpeg, encode_jpeg
from vms.loadgen.drift import (
    ONSET_TAG,
    DriftEvent,
    DriftInjectionSettings,
    active_clip,
    active_transforms,
    schedule_drift,
)
from vms.loadgen.frames import LoadedClip
from vms.loadgen.recorder import Recorder
from vms.loadgen.runner import Runner
from vms.loadgen.schedule import CameraPlan
from vms.loadgen.settings import LoadgenSettings

VRun = Callable[[Coroutine[Any, Any, Any]], Any]
START = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
PLANS = [CameraPlan(f"cam-00{i}", "real", 0, 0.0) for i in (1, 2, 3)]


class InlineExecutor(Executor):
    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        future: Future[Any] = Future()
        future.set_result(fn(*args, **kwargs))
        return future


def real_clip(n: int = 4) -> LoadedClip:
    frames = []
    for i in range(n):
        image = np.full((48, 64, 3), 150, dtype=np.uint8)
        image[10:30, 5 + i : 25 + i] = 30
        frames.append(encode_jpeg(image, 90))
    return LoadedClip("real", 5.0, 64, 48, tuple(frames))


def event(**kw: Any) -> DriftEvent:
    values: dict[str, Any] = {"kind": "darken", "cameras": [0], "onset_s": 1.0, "strength": 0.3}
    values.update(kw)
    return DriftEvent(**values)


@pytest.mark.parametrize(
    ("kw", "match"),
    [
        ({"strength": None}, "strength"),
        ({"strength": 1.5}, "out of range"),
        ({"kind": "switch_clip", "strength": None}, "clip"),
        ({"kind": "switch_clip", "strength": None, "clip": "b", "ramp_s": 5}, "abrupt"),
        ({"clip": "b"}, "no clip"),
        ({"cameras": [0, 0]}, "distinct"),
        ({"cameras": []}, "at least 1"),
    ],
)
def test_event_validation(kw: dict[str, Any], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        event(**kw)


def test_onsets_are_seeded_and_jittered_within_range() -> None:
    settings = DriftInjectionSettings(
        events=[event(cameras=[0, 2], onset_s=60, onset_jitter_s=30, ramp_s=10)]
    )
    a = schedule_drift(settings, PLANS, seed=1)
    assert sorted(a) == ["cam-001", "cam-003"]
    onsets = [a[c][0].onset_s for c in ("cam-001", "cam-003")]
    assert all(60 <= o < 90 for o in onsets)
    assert onsets[0] != onsets[1]
    assert schedule_drift(settings, PLANS, seed=1) == a
    assert schedule_drift(settings, PLANS, seed=2) != a
    with pytest.raises(ValueError, match="has 3"):
        schedule_drift(DriftInjectionSettings(events=[event(cameras=[3])]), PLANS, seed=1)


def test_intensity_ramps_and_records_the_onset() -> None:
    settings = DriftInjectionSettings(events=[event(ramp_s=10), event(kind="blur", strength=2)])
    gradual, abrupt = schedule_drift(settings, PLANS, seed=1)["cam-001"]
    assert [gradual.intensity(t) for t in (0.5, 1.0, 6.0, 11.0, 50.0)] == [0, 0, 0.5, 1, 1]
    assert [abrupt.intensity(t) for t in (0.99, 1.0)] == [0, 1]
    record = gradual.record(START)
    assert record["onset_at"] == "2026-10-16T12:00:01+00:00"
    assert record["kind"] == "darken"
    assert record["ramp_s"] == 10
    assert [e.kind for e, _ in active_transforms([gradual, abrupt], 6.0)] == ["darken", "blur"]
    assert active_transforms([gradual, abrupt], 0.5) == []


def test_latest_switch_wins() -> None:
    settings = DriftInjectionSettings(
        events=[
            event(kind="switch_clip", strength=None, clip="b", onset_s=1),
            event(kind="switch_clip", strength=None, clip="c", onset_s=2),
        ]
    )
    events = schedule_drift(settings, PLANS, seed=1)["cam-001"]
    assert [active_clip(events, t) for t in (0.5, 1.5, 2.5)] == [None, "b", "c"]


def run(
    vrun: VRun, recorder: Recorder, drift: DriftInjectionSettings, clips: dict[str, LoadedClip]
) -> tuple[ScriptedTransport, Runner]:
    s = LoadgenSettings(fps=5.0, duration_s=2.0, warmup_s=0.0, cameras=2, drift=drift)
    plans = PLANS[:2]
    transport = ScriptedTransport()
    runner = Runner(
        s,
        plans,
        clips,
        transport,
        recorder,
        clock=FakeClock(START),
        run_id="r",
        drift=schedule_drift(drift, plans, s.seed),
        render_executor=InlineExecutor(),
    )
    vrun(runner.run())
    return transport, runner


def test_runner_darkens_only_the_drifting_camera_after_onset(
    vrun: VRun, recorder: Recorder
) -> None:
    clip = real_clip()
    drift = DriftInjectionSettings(events=[event(onset_s=1.0, strength=0.3)])
    transport, runner = run(vrun, recorder, drift, {"real": clip})
    frames = {(f.camera_id, f.seq): f for f in transport.frames}
    for k in range(10):
        clean = frames[("cam-002", k)]
        assert clean.jpeg in clip.frames
        drifted = frames[("cam-001", k)]
        if k < 5:  # due at k / 5 s, before the onset at 1 s
            assert drifted.jpeg == clean.jpeg
        else:
            ratio = decode_jpeg(drifted.jpeg).mean() / decode_jpeg(clean.jpeg).mean()
            assert ratio == pytest.approx(0.3, abs=0.03)
    assert len(runner.render_s) == 5
    assert recorder.summary(2.0, 10.0)["frames"]["accepted"] == 20


def test_runner_switches_clips_at_onset(vrun: VRun, recorder: Recorder) -> None:
    other = fake_clip("other", 6, fps=5.0)
    drift = DriftInjectionSettings(
        events=[event(kind="switch_clip", strength=None, clip="other", onset_s=1.0)]
    )
    transport, runner = run(vrun, recorder, drift, {"real": real_clip(), "other": other})
    sent = [
        f.jpeg for f in sorted(transport.frames, key=lambda f: f.seq) if f.camera_id == "cam-001"
    ]
    assert all(j not in other.frames for j in sent[:5])
    assert sent[5:] == [other.frames[k % 6] for k in range(5, 10)]
    assert runner.render_s == []  # no pixel transform: frames are sent as stored


def test_a_failed_render_counts_as_a_failed_send(vrun: VRun, recorder: Recorder) -> None:
    bad = LoadedClip("real", 5.0, 64, 48, (b"not a jpeg",))
    drift = DriftInjectionSettings(events=[event(onset_s=0.0)])
    run(vrun, recorder, drift, {"real": bad})
    frames = recorder.summary(2.0, 10.0)["frames"]
    assert frames["failed"] == 10  # cam-001: every frame drifted and failed
    assert frames["accepted"] == 10  # cam-002 unaffected


def test_onset_annotation_is_a_point_if_abrupt_and_the_ramp_if_gradual() -> None:
    settings = DriftInjectionSettings(
        events=[
            event(cameras=[0], onset_s=10, strength=0.3),
            event(cameras=[1], onset_s=20, ramp_s=120, strength=0.3),
            event(kind="switch_clip", cameras=[2], strength=None, clip="b", onset_s=30),
        ]
    )
    drift = schedule_drift(settings, PLANS, seed=1)
    abrupt = drift["cam-001"][0].annotation(START, "run-r0")
    assert abrupt.time == START + timedelta(seconds=10)
    assert abrupt.time_end is None
    assert abrupt.tags == (ONSET_TAG, "darken", "cam-001")
    assert abrupt.text == "Injected darken on cam-001 (strength 0.3, abrupt); run run-r0"
    gradual = drift["cam-002"][0].annotation(START, "run-r0")
    assert gradual.time_end is not None
    assert gradual.time_end - gradual.time == timedelta(seconds=120)
    assert "ramp 120 s" in gradual.text
    switch = drift["cam-003"][0].annotation(START, "run-r0")
    assert "clip b" in switch.text


def test_runner_reports_its_time_zero_to_on_start(vrun: VRun, recorder: Recorder) -> None:
    started: list[datetime] = []
    s = LoadgenSettings(fps=5.0, duration_s=1.0, warmup_s=0.0, cameras=1)
    runner = Runner(
        s,
        PLANS[:1],
        {"real": real_clip()},
        ScriptedTransport(),
        recorder,
        clock=FakeClock(START),
        run_id="r",
        on_start=started.append,
    )
    vrun(runner.run())
    assert started == [START]
