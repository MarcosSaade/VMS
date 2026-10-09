"""Runner pacing on a virtual-time loop: exact, instant and deterministic."""

import asyncio
from collections import defaultdict
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import Any

import pytest

from tests.unit.loadgen.conftest import ScriptedTransport, fake_clip
from vms.common.clock import FakeClock
from vms.common.metrics import Metrics
from vms.loadgen.recorder import Recorder
from vms.loadgen.runner import Runner, RunTiming
from vms.loadgen.schedule import CameraPlan, clip_frame, plan_cameras
from vms.loadgen.settings import HttpSettings, LoadgenSettings
from vms.loadgen.transport import Outcome, OutgoingFrame, SendResult, Transport, from_status

VRun = Callable[[Coroutine[Any, Any, Any]], Any]
START = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
CLIPS = {"clip-a": fake_clip("clip-a", 12), "clip-b": fake_clip("clip-b", 7, fps=25.0)}


def settings(**overrides: Any) -> LoadgenSettings:
    values: dict[str, Any] = {"fps": 5.0, "duration_s": 2.0, "warmup_s": 0.0, "cameras": 3}
    values.update(overrides)
    return LoadgenSettings(**values)


def plans(s: LoadgenSettings, seed: int = 1) -> list[CameraPlan]:
    lengths = {k: len(v.frames) for k, v in CLIPS.items()}
    return plan_cameras(s.cameras, list(CLIPS), lengths, s.fps, seed)


def make_runner(
    s: LoadgenSettings, transport: Transport, recorder: Recorder, ps: list[CameraPlan]
) -> Runner:
    return Runner(s, ps, CLIPS, transport, recorder, clock=FakeClock(START), run_id="run-1")


def by_camera(frames: list[OutgoingFrame]) -> dict[str, list[OutgoingFrame]]:
    out: dict[str, list[OutgoingFrame]] = defaultdict(list)
    for f in frames:
        out[f.camera_id].append(f)
    return out


def test_paced_cameras_send_on_schedule(vrun: VRun, recorder: Recorder) -> None:
    s = settings()
    ps = plans(s)
    transport = ScriptedTransport(latency_s=0.01)
    timing = vrun(make_runner(s, transport, recorder, ps).run())

    cams = by_camera(transport.frames)
    assert sorted(cams) == ["cam-001", "cam-002", "cam-003"]
    for plan in ps:
        frames = cams[plan.camera_id]
        assert [f.seq for f in frames] == list(range(10))  # 2 s at 5 FPS
        assert {f.run_id for f in frames} == {"run-1"}
        # Capture timestamps are the wall-clock start plus each frame's due time.
        start_ms = int(START.timestamp() * 1000)
        expected_ms = [start_ms + int((plan.phase_s + k * 0.2) * 1000) for k in range(10)]
        assert [f.ts_unix_ms for f in frames] == pytest.approx(expected_ms, abs=1)
        clip = CLIPS[plan.clip_id]
        n = len(clip.frames)
        expected_jpeg = [clip.frames[clip_frame(plan, k, 5.0, clip.fps, n)] for k in range(10)]
        assert [f.jpeg for f in frames] == expected_jpeg
    summary = recorder.summary(2.0, 15.0)
    assert summary["frames"]["accepted"] == 30
    assert summary["schedule_lag_ms"]["max"] == pytest.approx(0.0, abs=1e-6)
    assert summary["send_latency_ms"]["p50"] == pytest.approx(10.0)
    assert isinstance(timing, RunTiming)


def test_sends_start_exactly_at_their_due_time(vrun: VRun, recorder: Recorder) -> None:
    s = settings(cameras=1, duration_s=1.0)
    ps = [CameraPlan("cam-001", "clip-a", 0, phase_s=0.03)]
    transport = ScriptedTransport()
    vrun(make_runner(s, transport, recorder, ps).run())
    assert transport.send_times == pytest.approx([0.03, 0.23, 0.43, 0.63, 0.83])


def test_open_loop_drops_frames_when_the_in_flight_limit_is_hit(
    vrun: VRun, recorder: Recorder
) -> None:
    """A slow server must not slow the camera down (no coordinated omission): frames keep
    coming due every 200 ms, and the ones that find 2 sends in flight are dropped."""
    s = settings(cameras=1, http=HttpSettings(max_in_flight_per_camera=2))
    ps = [CameraPlan("cam-001", "clip-a", 0, phase_s=0.01)]
    transport = ScriptedTransport(latency_s=0.9)
    vrun(make_runner(s, transport, recorder, ps).run())
    # Due at 0.01 + 0.2k. Sends 0 and 1 hold both slots until 0.91 and 1.11, so frames 2-4 are
    # dropped; 5 (due 1.01) and 6 (due 1.21) take the freed slots; 7-9 are dropped.
    assert [f.seq for f in transport.frames] == [0, 1, 5, 6]
    summary = recorder.summary(2.0, 5.0)
    assert summary["frames"]["dropped"] == 6
    assert summary["frames"]["accepted"] == 4
    assert summary["frames"]["incomplete"] == 0  # the drain waited for the last two sends


def test_unpaced_mode_keeps_the_in_flight_limit_busy(vrun: VRun, recorder: Recorder) -> None:
    s = settings(
        cameras=1, mode="unpaced", duration_s=1.0, http=HttpSettings(max_in_flight_per_camera=2)
    )
    ps = [CameraPlan("cam-001", "clip-a", 0, 0.0)]
    transport = ScriptedTransport(latency_s=0.01)
    vrun(make_runner(s, transport, recorder, ps).run())
    # Two slots, each turning over every 10 ms, for 1 s.
    assert len(transport.frames) == 200
    assert [f.seq for f in transport.frames] == list(range(200))
    assert recorder.summary(1.0, None)["frames"]["dropped"] == 0


def test_stop_ends_scheduling_and_drains_in_flight(vrun: VRun, recorder: Recorder) -> None:
    s = settings(cameras=1, duration_s=60.0)
    ps = [CameraPlan("cam-001", "clip-a", 0, phase_s=0.0)]
    transport = ScriptedTransport(latency_s=0.05)
    runner = make_runner(s, transport, recorder, ps)

    async def main() -> RunTiming:
        asyncio.get_running_loop().call_later(0.5, runner.stop)
        return await runner.run()

    timing = vrun(main())
    assert [f.seq for f in transport.frames] == [0, 1, 2]  # due 0.0, 0.2, 0.4
    assert runner.stopped
    assert timing.elapsed_s == pytest.approx(0.5)
    assert recorder.summary(0.5, None)["frames"]["accepted"] == 3


def test_outcomes_are_counted(vrun: VRun, recorder: Recorder) -> None:
    s = settings(cameras=1, duration_s=1.0)
    ps = [CameraPlan("cam-001", "clip-a", 0, 0.0)]
    script = [
        from_status(202),
        from_status(429),
        SendResult(Outcome.FAILED, None, "timeout"),
        from_status(500),
        from_status(202),
    ]
    vrun(make_runner(s, ScriptedTransport(script=script), recorder, ps).run())
    frames = recorder.summary(1.0, None)["frames"]
    assert (frames["accepted"], frames["rejected"], frames["failed"]) == (2, 1, 2)


def test_a_raising_transport_counts_as_failed(vrun: VRun, recorder: Recorder) -> None:
    def explode(frame: OutgoingFrame) -> SendResult:
        raise RuntimeError("bug in a transport")

    s = settings(cameras=1, duration_s=0.5)
    ps = [CameraPlan("cam-001", "clip-a", 0, 0.0)]
    vrun(make_runner(s, ScriptedTransport(respond=explode), recorder, ps).run())
    summary = recorder.summary(0.5, None)
    assert summary["frames"]["failed"] == 3
    assert summary["failure_reasons"] == {"error": 3}


def test_runs_are_reproducible(vrun: VRun) -> None:
    def frames() -> list[tuple[str, int, int, bytes]]:
        s = settings()
        transport = ScriptedTransport(latency_s=0.01)
        rec = Recorder(Metrics("loadgen"), 0.0)
        vrun(make_runner(s, transport, rec, plans(s, seed=42)).run())
        return [(f.camera_id, f.seq, f.ts_unix_ms, f.jpeg) for f in transport.frames]

    assert frames() == frames()
