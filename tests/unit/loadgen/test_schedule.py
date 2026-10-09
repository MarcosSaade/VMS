import pytest

from vms.loadgen.schedule import CameraPlan, camera_id, clip_frame, due_time, plan_cameras

LENGTHS = {"a": 100, "b": 50}


def test_camera_ids_are_padded_so_they_sort() -> None:
    assert camera_id("cam-", 0, 20) == "cam-001"
    assert camera_id("cam-", 19, 20) == "cam-020"
    ids = [camera_id("cam-", i, 1500) for i in range(1500)]
    assert ids[0] == "cam-0001"
    assert ids == sorted(ids)


def test_plans_are_deterministic_for_a_seed() -> None:
    first = plan_cameras(10, ["a", "b"], LENGTHS, 5.0, seed=7)
    assert first == plan_cameras(10, ["a", "b"], LENGTHS, 5.0, seed=7)
    assert first != plan_cameras(10, ["a", "b"], LENGTHS, 5.0, seed=8)


def test_round_robin_assignment_and_ranges() -> None:
    plans = plan_cameras(5, ["a", "b"], LENGTHS, 4.0, seed=1)
    assert [p.clip_id for p in plans] == ["a", "b", "a", "b", "a"]
    for p in plans:
        assert 0 <= p.start_frame < LENGTHS[p.clip_id]
        assert 0.0 <= p.phase_s < 1 / 4.0


def test_random_assignment_uses_only_listed_clips() -> None:
    plans = plan_cameras(50, ["a", "b"], LENGTHS, 5.0, seed=3, assignment="random")
    assert {p.clip_id for p in plans} == {"a", "b"}


def test_phases_spread_cameras_across_the_frame_interval() -> None:
    phases = sorted(p.phase_s for p in plan_cameras(200, ["a"], LENGTHS, 5.0, seed=0))
    # Not all bunched at the tick: both halves of the 200 ms interval are used.
    assert phases[0] < 0.1 < phases[-1]


def test_needs_a_clip() -> None:
    with pytest.raises(ValueError, match="at least one clip"):
        plan_cameras(1, [], {}, 5.0, seed=0)


def test_due_times_are_phase_plus_k_intervals() -> None:
    plan = CameraPlan("cam-001", "a", 0, phase_s=0.05)
    assert [due_time(plan, k, 5.0) for k in range(3)] == pytest.approx([0.05, 0.25, 0.45])


@pytest.mark.parametrize(
    ("clip_fps", "fps", "expected"),
    [
        (30.0, 5.0, [0, 6, 12, 18, 24]),  # real time: 6 clip frames per send
        (25.0, 5.0, [0, 5, 10, 15, 20]),
        (30.0, 3.0, [0, 10, 20, 30, 40]),  # 10.0 exactly, not floored to 9 by float error
        (10.0, 20.0, [0, 0, 1, 1, 2]),  # sending faster than the clip repeats frames
    ],
)
def test_clip_frame_plays_in_real_time(clip_fps: float, fps: float, expected: list[int]) -> None:
    plan = CameraPlan("cam-001", "a", 0, 0.0)
    assert [clip_frame(plan, k, fps, clip_fps, 1000) for k in range(5)] == expected


def test_clip_frame_starts_at_offset_and_loops() -> None:
    plan = CameraPlan("cam-001", "a", start_frame=8, phase_s=0.0)
    assert [clip_frame(plan, k, 5.0, 30.0, 20) for k in range(4)] == [8, 14, 0, 6]
