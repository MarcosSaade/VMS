"""`python -m vms.storage wait-drained`: the polling loop and the summary (no database)."""

from __future__ import annotations

from vms.common.clock import FakeClock
from vms.storage.cli import format_summary, wait_until_drained


def run(backlogs: list[int], timeout_s: float) -> tuple[bool, list[int], FakeClock]:
    clock = FakeClock()
    seen: list[int] = []
    left = iter(backlogs)
    ok = wait_until_drained(
        lambda: next(left),
        timeout_s=timeout_s,
        interval_s=2.0,
        clock=clock,
        sleep=clock.advance,
        report=seen.append,
    )
    return ok, seen, clock


def test_returns_once_the_backlog_is_empty() -> None:
    ok, seen, clock = run([30, 12, 0, 99], timeout_s=60)
    assert ok is True
    assert seen == [30, 12, 0]
    assert clock.monotonic() == 4.0


def test_times_out_while_frames_remain() -> None:
    ok, seen, clock = run([5] * 100, timeout_s=10)
    assert ok is False
    assert seen == [5] * 6  # t = 0, 2, 4, 6, 8, 10
    assert clock.monotonic() == 10.0


def test_summary() -> None:
    text = format_summary(
        {
            "frames": [("done", 120), ("failed", 1)],
            "tracks": [("cam-001", "person", 9), ("cam-001", "car", 2), ("cam-003", "car", 4)],
            "events": [("cam-001", "zone_enter", 5), ("cam-001", "zone_exit", 4)],
        }
    )
    assert text.splitlines() == [
        "Pipeline output:",
        "  frames: done 120, failed 1",
        "  tracks cam-001: 9 person, 2 car",
        "  tracks cam-003: 4 car",
        "  events cam-001: 5 zone_enter, 4 zone_exit",
    ]
