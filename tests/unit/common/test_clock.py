from datetime import UTC, datetime, timedelta, timezone

import pytest

from vms.common.clock import FakeClock, SystemClock


def test_system_clock_is_utc_aware() -> None:
    now = SystemClock().now()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)


def test_system_monotonic_never_decreases() -> None:
    clock = SystemClock()
    first = clock.monotonic()
    assert clock.monotonic() >= first


def test_fake_clock_advances_both_readings() -> None:
    clock = FakeClock(datetime(2026, 5, 1, 12, 0, tzinfo=UTC))
    clock.advance(1.5)
    assert clock.now() == datetime(2026, 5, 1, 12, 0, 1, 500000, tzinfo=UTC)
    assert clock.monotonic() == 1.5


def test_fake_clock_set_jumps_wall_time_only() -> None:
    clock = FakeClock()
    clock.advance(2)
    clock.set(datetime(2020, 1, 1, tzinfo=UTC))
    assert clock.now() == datetime(2020, 1, 1, tzinfo=UTC)
    assert clock.monotonic() == 2


def test_fake_clock_normalises_to_utc() -> None:
    plus_two = timezone(timedelta(hours=2))
    clock = FakeClock(datetime(2026, 1, 1, 2, 0, tzinfo=plus_two))
    assert clock.now() == datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    assert clock.now().tzinfo == UTC


@pytest.mark.parametrize("bad", [datetime(2026, 1, 1)])
def test_fake_clock_rejects_naive_datetimes(bad: datetime) -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        FakeClock(bad)
    with pytest.raises(ValueError, match="timezone-aware"):
        FakeClock().set(bad)


def test_fake_clock_cannot_advance_backwards() -> None:
    with pytest.raises(ValueError, match="backwards"):
        FakeClock().advance(-1)
