"""Camera-local time ranges resolve to UTC with an injected now, including across DST."""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pytest

from vms.query.plan import AbsoluteTimeRange, LastTimeRange, NamedPeriod, NamedTimeRange, TimeUnit
from vms.query.timerange import UtcRange, resolve

UTC = dt.UTC
LONDON = ZoneInfo("Europe/London")
TOKYO = ZoneInfo("Asia/Tokyo")
# Monday 2026-10-12 13:30 UTC = 14:30 in London (BST, UTC+1).
NOW = dt.datetime(2026, 10, 12, 13, 30, tzinfo=UTC)


def u(*args: int) -> dt.datetime:
    return dt.datetime(*args, tzinfo=UTC)


@pytest.mark.parametrize(
    ("name", "start", "end"),
    [
        (NamedPeriod.TODAY, u(2026, 10, 11, 23), u(2026, 10, 12, 23)),
        (NamedPeriod.YESTERDAY, u(2026, 10, 10, 23), u(2026, 10, 11, 23)),
        (NamedPeriod.THIS_MORNING, u(2026, 10, 12, 5), u(2026, 10, 12, 11)),
        (NamedPeriod.THIS_AFTERNOON, u(2026, 10, 12, 11), u(2026, 10, 12, 17)),
        (NamedPeriod.THIS_EVENING, u(2026, 10, 12, 17), u(2026, 10, 12, 23)),
        (NamedPeriod.LAST_NIGHT, u(2026, 10, 11, 17), u(2026, 10, 12, 5)),
        (NamedPeriod.THIS_WEEK, u(2026, 10, 11, 23), u(2026, 10, 18, 23)),
        (NamedPeriod.LAST_WEEK, u(2026, 10, 4, 23), u(2026, 10, 11, 23)),
    ],
)
def test_named_periods_in_london_summer_time(
    name: NamedPeriod, start: dt.datetime, end: dt.datetime
) -> None:
    assert resolve(NamedTimeRange(name=name), LONDON, NOW) == UtcRange(start, end)


def test_today_depends_on_the_camera_timezone() -> None:
    # 13:30 UTC is already 22:30 in Tokyo (UTC+9): "today" there started at 15:00 UTC.
    assert resolve(NamedTimeRange(name=NamedPeriod.TODAY), TOKYO, NOW) == UtcRange(
        u(2026, 10, 11, 15), u(2026, 10, 12, 15)
    )


def test_local_date_differs_from_utc_date() -> None:
    # 23:30 UTC on Oct 12 is already Oct 13 in Tokyo, so "yesterday" there is Oct 12.
    late = u(2026, 10, 12, 23, 30)
    assert resolve(NamedTimeRange(name=NamedPeriod.YESTERDAY), TOKYO, late) == UtcRange(
        u(2026, 10, 11, 15), u(2026, 10, 12, 15)
    )


def test_day_with_dst_change_is_25_hours() -> None:
    # London leaves BST on Sunday 2026-10-25 at 02:00 local: that day lasts 25 hours.
    now = u(2026, 10, 25, 12)
    today = resolve(NamedTimeRange(name=NamedPeriod.TODAY), LONDON, now)
    assert today == UtcRange(u(2026, 10, 24, 23), u(2026, 10, 26, 0))
    assert today.duration == dt.timedelta(hours=25)
    week = resolve(NamedTimeRange(name=NamedPeriod.THIS_WEEK), LONDON, now)
    assert week.duration == dt.timedelta(days=7, hours=1)


def test_spring_forward_day_is_23_hours() -> None:
    today = resolve(NamedTimeRange(name=NamedPeriod.TODAY), LONDON, u(2026, 3, 29, 12))
    assert today.duration == dt.timedelta(hours=23)


@pytest.mark.parametrize(
    ("amount", "unit", "delta"),
    [
        (15, TimeUnit.MINUTES, dt.timedelta(minutes=15)),
        (2, TimeUnit.HOURS, dt.timedelta(hours=2)),
        (7, TimeUnit.DAYS, dt.timedelta(days=7)),
        (1, TimeUnit.WEEKS, dt.timedelta(weeks=1)),
    ],
)
def test_last_n_units_end_now(amount: int, unit: TimeUnit, delta: dt.timedelta) -> None:
    tr = LastTimeRange(amount=amount, unit=unit)
    assert resolve(tr, LONDON, NOW) == UtcRange(NOW - delta, NOW)
    assert resolve(tr, TOKYO, NOW) == resolve(tr, LONDON, NOW)  # timezone-independent


def test_absolute_is_local_wall_clock() -> None:
    tr = AbsoluteTimeRange(start=dt.datetime(2026, 10, 3, 9), end=dt.datetime(2026, 10, 3, 17))
    assert resolve(tr, LONDON, NOW) == UtcRange(u(2026, 10, 3, 8), u(2026, 10, 3, 16))
    assert resolve(tr, TOKYO, NOW) == UtcRange(u(2026, 10, 3, 0), u(2026, 10, 3, 8))


def test_absolute_in_winter_uses_gmt() -> None:
    tr = AbsoluteTimeRange(start=dt.datetime(2026, 12, 1, 9), end=dt.datetime(2026, 12, 1, 10))
    assert resolve(tr, LONDON, NOW) == UtcRange(u(2026, 12, 1, 9), u(2026, 12, 1, 10))


def test_now_must_be_aware() -> None:
    with pytest.raises(ValueError, match="aware"):
        resolve(NamedTimeRange(name=NamedPeriod.TODAY), LONDON, dt.datetime(2026, 1, 1))
