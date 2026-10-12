"""Resolve a plan's camera-local time range to UTC bounds, using an injected "now".

The LLM never does date arithmetic: it emits "this morning" or "the last 2 hours", and this
module turns that into a half-open UTC interval [start, end) for one camera's IANA timezone.
Because "now" is passed in (from the service's `Clock`), every boundary is reproducible in
tests, including DST transitions.

Named periods (camera-local):
    today           00:00 today          -> 00:00 tomorrow
    yesterday       00:00 yesterday      -> 00:00 today
    this_morning    06:00 today          -> 12:00 today
    this_afternoon  12:00 today          -> 18:00 today
    this_evening    18:00 today          -> 00:00 tomorrow
    last_night      18:00 yesterday      -> 06:00 today
    this_week       Monday 00:00         -> next Monday 00:00
    last_week       last Monday 00:00    -> this Monday 00:00

A period can end in the future ("today" ends at midnight); rows cannot be in the future, so it
counts the same as ending now. "last N units" is [now - N, now). Absolute ranges are local wall
clock times. A local time that does not exist (skipped by a DST jump) or is ambiguous (repeated
when clocks go back) resolves with zoneinfo's `fold=0` rule: the earlier offset.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from zoneinfo import ZoneInfo

from vms.query.plan import (
    AbsoluteTimeRange,
    LastTimeRange,
    NamedPeriod,
    NamedTimeRange,
    TimeUnit,
)

UTC = dt.UTC

_UNIT = {
    TimeUnit.MINUTES: dt.timedelta(minutes=1),
    TimeUnit.HOURS: dt.timedelta(hours=1),
    TimeUnit.DAYS: dt.timedelta(days=1),
    TimeUnit.WEEKS: dt.timedelta(weeks=1),
}


@dataclass(frozen=True)
class UtcRange:
    start: dt.datetime  # inclusive, UTC
    end: dt.datetime  # exclusive, UTC

    @property
    def duration(self) -> dt.timedelta:
        return self.end - self.start


def last_duration(tr: LastTimeRange) -> dt.timedelta:
    return tr.amount * _UNIT[tr.unit]


def _local(day: dt.date, hour: int, tz: ZoneInfo) -> dt.datetime:
    return dt.datetime.combine(day, dt.time(hour), tzinfo=tz).astimezone(UTC)


def _wall(value: dt.datetime, tz: ZoneInfo) -> dt.datetime:
    return value.replace(tzinfo=tz).astimezone(UTC)


def resolve(
    tr: AbsoluteTimeRange | LastTimeRange | NamedTimeRange, tz: ZoneInfo, now: dt.datetime
) -> UtcRange:
    """The UTC interval for `tr` in timezone `tz` at instant `now` (timezone-aware)."""
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    now = now.astimezone(UTC)
    if isinstance(tr, LastTimeRange):
        return UtcRange(now - last_duration(tr), now)
    if isinstance(tr, AbsoluteTimeRange):
        return UtcRange(_wall(tr.start, tz), _wall(tr.end, tz))

    today = now.astimezone(tz).date()
    yesterday, tomorrow = today - dt.timedelta(days=1), today + dt.timedelta(days=1)
    monday = today - dt.timedelta(days=today.weekday())
    match tr.name:
        case NamedPeriod.TODAY:
            return UtcRange(_local(today, 0, tz), _local(tomorrow, 0, tz))
        case NamedPeriod.YESTERDAY:
            return UtcRange(_local(yesterday, 0, tz), _local(today, 0, tz))
        case NamedPeriod.THIS_MORNING:
            return UtcRange(_local(today, 6, tz), _local(today, 12, tz))
        case NamedPeriod.THIS_AFTERNOON:
            return UtcRange(_local(today, 12, tz), _local(today, 18, tz))
        case NamedPeriod.THIS_EVENING:
            return UtcRange(_local(today, 18, tz), _local(tomorrow, 0, tz))
        case NamedPeriod.LAST_NIGHT:
            return UtcRange(_local(yesterday, 18, tz), _local(today, 6, tz))
        case NamedPeriod.THIS_WEEK:
            return UtcRange(_local(monday, 0, tz), _local(monday + dt.timedelta(weeks=1), 0, tz))
        case NamedPeriod.LAST_WEEK:
            return UtcRange(_local(monday - dt.timedelta(weeks=1), 0, tz), _local(monday, 0, tz))
    raise AssertionError(f"unhandled period {tr.name}")  # pragma: no cover
