"""Injectable time source.

Code that needs "now" takes a `Clock` instead of calling `datetime.now()` or `time.monotonic()`
directly, so tests can control time deterministically (no sleeps, no flaky timing).

- `now()` is wall-clock time (UTC, timezone-aware): use it for timestamps that are stored or
  shown, such as frame `ts` or ULIDs.
- `monotonic()` never goes backwards: use it to measure durations and latencies, since wall-clock
  time can jump (NTP, DST, manual changes).
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta
from typing import Protocol


class Clock(Protocol):
    def now(self) -> datetime:
        """Current wall-clock time, timezone-aware UTC."""
        ...

    def monotonic(self) -> float:
        """Seconds from an arbitrary fixed point; only differences are meaningful."""
        ...


class SystemClock:
    """The real clock."""

    def now(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()


class PinnedClock:
    """Wall-clock time fixed at one instant, real monotonic time.

    For evaluation against a frozen snapshot: "today" and "last week" must resolve against the
    snapshot's eval clock, while latencies are still measured for real.
    """

    def __init__(self, now: datetime) -> None:
        if now.tzinfo is None:
            raise ValueError("PinnedClock time must be timezone-aware")
        self._now = now.astimezone(UTC)

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return time.monotonic()


class FakeClock:
    """A clock that only moves when told to. Both readings advance together."""

    def __init__(self, start: datetime | None = None) -> None:
        start = start or datetime(2026, 1, 1, tzinfo=UTC)
        if start.tzinfo is None:
            raise ValueError("FakeClock start must be timezone-aware")
        self._now = start.astimezone(UTC)
        self._monotonic = 0.0

    def now(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._monotonic

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("FakeClock cannot go backwards; use set() for wall-clock jumps")
        self._now += timedelta(seconds=seconds)
        self._monotonic += seconds

    def set(self, when: datetime) -> None:
        """Jump wall-clock time (like an NTP correction). Monotonic time is unaffected."""
        if when.tzinfo is None:
            raise ValueError("FakeClock time must be timezone-aware")
        self._now = when.astimezone(UTC)
