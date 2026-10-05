"""ULID generation (https://github.com/ulid/spec).

A ULID is 128 bits: a 48-bit millisecond Unix timestamp followed by 80 random bits, encoded as
26 Crockford base32 characters. We use ULIDs for `frame_id` and other IDs because they:

- sort lexicographically by creation time, so time-range scans and B-tree inserts stay local
  (random UUIDv4 keys scatter inserts across the index);
- are generated without coordination, so ingest replicas never need a shared counter;
- embed their timestamp, which helps debugging.

Implemented here rather than via a library so the clock and the randomness source can be
injected: tests get deterministic IDs and the timestamp comes from the same `Clock` as the rest
of the code.

Within one millisecond, `UlidGenerator` is monotonic: it increments the random part instead of
drawing a new one, so IDs from one generator are strictly increasing.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from datetime import UTC, datetime

from vms.common.clock import Clock, SystemClock

CROCKFORD32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
_DECODE = {c: i for i, c in enumerate(CROCKFORD32)}
ULID_LENGTH = 26
_TIMESTAMP_BITS = 48
_RANDOM_BITS = 80
_MAX_TIMESTAMP = (1 << _TIMESTAMP_BITS) - 1
_MAX_RANDOM = (1 << _RANDOM_BITS) - 1


def encode_ulid(timestamp_ms: int, randomness: int) -> str:
    """Encode a timestamp (ms) and 80 random bits as a 26-character ULID string."""
    if not 0 <= timestamp_ms <= _MAX_TIMESTAMP:
        raise ValueError(f"timestamp_ms out of range: {timestamp_ms}")
    if not 0 <= randomness <= _MAX_RANDOM:
        raise ValueError("randomness must fit in 80 bits")
    value = (timestamp_ms << _RANDOM_BITS) | randomness
    chars = []
    for _ in range(ULID_LENGTH):
        chars.append(CROCKFORD32[value & 0x1F])
        value >>= 5
    return "".join(reversed(chars))


def decode_ulid(ulid: str) -> tuple[int, int]:
    """Return (timestamp_ms, randomness). Accepts lowercase; rejects invalid strings."""
    if len(ulid) != ULID_LENGTH:
        raise ValueError(f"ULID must be {ULID_LENGTH} characters, got {len(ulid)}")
    value = 0
    for char in ulid.upper():
        try:
            value = (value << 5) | _DECODE[char]
        except KeyError:
            raise ValueError(f"invalid ULID character: {char!r}") from None
    if value >> 128:
        raise ValueError("ULID overflows 128 bits")
    return value >> _RANDOM_BITS, value & _MAX_RANDOM


def ulid_datetime(ulid: str) -> datetime:
    """The creation time embedded in a ULID."""
    timestamp_ms, _ = decode_ulid(ulid)
    return datetime.fromtimestamp(timestamp_ms / 1000, tz=UTC)


def is_ulid(value: str) -> bool:
    try:
        decode_ulid(value)
    except ValueError:
        return False
    return True


class UlidGenerator:
    """Thread-safe, monotonic ULID generator with injectable clock and randomness."""

    def __init__(
        self,
        clock: Clock | None = None,
        randbytes: Callable[[int], bytes] = os.urandom,
    ) -> None:
        self._clock = clock or SystemClock()
        self._randbytes = randbytes
        self._lock = threading.Lock()
        self._last_ms = -1
        self._last_random = 0

    def new(self) -> str:
        timestamp_ms = int(self._clock.now().timestamp() * 1000)
        with self._lock:
            if timestamp_ms <= self._last_ms:
                # Same millisecond, or the wall clock stepped back: keep ordering by
                # incrementing the previous ID instead of using the (earlier) timestamp.
                timestamp_ms = self._last_ms
                randomness = self._last_random + 1
                if randomness > _MAX_RANDOM:
                    raise OverflowError("ULID random component exhausted within one millisecond")
            else:
                randomness = int.from_bytes(self._randbytes(10), "big")
            self._last_ms = timestamp_ms
            self._last_random = randomness
        return encode_ulid(timestamp_ms, randomness)


_default_generator = UlidGenerator()


def new_ulid() -> str:
    """A new ULID from the process-wide generator (system clock, OS randomness)."""
    return _default_generator.new()
