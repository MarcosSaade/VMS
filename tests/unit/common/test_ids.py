import random
import threading
from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from vms.common.clock import FakeClock
from vms.common.ids import (
    CROCKFORD32,
    UlidGenerator,
    decode_ulid,
    encode_ulid,
    is_ulid,
    new_ulid,
    ulid_datetime,
)

T0 = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)


def seeded_bytes(seed: int = 7) -> Callable[[int], bytes]:
    rng = random.Random(seed)
    return lambda n: rng.randbytes(n)


def test_spec_boundary_values() -> None:
    assert encode_ulid(0, 0) == "0" * 26
    max_ulid = encode_ulid((1 << 48) - 1, (1 << 80) - 1)
    assert max_ulid == "7" + "Z" * 25
    assert decode_ulid(max_ulid) == ((1 << 48) - 1, (1 << 80) - 1)


def test_roundtrip_and_alphabet() -> None:
    ulid = encode_ulid(1_759_665_600_000, 123_456_789)
    assert len(ulid) == 26
    assert set(ulid) <= set(CROCKFORD32)
    assert decode_ulid(ulid) == (1_759_665_600_000, 123_456_789)
    assert decode_ulid(ulid.lower()) == decode_ulid(ulid)


@pytest.mark.parametrize(
    "bad",
    ["", "0" * 25, "0" * 27, "I" + "0" * 25, "U" + "0" * 25, "8" + "0" * 25],
    ids=["empty", "short", "long", "letter-I", "letter-U", "overflow"],
)
def test_invalid_strings_are_rejected(bad: str) -> None:
    assert not is_ulid(bad)
    with pytest.raises(ValueError, match="ULID"):
        decode_ulid(bad)


@pytest.mark.parametrize(("ts", "rand"), [(-1, 0), (1 << 48, 0), (0, -1), (0, 1 << 80)])
def test_encode_rejects_out_of_range(ts: int, rand: int) -> None:
    with pytest.raises(ValueError, match=r"out of range|80 bits"):
        encode_ulid(ts, rand)


def test_embeds_clock_time() -> None:
    gen = UlidGenerator(FakeClock(T0), seeded_bytes())
    assert ulid_datetime(gen.new()) == T0


def test_deterministic_with_seeded_randomness() -> None:
    a = UlidGenerator(FakeClock(T0), seeded_bytes(1))
    b = UlidGenerator(FakeClock(T0), seeded_bytes(1))
    assert [a.new() for _ in range(5)] == [b.new() for _ in range(5)]


def test_sorts_by_time() -> None:
    clock = FakeClock(T0)
    gen = UlidGenerator(clock, seeded_bytes())
    ids = []
    for _ in range(50):
        ids.append(gen.new())
        clock.advance(0.001)
    assert ids == sorted(ids)


def test_monotonic_within_same_millisecond() -> None:
    gen = UlidGenerator(FakeClock(T0), seeded_bytes())
    ids = [gen.new() for _ in range(1000)]
    assert ids == sorted(ids)
    assert len(set(ids)) == 1000
    randoms = [decode_ulid(i)[1] for i in ids]
    assert randoms == list(range(randoms[0], randoms[0] + 1000))


def test_wall_clock_stepping_back_keeps_order() -> None:
    clock = FakeClock(T0)
    gen = UlidGenerator(clock, seeded_bytes())
    first = gen.new()
    clock.set(datetime(2026, 10, 5, 11, 0, tzinfo=UTC))
    assert gen.new() > first


def test_overflow_within_a_millisecond_raises() -> None:
    gen = UlidGenerator(FakeClock(T0), lambda n: b"\xff" * n)
    gen.new()
    with pytest.raises(OverflowError):
        gen.new()


def test_thread_safe_uniqueness() -> None:
    gen = UlidGenerator(FakeClock(T0), seeded_bytes())
    results: list[str] = []
    lock = threading.Lock()

    def work() -> None:
        local = [gen.new() for _ in range(500)]
        with lock:
            results.extend(local)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(results)) == 4000


def test_default_generator() -> None:
    assert is_ulid(new_ulid())
