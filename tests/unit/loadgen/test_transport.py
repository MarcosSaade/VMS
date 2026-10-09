import random
from collections.abc import Callable, Coroutine
from typing import Any

import pytest

from tests.unit.loadgen.conftest import ScriptedTransport
from vms.loadgen.transport import (
    NullTransport,
    Outcome,
    OutgoingFrame,
    RetryingTransport,
    SendResult,
    classify_status,
    from_status,
)

VRun = Callable[[Coroutine[Any, Any, Any]], Any]
FRAME = OutgoingFrame("cam-001", "run", 0, 0, b"jpeg")
TIMEOUT = SendResult(Outcome.FAILED, None, "timeout")


@pytest.mark.parametrize(
    ("status", "outcome"),
    [
        (200, Outcome.ACCEPTED),
        (202, Outcome.ACCEPTED),
        (204, Outcome.ACCEPTED),
        (429, Outcome.REJECTED),
        (503, Outcome.REJECTED),
        (400, Outcome.FAILED),
        (422, Outcome.FAILED),
        (500, Outcome.FAILED),
        (502, Outcome.FAILED),
        (301, Outcome.FAILED),
    ],
)
def test_classify_status(status: int, outcome: Outcome) -> None:
    assert classify_status(status) is outcome


@pytest.mark.parametrize(
    ("result", "retryable"),
    [
        (TIMEOUT, True),
        (SendResult(Outcome.FAILED, None, "connection"), True),
        (from_status(500), True),
        (from_status(502), True),
        (from_status(503), False),  # shed load stays shed
        (from_status(429), False),
        (from_status(400), False),  # a client bug does not fix itself
        (from_status(202), False),
    ],
)
def test_only_transient_failures_are_retryable(result: SendResult, retryable: bool) -> None:
    assert result.retryable is retryable


def _retrying(inner: ScriptedTransport, max_retries: int, sleeps: list[float]) -> RetryingTransport:
    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    return RetryingTransport(
        inner, max_retries=max_retries, base_s=0.1, max_s=0.25, rng=random.Random(0), sleep=sleep
    )


def test_retries_until_success(vrun: VRun) -> None:
    inner = ScriptedTransport(script=[TIMEOUT, from_status(502), from_status(202)])
    sleeps: list[float] = []
    result = vrun(_retrying(inner, 3, sleeps).send(FRAME))
    assert result.outcome is Outcome.ACCEPTED
    assert result.attempts == 3
    assert len(inner.frames) == 3
    assert len(sleeps) == 2


def test_retries_are_bounded(vrun: VRun) -> None:
    inner = ScriptedTransport(script=[TIMEOUT] * 10)
    sleeps: list[float] = []
    result = vrun(_retrying(inner, 2, sleeps).send(FRAME))
    assert (result.outcome, result.reason, result.attempts) == (Outcome.FAILED, "timeout", 3)
    assert len(inner.frames) == 3


def test_rejections_are_not_retried(vrun: VRun) -> None:
    inner = ScriptedTransport(script=[from_status(429)])
    result = vrun(_retrying(inner, 5, []).send(FRAME))
    assert (result.outcome, result.attempts) == (Outcome.REJECTED, 1)


def test_backoff_is_full_jitter_and_capped() -> None:
    transport = _retrying(ScriptedTransport(), 10, [])
    for retry, ceiling in [(1, 0.1), (2, 0.2), (3, 0.25), (8, 0.25)]:
        samples = [transport.backoff(retry) for _ in range(200)]
        assert all(0.0 <= s <= ceiling for s in samples)
        assert max(samples) > ceiling * 0.8  # spread across the range, not a constant


def test_null_transport_accepts(vrun: VRun) -> None:
    transport = NullTransport()
    assert vrun(transport.send(FRAME)).outcome is Outcome.ACCEPTED
    assert transport.sent == 1
