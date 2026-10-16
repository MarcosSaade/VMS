"""Grafana annotations: payloads, and best-effort posting with bounded, jittered retries."""

from __future__ import annotations

import asyncio
import random
from datetime import UTC, datetime, timedelta

from vms.observe.annotations import Annotation, AnnotationError, PostResult, post_all

T0 = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)


class ScriptedSink:
    """Raises the scripted errors in order, then accepts."""

    def __init__(self, errors: list[AnnotationError] | None = None) -> None:
        self.errors = list(errors or [])
        self.attempts: list[Annotation] = []
        self.posted: list[Annotation] = []
        self.closed = False

    async def post(self, annotation: Annotation) -> None:
        self.attempts.append(annotation)
        if self.errors:
            raise self.errors.pop(0)
        self.posted.append(annotation)

    async def aclose(self) -> None:
        self.closed = True


def run_post(
    sink: ScriptedSink, annotations: list[Annotation], max_retries: int = 2
) -> tuple[PostResult, list[float]]:
    sleeps: list[float] = []

    async def sleep(seconds: float) -> None:
        sleeps.append(seconds)

    result = asyncio.run(
        post_all(
            sink,
            annotations,
            max_retries=max_retries,
            base_s=0.2,
            max_s=2.0,
            rng=random.Random(0),
            sleep=sleep,
        )
    )
    return result, sleeps


def point(text: str = "a") -> Annotation:
    return Annotation(time=T0, text=text, tags=("drift-onset",))


def test_payload_is_unix_ms_and_a_region_has_an_end() -> None:
    assert point().payload() == {
        "time": round(T0.timestamp() * 1000),
        "tags": ["drift-onset"],
        "text": "a",
    }
    region = Annotation(T0, "r", ("x",), time_end=T0 + timedelta(seconds=120))
    assert region.payload()["timeEnd"] - region.payload()["time"] == 120_000


def test_transient_failures_are_retried_with_capped_jitter() -> None:
    sink = ScriptedSink([AnnotationError("timeout", retryable=True)] * 2)
    result, sleeps = run_post(sink, [point()])
    assert (result.posted, result.failed) == (1, 0)
    assert len(sink.attempts) == 3
    assert len(sleeps) == 2
    assert 0 <= sleeps[0] <= 0.2
    assert 0 <= sleeps[1] <= 0.4


def test_retries_are_bounded_and_failures_counted_not_raised() -> None:
    sink = ScriptedSink([AnnotationError("connection", retryable=True)] * 10)
    result, _ = run_post(sink, [point("a"), point("b")], max_retries=2)
    # a: 3 attempts, all fail; b: 3 attempts (errors 4-6), all fail
    assert (result.posted, result.failed) == (0, 2)
    assert len(sink.attempts) == 6
    assert result.summary()["errors"] == ["connection"]


def test_client_errors_are_not_retried() -> None:
    sink = ScriptedSink([AnnotationError("http 401", retryable=False)])
    result, sleeps = run_post(sink, [point("a"), point("b")])
    assert (result.posted, result.failed) == (1, 1)
    assert sleeps == []
    assert [a.text for a in sink.posted] == ["b"]
