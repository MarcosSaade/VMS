import asyncio
import random
from collections.abc import Callable, Coroutine
from typing import Any

import pytest

from tests.unit.bench.conftest import WALL0, FakeQueryClient, vnow
from vms.bench.queryload import Reply, Sample, parse_reply, run_clients, summarize

VRun = Callable[[Coroutine[Any, Any, Any]], Any]


@pytest.mark.parametrize(
    ("body", "reply"),
    [
        ({"status": "ok", "tokens": 812}, Reply("ok", 812)),
        ({"status": "abstained", "tokens": 0}, Reply("abstained", 0)),
        (
            {"status": "error", "error": {"type": "planner_unavailable"}},
            Reply("error:planner_unavailable"),
        ),
        ({"status": "error", "error": None, "tokens": None}, Reply("error:unknown")),
        ({"detail": "Not Found"}, Reply("failed:bad_response")),
        (["not", "a", "dict"], Reply("failed:bad_response")),
    ],
)
def test_parse_reply(body: Any, reply: Reply) -> None:
    assert parse_reply(body) == reply


def test_clients_run_closed_loop_until_stopped(vrun: VRun) -> None:
    client = FakeQueryClient(latency_s=0.5)

    async def scenario() -> list[Sample]:
        stop = asyncio.Event()
        asyncio.get_running_loop().call_later(10.0, stop.set)
        return await run_clients(
            client, ["a", "b"], 3, stop, wall_now=vnow, rng=random.Random(1), think_time_s=0.5
        )

    samples = vrun(scenario())
    # 3 clients, one question per second each (0.5 s answer + 0.5 s think) for 10 s.
    assert len(samples) == 30
    assert all(s.latency_s == pytest.approx(0.5) for s in samples)
    starts = sorted(s.start_unix_s - WALL0.timestamp() for s in samples)
    assert starts[:3] == [0.0, 0.0, 0.0]
    assert starts[-1] == pytest.approx(9.0)
    assert set(client.asked) == {"a", "b"}


def test_question_choice_is_seeded(vrun: VRun) -> None:
    def asked(seed: int) -> list[str]:
        client = FakeQueryClient(latency_s=1.0)

        async def go() -> None:
            stop = asyncio.Event()
            asyncio.get_running_loop().call_later(5.5, stop.set)
            await run_clients(
                client, list("abcdef"), 1, stop, wall_now=vnow, rng=random.Random(seed)
            )

        vrun(go())
        return client.asked

    assert asked(7) == asked(7)
    assert asked(7) != asked(8)


def test_summarize_keeps_only_questions_started_in_the_window() -> None:
    samples = [
        Sample(9.0, 5.0, "ok", 100),  # warmup: started before the window
        Sample(10.0, 0.1, "ok", 100),
        Sample(12.0, 0.2, "abstained", 0),
        Sample(15.0, 0.3, "error:llm_timeout", 50),
        Sample(19.9, 0.4, "failed:timeout", 0),
        Sample(20.0, 9.0, "ok", 100),  # after the window
    ]
    s = summarize(samples, 10.0, 20.0, cpu_cores=0.05)
    assert s["questions"] == 4
    assert s["qps"] == 0.4
    assert s["outcomes"] == {"abstained": 1, "error:llm_timeout": 1, "failed:timeout": 1, "ok": 1}
    assert s["error_rate"] == 0.5
    assert s["abstain_rate"] == 0.25
    assert s["tokens_per_question"] == 37.5
    assert s["latency_ms"]["max"] == 400.0
    assert s["client_cpu_cores"] == 0.05


def test_summarize_without_samples() -> None:
    s = summarize([], 0.0, 10.0, cpu_cores=None)
    assert s["questions"] == 0
    assert s["error_rate"] is None
    assert s["latency_ms"]["p95"] is None
