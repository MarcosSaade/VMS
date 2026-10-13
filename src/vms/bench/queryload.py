"""Closed-loop query load: `concurrency` clients, each asking its next question as soon as the
previous reply arrives (plus an optional think time).

Closed loop is deliberate here. Questions are LLM-bound and the plan measures latency at a few
concurrency levels, which is exactly what a fixed client pool controls. The cost is the usual
one: a slow server lowers the offered rate instead of building a queue, so the results report
achieved queries/s next to latency and never call it a capacity. Ingest load, where open loop
matters, comes from loadgen (D17).

Samples carry their wall-clock start so the runner can keep only the steady-state window after
the fact; the client runs for the whole step, warmup included.
"""

from __future__ import annotations

import asyncio
import random
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import aiohttp

from vms.bench.stats import percentiles_ms


@dataclass(frozen=True)
class Reply:
    # ok, abstained, error:<type> (the service answered with an error), failed:<reason> (no
    # usable answer: timeout, connection, bad response)
    outcome: str
    tokens: int = 0


@dataclass(frozen=True)
class Sample:
    start_unix_s: float
    latency_s: float
    outcome: str
    tokens: int


class QueryClient(Protocol):
    async def ask(self, question: str) -> Reply: ...

    async def aclose(self) -> None: ...


class HttpQueryClient:
    """POST /v1/query over one pooled aiohttp session."""

    def __init__(self, base_url: str, *, timeout_s: float, max_connections: int) -> None:
        self._url = base_url.rstrip("/") + "/v1/query"
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._limit = max_connections
        self._session: aiohttp.ClientSession | None = None

    async def ask(self, question: str) -> Reply:
        if self._session is None:
            self._session = aiohttp.ClientSession(
                timeout=self._timeout, connector=aiohttp.TCPConnector(limit=self._limit)
            )
        try:
            async with self._session.post(self._url, json={"question": question}) as resp:
                body = await resp.json(content_type=None)
        except TimeoutError:
            return Reply("failed:timeout")
        except aiohttp.ClientConnectionError:
            return Reply("failed:connection")
        except (aiohttp.ClientError, ValueError):
            return Reply("failed:bad_response")
        return parse_reply(body)

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()


def parse_reply(body: Any) -> Reply:
    """Map a QueryResponse body to an outcome. Abstentions are answers, not errors."""
    if not isinstance(body, dict) or "status" not in body:
        return Reply("failed:bad_response")
    tokens = int(body.get("tokens") or 0)
    status = str(body["status"])
    if status == "error":
        error = body.get("error") or {}
        return Reply(f"error:{error.get('type', 'unknown')}", tokens)
    return Reply(status, tokens)


async def run_clients(
    client: QueryClient,
    questions: Sequence[str],
    concurrency: int,
    stop: asyncio.Event,
    *,
    wall_now: Callable[[], float],
    rng: random.Random,
    think_time_s: float = 0.0,
) -> list[Sample]:
    """Run `concurrency` clients until `stop` is set; every finished question is a sample.

    Latency uses the event loop's monotonic clock; `wall_now` (unix seconds) stamps each start.
    """
    loop = asyncio.get_running_loop()
    samples: list[Sample] = []

    async def one_client() -> None:
        while not stop.is_set():
            question = rng.choice(questions)
            wall, t0 = wall_now(), loop.time()
            reply = await client.ask(question)
            samples.append(Sample(wall, loop.time() - t0, reply.outcome, reply.tokens))
            if think_time_s:
                await asyncio.sleep(think_time_s)

    await asyncio.gather(*(one_client() for _ in range(concurrency)))
    return samples


def summarize(
    samples: Sequence[Sample], start_unix_s: float, end_unix_s: float, cpu_cores: float | None
) -> dict[str, Any]:
    """Steady-state summary: questions that started inside the window."""
    window_s = end_unix_s - start_unix_s
    steady = [s for s in samples if start_unix_s <= s.start_unix_s < end_unix_s]
    outcomes = Counter(s.outcome for s in steady)
    n = len(steady)
    errors = sum(c for o, c in outcomes.items() if o.startswith(("error:", "failed:")))
    return {
        "questions": n,
        "qps": round(n / window_s, 3) if window_s > 0 else None,
        "outcomes": dict(sorted(outcomes.items())),
        "error_rate": round(errors / n, 6) if n else None,
        "abstain_rate": round(outcomes["abstained"] / n, 6) if n else None,
        "tokens_per_question": round(sum(s.tokens for s in steady) / n, 1) if n else None,
        "latency_ms": percentiles_ms([s.latency_s for s in steady]),
        # The generator's own CPU over the whole step (all bench-process threads): ~1.0 means
        # its single event loop is saturated and the latencies include the client's own delay.
        "client_cpu_cores": cpu_cores,
    }
