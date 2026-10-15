"""Fakes for the benchmark runner: Prometheus, ingest load and query clients on virtual time."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from tests.unit.loadgen.conftest import VirtualTimeLoop
from vms.bench.ingestload import IngestResult, IngestRun
from vms.bench.prom import Value
from vms.bench.queryload import Reply
from vms.bench.settings import SERVICE, BenchSettings
from vms.common.settings import load_settings

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = REPO_ROOT / "configs"
WALL0 = datetime(2026, 10, 13, 12, tzinfo=UTC)


@pytest.fixture
def vrun() -> Callable[[Coroutine[Any, Any, Any]], Any]:
    def run(coro: Coroutine[Any, Any, Any]) -> Any:
        loop = VirtualTimeLoop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    return run


@pytest.fixture
def settings() -> BenchSettings:
    return load_settings(BenchSettings, SERVICE, config_dir=CONFIGS)


def vnow() -> float:
    """Unix seconds on the virtual loop, anchored at WALL0 (the runner's anchor in tests)."""
    return WALL0.timestamp() + asyncio.get_running_loop().time()


class FakePrometheus:
    """Answers the configured query `name` (matched on its text) with `values(name)`, and the
    drain query from `backlog` (consumed per call, the last value repeated)."""

    def __init__(
        self,
        settings: BenchSettings,
        values: Callable[[str], Any],
        backlog: list[float | None] | None = None,
        workers: list[dict[str, str]] | None = None,
    ) -> None:
        self.workers = workers if workers is not None else []
        self._templates = {name: q.query for name, q in settings.queries.items()}
        self._by = {name: q.by for name, q in settings.queries.items()}
        self._values = values
        self.backlog = list(backlog or [0.0])
        self.calls: list[tuple[str, float]] = []

    def _name(self, promql: str) -> str:
        for name, template in self._templates.items():
            head, _, tail = template.partition("$window")
            if promql.startswith(head) and promql.endswith(tail):
                return name
        raise AssertionError(f"unexpected query {promql}")

    async def query(self, promql: str, at_unix_s: float) -> list[tuple[dict[str, str], Value]]:
        self.calls.append((promql, at_unix_s))
        if promql == "max(vms_worker_backlog_frames)":
            value = self.backlog.pop(0) if len(self.backlog) > 1 else self.backlog[0]
            return [] if value is None else [({}, value)]
        if promql == "vms_worker_model_info":
            labels = {"__name__": "vms_worker_model_info", "job": "worker", "instance": "w:8090"}
            return [(labels | w, 1.0) for w in self.workers]
        name = self._name(promql)
        value = self._values(name)
        if value is None:
            return []
        by = self._by[name]
        if by is not None:
            return [({by: k}, v) for k, v in value.items()]
        return [({}, value)]


class FakeIngest:
    """Takes warmup + hold (+ `startup_s`) of virtual time; reports a scripted summary."""

    def __init__(self, startup_s: float = 2.0, frames: dict[str, int] | None = None) -> None:
        self.startup_s = startup_s
        self.frames = frames or {}
        self.runs: list[IngestRun] = []

    async def run(self, spec: IngestRun) -> IngestResult:
        self.runs.append(spec)
        await asyncio.sleep(self.startup_s)
        started = datetime.fromtimestamp(vnow(), UTC)
        await asyncio.sleep(spec.warmup_s + spec.hold_s)
        offered = spec.cameras * spec.fps
        summary = {
            "offered_fps": offered,
            "sent_fps": offered,
            "accepted_fps": offered,
            "frames": {
                "sent": int(offered * spec.hold_s),
                "accepted": int(offered * spec.hold_s),
                "rejected": 0,
                "failed": 0,
                "dropped": 0,
                "incomplete": 0,
                **self.frames,
            },
            "send_latency_ms": {"p50": 5.0, "p95": 9.0, "p99": 12.0},
            "schedule_lag_ms": {"p50": 0.1, "p95": 0.5, "p99": 1.0},
            "cpu": {"utilization_cores": 0.1 * spec.cameras},
            "generator_ok": True,
            "started_at": started.isoformat(),
        }
        return IngestResult(started, summary, {"cameras": spec.cameras}, f"lg-{len(self.runs)}")


class FakeQueryClient:
    def __init__(self, latency_s: float = 0.5, reply: Reply | None = None) -> None:
        self.latency_s = latency_s
        self.reply = reply or Reply("ok", tokens=100)
        self.asked: list[str] = []
        self.closed = False

    async def ask(self, question: str) -> Reply:
        self.asked.append(question)
        await asyncio.sleep(self.latency_s)
        return self.reply

    async def aclose(self) -> None:
        self.closed = True


def at(seconds: float) -> datetime:
    return WALL0 + timedelta(seconds=seconds)


@pytest.fixture
def vrun_real() -> Callable[[Coroutine[Any, Any, Any]], Any]:
    """A real event loop, for tests that wait on a subprocess."""
    return asyncio.run
