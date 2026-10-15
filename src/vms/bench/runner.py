"""The end-to-end benchmark runner: ingest and query load against the running stack, measured
over each step's steady-state window.

For each repetition, for each step (`BenchScenario.steps`):

1. Drain: with ingest load in the step, wait until the worker backlog is empty (Prometheus), so
   the step starts from an empty queue instead of the previous step's leftovers.
2. Load: start the query clients, then run loadgen for `warmup_s + hold_s` (or just wait that
   long without ingest). The clients stop when loadgen exits.
3. Window: the last `hold_s` of loadgen's schedule, from loadgen's own start time. Query
   samples are filtered to it; loadgen's summary already excludes its warmup.
4. Read: wait `scrape_settle_s` for the last scrape, then evaluate every configured PromQL
   query at the window's end over the window (`vms.bench.prom`).
5. Judge: the ingest SLO (freshness p95, zero frame loss, a backlog that is not growing), the
   query SLO, and whether the harness itself kept up (`harness_ok`).

Across repetitions each step gets mean, std, min and max of its key numbers, and the scenario
gets a rated capacity: the largest camera count whose step, and every smaller one, met the
ingest SLO in every repetition (ARCHITECTURE section 2).
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vms.bench import stats
from vms.bench.ingestload import IngestLoad, IngestRun
from vms.bench.prom import Prometheus, Reading, as_float, backlog, read_window
from vms.bench.queryload import QueryClient, Sample, run_clients, summarize
from vms.bench.scenario import BenchScenario, Step
from vms.bench.settings import BenchSettings
from vms.common.clock import Clock
from vms.common.logs import get_logger

log = get_logger(__name__)

# Numbers summarized across repetitions (dotted paths into a StepResult).
AGGREGATED = (
    "loadgen.offered_fps",
    "loadgen.accepted_fps",
    "loadgen.send_latency_ms.p95",
    "loadgen.schedule_lag_ms.p99",
    "loadgen.cpu.utilization_cores",
    "lost_frames",
    "prometheus.worker_done_fps",
    "prometheus.freshness_p50_s",
    "prometheus.freshness_p95_s",
    "prometheus.freshness_p99_s",
    "prometheus.queue_wait_p95_s",
    "prometheus.backlog_slope_fps",
    "prometheus.backlog_max_frames",
    "prometheus.cpu_cores.ingest",
    "prometheus.cpu_cores.worker",
    "prometheus.cpu_cores.query",
    "query.qps",
    "query.latency_ms.p50",
    "query.latency_ms.p95",
    "query.latency_ms.p99",
    "query.error_rate",
    "query.abstain_rate",
    "query.client_cpu_cores",
)
LOSS_FIELDS = ("dropped", "rejected", "failed", "incomplete")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Window(_Model):
    start: datetime
    end: datetime
    seconds: float


class Verdict(_Model):
    """None: not applicable (no such load in the step) or not measurable."""

    freshness_ok: bool | None
    loss_ok: bool | None
    backlog_ok: bool | None
    ingest_ok: bool | None  # all three: the rated-capacity condition
    query_ok: bool | None
    harness_ok: bool


class StepResult(_Model):
    rep: int
    step: int
    cameras: int
    fps: float
    concurrency: int
    seed: int
    window: Window
    drain_s: float | None
    loadgen: dict[str, Any] | None  # loadgen's steady-state summary (its run_id included)
    query: dict[str, Any] | None  # vms.bench.queryload.summarize
    prometheus: dict[str, Reading]
    lost_frames: int | None
    slo: Verdict


class StepAggregate(_Model):
    step: int
    cameras: int
    concurrency: int
    reps: int
    ingest_ok_reps: int
    harness_ok_reps: int
    stats: dict[str, dict[str, float | None]]


class RatedCapacity(_Model):
    fps: float
    cameras: int | None  # None: even the smallest step failed
    lower_bound: bool  # the largest step passed; capacity may be higher
    first_failure: int | None


class E2EMetrics(_Model):
    """The `metrics` of a `bench.e2e` results file."""

    steps: list[StepResult]
    aggregate: list[StepAggregate]
    rated_capacity: RatedCapacity | None
    loadgen_config: dict[str, Any] | None  # resolved loadgen settings of the first ingest step
    interrupted: bool = False
    # Labels of each running worker's vms_worker_model_info when the run started.
    workers: list[dict[str, str]] = Field(default_factory=list)


QueryClientFactory = Callable[[int], QueryClient]


class BenchRunner:
    def __init__(
        self,
        scenario: BenchScenario,
        settings: BenchSettings,
        *,
        ingest: IngestLoad,
        query_client: QueryClientFactory,
        questions: Sequence[str],
        prom: Prometheus,
        clock: Clock,
        process_time: Callable[[], float] = time.process_time,
    ) -> None:
        self._sc = scenario
        self._s = settings
        self._ingest = ingest
        self._query_client = query_client
        self._questions = questions
        self._prom = prom
        self._clock = clock
        self._process_time = process_time
        self._seed = scenario.seed if scenario.seed is not None else settings.seed
        self._stop = asyncio.Event()
        self._anchor: tuple[float, float] = (0.0, 0.0)
        self._loadgen_config: dict[str, Any] | None = None

    @property
    def seed(self) -> int:
        return self._seed

    def stop(self) -> None:
        """Finish the current step, then stop (results so far are kept)."""
        self._stop.set()

    def _wall(self) -> float:
        """Wall-clock unix seconds, advanced by the event loop's monotonic clock."""
        wall0, t0 = self._anchor
        return wall0 + asyncio.get_running_loop().time() - t0

    async def run(self) -> E2EMetrics:
        if self._sc.hold_s < self._s.min_window_s:
            raise ValueError(
                f"hold_s {self._sc.hold_s} is shorter than min_window_s {self._s.min_window_s}"
            )
        self._anchor = (self._clock.now().timestamp(), asyncio.get_running_loop().time())
        workers = await worker_models(self._prom, self._wall())
        check_workers(workers, self._sc.expect_worker)
        results: list[StepResult] = []
        for rep in range(self._sc.repeats):
            for step in self._sc.steps():
                if self._stop.is_set():
                    break
                results.append(await self._step(step, rep))
        return E2EMetrics(
            steps=results,
            aggregate=aggregate(results),
            rated_capacity=rated_capacity(results, self._sc.ingest.fps)
            if self._sc.ingest
            else None,
            loadgen_config=self._loadgen_config,
            interrupted=self._stop.is_set(),
            workers=workers,
        )

    async def _step(self, step: Step, rep: int) -> StepResult:
        sc = self._sc
        seed = self._seed + 1000 * rep + step.index
        fps = sc.ingest.fps if sc.ingest else 0.0
        log.info(
            "step starting",
            extra={
                "rep": rep,
                "step": step.index,
                "cameras": step.cameras,
                "clients": step.concurrency,
            },
        )
        drain_s = await self._drain() if step.cameras and self._s.drain.enabled else None

        stop_clients = asyncio.Event()
        client = self._query_client(step.concurrency) if step.concurrency else None
        cpu0, launched = self._process_time(), self._wall()
        clients: asyncio.Task[list[Sample]] | None = None
        if client is not None:
            clients = asyncio.create_task(
                run_clients(
                    client,
                    self._questions,
                    step.concurrency,
                    stop_clients,
                    wall_now=self._wall,
                    rng=random.Random(seed),
                    think_time_s=sc.query.think_time_s if sc.query else 0.0,
                )
            )
        loadgen: dict[str, Any] | None = None
        try:
            if step.cameras and sc.ingest:
                ingest = await self._ingest.run(
                    IngestRun(
                        cameras=step.cameras,
                        fps=fps,
                        warmup_s=sc.warmup_s,
                        hold_s=sc.hold_s,
                        seed=seed,
                        target_url=self._s.ingest_url,
                        name=f"{sc.name}-r{rep}-s{step.index}",
                        loadgen=sc.ingest.loadgen,
                    )
                )
                start = ingest.started_at.timestamp() + sc.warmup_s
                loadgen = {"run_id": ingest.run_id, **ingest.summary}
                if self._loadgen_config is None:
                    self._loadgen_config = ingest.config
            else:
                await asyncio.sleep(sc.step_s)
                start = launched + sc.warmup_s
        finally:
            stop_clients.set()
            samples = await clients if clients is not None else []
            if client is not None:
                await client.aclose()
        end = start + sc.hold_s
        elapsed = self._wall() - launched
        client_cpu = (self._process_time() - cpu0) / elapsed if elapsed > 0 else None

        settle = end + self._s.scrape_settle_s - self._wall()
        if settle > 0:
            await asyncio.sleep(settle)
        prometheus = await read_window(self._prom, self._s.queries, end, sc.hold_s)

        query = (
            summarize(samples, start, end, round(client_cpu, 3) if client_cpu is not None else None)
            if step.concurrency
            else None
        )
        lost = lost_frames(loadgen, prometheus)
        result = StepResult(
            rep=rep,
            step=step.index,
            cameras=step.cameras,
            fps=fps,
            concurrency=step.concurrency,
            seed=seed,
            window=Window(
                start=datetime.fromtimestamp(start, UTC),
                end=datetime.fromtimestamp(end, UTC),
                seconds=sc.hold_s,
            ),
            drain_s=drain_s,
            loadgen=loadgen,
            query=query,
            prometheus=prometheus,
            lost_frames=lost,
            slo=judge(self._s, loadgen, query, prometheus, lost),
        )
        log.info(
            "step done",
            extra={
                "rep": rep,
                "step": step.index,
                "cameras": step.cameras,
                "clients": step.concurrency,
                "freshness_p95_s": prometheus.get("freshness_p95_s"),
                "lost_frames": lost,
                "query_p95_ms": query["latency_ms"]["p95"] if query else None,
                "slo": result.slo.model_dump(),
            },
        )
        return result

    async def _drain(self) -> float:
        """Wait until the backlog is empty; seconds waited. Raises after `drain.timeout_s`."""
        d = self._s.drain
        t0 = self._wall()
        while True:
            left = await backlog(self._prom, self._wall())
            if left is None or left <= 0:
                return round(self._wall() - t0, 3)
            if self._wall() - t0 >= d.timeout_s:
                raise RuntimeError(f"backlog still {left:.0f} frames after {d.timeout_s} s")
            log.info("waiting for the worker backlog to drain", extra={"backlog": left})
            await asyncio.sleep(d.interval_s)


class WorkerMismatchError(ValueError):
    pass


async def worker_models(prom: Prometheus, at_unix_s: float) -> list[dict[str, str]]:
    """Labels of every worker's `vms_worker_model_info` (job and instance dropped)."""
    rows = await prom.query("vms_worker_model_info", at_unix_s)
    drop = {"__name__", "job", "instance"}
    return sorted(
        ({k: v for k, v in labels.items() if k not in drop} for labels, _ in rows),
        key=lambda d: sorted(d.items()),
    )


def check_workers(workers: list[dict[str, str]], expect: dict[str, str]) -> None:
    if not expect:
        return
    if not workers:
        raise WorkerMismatchError("expect_worker is set but no worker reports its model info")
    for w in workers:
        wrong = {k: w.get(k) for k, v in expect.items() if w.get(k) != v}
        if wrong:
            raise WorkerMismatchError(f"worker {w} does not match expect_worker: {wrong}")


def lost_frames(loadgen: dict[str, Any] | None, prometheus: dict[str, Reading]) -> int | None:
    """Frames that never became detections: not accepted by ingest, or lost by the worker."""
    if loadgen is None:
        return None
    frames = loadgen.get("frames", {})
    at_ingest = sum(int(frames.get(k) or 0) for k in LOSS_FIELDS)
    # No series means the worker never lost a frame since it started.
    in_worker = as_float(prometheus.get("worker_lost_frames")) or 0.0
    return at_ingest + round(in_worker)


def judge(
    settings: BenchSettings,
    loadgen: dict[str, Any] | None,
    query: dict[str, Any] | None,
    prometheus: dict[str, Reading],
    lost: int | None,
) -> Verdict:
    slo = settings.slo
    freshness_ok = loss_ok = backlog_ok = ingest_ok = None
    if loadgen is not None:
        # No freshness data with ingest load means no frame reached the worker: a failure.
        p95 = as_float(prometheus.get("freshness_p95_s"))
        freshness_ok = p95 is not None and p95 < slo.freshness_p95_s
        loss_ok = lost is not None and lost <= slo.max_lost_frames
        slope = as_float(prometheus.get("backlog_slope_fps"))
        backlog_ok = slope is None or slope <= slo.max_backlog_slope_fps
        ingest_ok = freshness_ok and loss_ok and backlog_ok
    query_ok = None
    if query is not None and query["latency_ms"]["p95"] is not None:
        query_ok = query["latency_ms"]["p95"] < slo.query_p95_s * 1000
    harness_ok = True
    if loadgen is not None and loadgen.get("generator_ok") is False:
        harness_ok = False
    cpu = query.get("client_cpu_cores") if query else None
    if cpu is not None and cpu > settings.max_client_cpu_cores:
        harness_ok = False
    return Verdict(
        freshness_ok=freshness_ok,
        loss_ok=loss_ok,
        backlog_ok=backlog_ok,
        ingest_ok=ingest_ok,
        query_ok=query_ok,
        harness_ok=harness_ok,
    )


def aggregate(results: Sequence[StepResult]) -> list[StepAggregate]:
    by_step: dict[int, list[StepResult]] = {}
    for r in results:
        by_step.setdefault(r.step, []).append(r)
    out = []
    for index in sorted(by_step):
        reps = by_step[index]
        dumps = [r.model_dump(mode="json") for r in reps]
        out.append(
            StepAggregate(
                step=index,
                cameras=reps[0].cameras,
                concurrency=reps[0].concurrency,
                reps=len(reps),
                ingest_ok_reps=sum(r.slo.ingest_ok is True for r in reps),
                harness_ok_reps=sum(r.slo.harness_ok for r in reps),
                stats=stats.aggregate(dumps, AGGREGATED),
            )
        )
    return out


def rated_capacity(results: Sequence[StepResult], fps: float) -> RatedCapacity:
    """Largest camera count that, with every smaller one, met the ingest SLO in every run."""
    ok_by_cameras: dict[int, bool] = {}
    for r in results:
        if r.cameras:
            ok_by_cameras[r.cameras] = (
                ok_by_cameras.get(r.cameras, True) and r.slo.ingest_ok is True
            )
    rated: int | None = None
    first_failure: int | None = None
    for cameras in sorted(ok_by_cameras):
        if not ok_by_cameras[cameras]:
            first_failure = cameras
            break
        rated = cameras
    return RatedCapacity(
        fps=fps, cameras=rated, lower_bound=first_failure is None, first_failure=first_failure
    )
