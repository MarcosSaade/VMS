from collections.abc import Callable, Coroutine
from typing import Any

import pytest

from tests.unit.bench.conftest import WALL0, FakeIngest, FakePrometheus, FakeQueryClient, at
from vms.bench.queryload import QueryClient, Reply
from vms.bench.runner import (
    BenchRunner,
    E2EMetrics,
    StepResult,
    Verdict,
    Window,
    WorkerMismatchError,
    judge,
    lost_frames,
    rated_capacity,
)
from vms.bench.scenario import BenchScenario
from vms.bench.settings import BenchSettings
from vms.common.clock import FakeClock

VRun = Callable[[Coroutine[Any, Any, Any]], Any]
HEALTHY = {
    "freshness_p50_s": 0.4,
    "freshness_p95_s": 0.9,
    "worker_done_fps": 5.0,
    "backlog_slope_fps": 0.0,
    "cpu_cores": {"worker": 3.5, "ingest": 0.2, "query": 0.1},
}


def scenario(**overrides: Any) -> BenchScenario:
    data: dict[str, Any] = {
        "name": "t",
        "kind": "e2e",
        "repeats": 2,
        "warmup_s": 5,
        "hold_s": 20,
        "ingest": {"cameras": [1, 2, 4], "fps": 5},
        "query": {"concurrency": [1]},
        "seed": 100,
    }
    data.update(overrides)
    return BenchScenario.model_validate(data)


def make_runner(
    sc: BenchScenario,
    settings: BenchSettings,
    ingest: FakeIngest,
    prom: FakePrometheus,
    clients: list[FakeQueryClient] | None = None,
) -> BenchRunner:
    made = clients if clients is not None else []

    def factory(concurrency: int) -> QueryClient:
        made.append(FakeQueryClient(latency_s=0.25))
        return made[-1]

    return BenchRunner(
        sc,
        settings,
        ingest=ingest,
        query_client=factory,
        questions=["q1", "q2"],
        prom=prom,
        clock=FakeClock(WALL0),
        process_time=lambda: 0.0,
    )


def test_ramp_measures_each_step_over_its_steady_window(
    vrun: VRun, settings: BenchSettings
) -> None:
    ingest = FakeIngest(startup_s=2.0)

    def values(name: str) -> Any:
        cameras = ingest.runs[-1].cameras
        if name == "freshness_p95_s":
            return 0.9 if cameras < 4 else 3.5  # 4 cameras break freshness
        return HEALTHY.get(name)

    prom = FakePrometheus(settings, values)
    clients: list[FakeQueryClient] = []
    metrics: E2EMetrics = vrun(make_runner(scenario(), settings, ingest, prom, clients).run())

    assert [(r.rep, r.cameras) for r in metrics.steps] == [
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 1),
        (1, 2),
        (1, 4),
    ]
    first = metrics.steps[0]
    # Loadgen's schedule starts 2 s after launch (clip loading); its warmup is 5 s.
    assert first.window == Window(start=at(7), end=at(27), seconds=20)
    # The window is read once the last scrape has landed, at its end, over 20 s.
    window_reads = [(q, t) for q, t in prom.calls if "[20s]" in q]
    assert window_reads[0][1] == at(27).timestamp()
    assert first.prometheus["freshness_p95_s"] == 0.9
    assert first.prometheus["cpu_cores"] == {"ingest": 0.2, "query": 0.1, "worker": 3.5}
    # Query clients ran the whole step; only questions started in the window count (4 per s).
    assert first.query is not None
    assert first.query["questions"] == 80
    assert first.query["outcomes"] == {"ok": 80}
    assert all(c.closed for c in clients)
    # Seeds differ per step and repetition, derived from the scenario seed.
    assert [r.seed for r in metrics.steps] == [100, 101, 102, 1100, 1101, 1102]
    assert [run.seed for run in ingest.runs] == [100, 101, 102, 1100, 1101, 1102]
    assert ingest.runs[0].target_url == settings.ingest_url

    assert [r.slo.ingest_ok for r in metrics.steps] == [True, True, False] * 2
    assert metrics.steps[2].slo.freshness_ok is False
    assert metrics.steps[2].slo.loss_ok is True
    assert metrics.rated_capacity is not None
    assert metrics.rated_capacity.cameras == 2
    assert metrics.rated_capacity.first_failure == 4
    assert not metrics.rated_capacity.lower_bound

    agg = {a.cameras: a for a in metrics.aggregate}
    assert agg[1].reps == 2
    assert agg[1].ingest_ok_reps == 2
    assert agg[4].ingest_ok_reps == 0
    assert agg[4].stats["prometheus.freshness_p95_s"] == {
        "mean": 3.5,
        "std": 0.0,
        "min": 3.5,
        "max": 3.5,
    }
    assert agg[1].stats["loadgen.offered_fps"]["mean"] == 5.0
    assert E2EMetrics.model_validate(metrics.model_dump(mode="json")) == metrics


def test_each_ingest_step_waits_for_the_backlog_to_drain(
    vrun: VRun, settings: BenchSettings
) -> None:
    ingest = FakeIngest()
    # First drain: 30, 12, 0 frames left (two waits); the window read then sees 0.
    prom = FakePrometheus(settings, HEALTHY.get, backlog=[30.0, 12.0, 0.0])
    sc = scenario(repeats=1, ingest={"cameras": [1]}, query=None)
    metrics: E2EMetrics = vrun(make_runner(sc, settings, ingest, prom).run())
    assert metrics.steps[0].drain_s == 2 * settings.drain.interval_s
    assert metrics.steps[0].query is None


def test_drain_gives_up_after_its_timeout(vrun: VRun, settings: BenchSettings) -> None:
    settings = settings.model_copy(
        update={"drain": settings.drain.model_copy(update={"timeout_s": 10.0})}
    )
    prom = FakePrometheus(settings, HEALTHY.get, backlog=[50.0])
    sc = scenario(repeats=1, ingest={"cameras": [1]}, query=None)
    with pytest.raises(RuntimeError, match="backlog still 50"):
        vrun(make_runner(sc, settings, FakeIngest(), prom).run())


def test_query_only_step_has_no_ingest_verdict(vrun: VRun, settings: BenchSettings) -> None:
    prom = FakePrometheus(settings, HEALTHY.get, backlog=[99.0])  # never drained: no ingest
    sc = scenario(repeats=1, ingest=None, query={"concurrency": [2]})
    metrics: E2EMetrics = vrun(make_runner(sc, settings, FakeIngest(), prom).run())
    step = metrics.steps[0]
    assert step.window == Window(start=at(5), end=at(25), seconds=20)
    assert step.loadgen is None
    assert step.lost_frames is None
    assert step.slo.ingest_ok is None
    assert step.slo.query_ok is True  # 250 ms < 1.5 s
    assert step.query is not None
    assert step.query["questions"] == 160
    assert metrics.rated_capacity is None


def test_too_short_a_hold_is_refused(vrun: VRun, settings: BenchSettings) -> None:
    sc = scenario(hold_s=5)
    with pytest.raises(ValueError, match="min_window_s"):
        vrun(make_runner(sc, settings, FakeIngest(), FakePrometheus(settings, HEALTHY.get)).run())


def test_stop_finishes_the_current_step_and_marks_the_run(
    vrun: VRun, settings: BenchSettings
) -> None:
    ingest = FakeIngest()
    runner = make_runner(scenario(), settings, ingest, FakePrometheus(settings, HEALTHY.get))
    original = ingest.run

    async def run_then_stop(spec: Any) -> Any:
        runner.stop()
        return await original(spec)

    ingest.run = run_then_stop  # type: ignore[method-assign]
    metrics: E2EMetrics = vrun(runner.run())
    assert len(metrics.steps) == 1
    assert metrics.interrupted


def _loadgen(**frames: int) -> dict[str, Any]:
    return {"frames": {"sent": 100, "accepted": 100, **frames}, "generator_ok": True}


def test_lost_frames_counts_ingest_and_worker_losses() -> None:
    assert lost_frames(None, {}) is None
    assert lost_frames(_loadgen(), {}) == 0  # no worker series: nothing lost
    assert lost_frames(_loadgen(dropped=2, rejected=1, failed=1, incomplete=1), {}) == 5
    assert lost_frames(_loadgen(), {"worker_lost_frames": 2.6}) == 3


@pytest.mark.parametrize(
    ("prometheus", "lost", "expected"),
    [
        ({"freshness_p95_s": 1.99, "backlog_slope_fps": 0.1}, 0, (True, True, True, True)),
        ({"freshness_p95_s": 2.0, "backlog_slope_fps": 0.1}, 0, (False, True, True, False)),
        ({"freshness_p95_s": None}, 0, (False, True, True, False)),  # nothing processed
        ({"freshness_p95_s": 0.5}, 1, (True, False, True, False)),
        ({"freshness_p95_s": 0.5, "backlog_slope_fps": 3.0}, 0, (True, True, False, False)),
    ],
)
def test_ingest_verdict(
    settings: BenchSettings,
    prometheus: dict[str, Any],
    lost: int,
    expected: tuple[bool, bool, bool, bool],
) -> None:
    v = judge(settings, _loadgen(), None, prometheus, lost)
    assert (v.freshness_ok, v.loss_ok, v.backlog_ok, v.ingest_ok) == expected
    assert v.query_ok is None
    assert v.harness_ok


def test_harness_verdict(settings: BenchSettings) -> None:
    behind = {**_loadgen(), "generator_ok": False}
    assert not judge(settings, behind, None, {}, 0).harness_ok
    busy = {"latency_ms": {"p95": 100.0}, "client_cpu_cores": 0.9}
    v = judge(settings, None, busy, {}, None)
    assert not v.harness_ok
    assert v.query_ok is True


def _step(cameras: int, ok: bool, rep: int = 0) -> StepResult:
    return StepResult(
        rep=rep,
        step=cameras,
        cameras=cameras,
        fps=5.0,
        concurrency=0,
        seed=0,
        window=Window(start=at(0), end=at(1), seconds=1),
        drain_s=None,
        loadgen=None,
        query=None,
        prometheus={},
        lost_frames=0,
        slo=Verdict(
            freshness_ok=ok,
            loss_ok=True,
            backlog_ok=True,
            ingest_ok=ok,
            query_ok=None,
            harness_ok=True,
        ),
    )


def test_rated_capacity_needs_every_repetition_and_every_smaller_step() -> None:
    rc = rated_capacity([_step(1, True), _step(2, True), _step(4, True)], 5.0)
    assert (rc.cameras, rc.lower_bound, rc.first_failure) == (4, True, None)
    # One bad repetition of 4 fails the step.
    rc = rated_capacity([_step(1, True), _step(4, True), _step(4, False, rep=1)], 5.0)
    assert (rc.cameras, rc.first_failure) == (1, 4)
    # A pass above a failure does not count.
    rc = rated_capacity([_step(1, True), _step(2, False), _step(4, True)], 5.0)
    assert (rc.cameras, rc.first_failure) == (1, 2)
    rc = rated_capacity([_step(1, False)], 5.0)
    assert (rc.cameras, rc.first_failure) == (None, 1)


def test_failed_replies_count_as_query_errors(vrun: VRun, settings: BenchSettings) -> None:
    sc = scenario(repeats=1, ingest=None, query={"concurrency": [1]})
    runner = BenchRunner(
        sc,
        settings,
        ingest=FakeIngest(),
        query_client=lambda n: FakeQueryClient(0.5, Reply("failed:timeout")),
        questions=["q"],
        prom=FakePrometheus(settings, HEALTHY.get),
        clock=FakeClock(WALL0),
        process_time=lambda: 0.0,
    )
    metrics: E2EMetrics = vrun(runner.run())
    assert metrics.steps[0].query is not None
    assert metrics.steps[0].query["error_rate"] == 1.0


def test_shipped_config_has_the_queries_the_verdict_reads(settings: BenchSettings) -> None:
    assert {"freshness_p95_s", "worker_lost_frames", "backlog_slope_fps"} <= settings.queries.keys()
    assert all("$window" in q.query or "[" not in q.query for q in settings.queries.values())
    assert settings.min_window_s >= 3 * 5  # three scrapes at the services' 5 s interval


ORT = {"model_version": "yolo11n@0ebbc80d4a76+onnx", "backend": "onnxruntime", "infer_batch": "4"}


def test_the_workers_model_is_recorded(vrun: VRun, settings: BenchSettings) -> None:
    prom = FakePrometheus(settings, HEALTHY.get, workers=[ORT])
    metrics: E2EMetrics = vrun(make_runner(scenario(), settings, FakeIngest(), prom).run())
    assert metrics.workers == [ORT]  # job and instance labels dropped


def test_a_worker_that_does_not_match_expect_worker_stops_the_run(
    vrun: VRun, settings: BenchSettings
) -> None:
    sc = scenario(expect_worker={"backend": "onnxruntime", "infer_batch": "1"})
    ingest = FakeIngest()
    prom = FakePrometheus(settings, HEALTHY.get, workers=[ORT])
    with pytest.raises(WorkerMismatchError, match="infer_batch"):
        vrun(make_runner(sc, settings, ingest, prom).run())
    with pytest.raises(WorkerMismatchError, match="no worker"):
        vrun(make_runner(sc, settings, ingest, FakePrometheus(settings, HEALTHY.get)).run())
    ok = scenario(expect_worker={"backend": "onnxruntime"})
    assert vrun(make_runner(ok, settings, ingest, prom).run()).workers == [ORT]
