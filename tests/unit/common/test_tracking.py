import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import pytest

from vms.common.clock import FakeClock
from vms.common.settings import TrackingSettings, WandbSettings
from vms.common.tracking import (
    NoopTracker,
    ParamValue,
    RunRef,
    RunStatus,
    RunTracker,
    TrackerError,
    build_tracker,
    flatten_params,
)
from vms.common.tracking.wandb_tracker import WandbTracker, resolve_mode


@dataclass
class RecordingTracker:
    """A fake backend that records every call, optionally failing on start."""

    backend: str
    fail_on_start: bool = False
    calls: list[tuple[Any, ...]] = field(default_factory=list)

    def start_run(
        self,
        name: str,
        *,
        job_type: str = "eval",
        tags: Mapping[str, str] | None = None,
        params: Mapping[str, ParamValue] | None = None,
    ) -> RunRef:
        if self.fail_on_start:
            raise ConnectionError(f"{self.backend} unreachable")
        self.calls.append(("start", name, job_type, dict(tags or {}), dict(params or {})))
        return RunRef(self.backend, f"{self.backend}-1")

    def log_params(self, params: Mapping[str, ParamValue]) -> None:
        self.calls.append(("params", dict(params)))

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        self.calls.append(("metrics", dict(metrics), step))

    def log_artifact_ref(self, name: str, uri: str) -> None:
        self.calls.append(("artifact", name, uri))

    def end_run(self, status: RunStatus = "finished") -> RunRef:
        self.calls.append(("end", status))
        return RunRef(self.backend, f"{self.backend}-1")


# --- NoopTracker -------------------------------------------------------------------------------


def test_noop_lifecycle_returns_ref() -> None:
    tracker = NoopTracker(FakeClock())
    ref = tracker.start_run("eval-structured")
    tracker.log_params({"k": 1})
    tracker.log_metrics({"accuracy": 0.9}, step=1)
    tracker.log_artifact_ref("results", "eval/results/run.json")
    assert tracker.end_run() == ref
    assert ref.backend == "noop"
    assert ref.url is None


def test_noop_enforces_lifecycle() -> None:
    tracker = NoopTracker()
    with pytest.raises(TrackerError, match="no active run"):
        tracker.log_metrics({"x": 1.0})
    with pytest.raises(TrackerError, match="no active run"):
        tracker.end_run()
    tracker.start_run("a")
    with pytest.raises(TrackerError, match="still active"):
        tracker.start_run("b")


def test_noop_run_ids_are_unique() -> None:
    tracker = NoopTracker()
    ids = set()
    for _ in range(3):
        ids.add(tracker.start_run("r").run_id)
        tracker.end_run()
    assert len(ids) == 3


# --- RunTracker --------------------------------------------------------------------------------


def test_fans_out_every_call_in_order() -> None:
    a, b = RecordingTracker("a"), RecordingTracker("b")
    tracker = RunTracker([a, b])
    refs = tracker.start_run("run", job_type="train", tags={"t": "1"}, params={"lr": 0.1})
    tracker.log_params({"seed": 1})
    tracker.log_metrics({"loss": 0.5}, step=3)
    tracker.log_artifact_ref("adapter", "dvc://training/adapter")
    ends = tracker.end_run()
    assert [r.backend for r in refs] == ["a", "b"] == [r.backend for r in ends]
    expected = [
        ("start", "run", "train", {"t": "1"}, {"lr": 0.1}),
        ("params", {"seed": 1}),
        ("metrics", {"loss": 0.5}, 3),
        ("artifact", "adapter", "dvc://training/adapter"),
        ("end", "finished"),
    ]
    assert a.calls == expected == b.calls


def test_context_manager_marks_success_and_failure() -> None:
    rec = RecordingTracker("a")
    tracker = RunTracker([rec])
    with tracker.run("ok") as refs:
        assert refs == [RunRef("a", "a-1")]
    assert rec.calls[-1] == ("end", "finished")
    with pytest.raises(KeyError), tracker.run("bad"):
        raise KeyError("x")
    assert rec.calls[-1] == ("end", "failed")


def test_failed_start_ends_already_started_backends() -> None:
    ok, broken = RecordingTracker("ok"), RecordingTracker("broken", fail_on_start=True)
    with pytest.raises(ConnectionError):
        RunTracker([ok, broken]).start_run("run")
    assert ok.calls[-1] == ("end", "failed")


def test_requires_a_tracker() -> None:
    with pytest.raises(ValueError, match="at least one"):
        RunTracker([])


# --- Selection by config -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("backends", "expected"),
    [
        (["noop"], ["noop"]),
        ([], ["noop"]),
        (["mlflow"], ["mlflow"]),
        (["mlflow", "wandb"], ["mlflow", "wandb"]),
        (["wandb", "wandb"], ["wandb"]),
    ],
)
def test_build_tracker_follows_config(backends: list[Any], expected: list[str]) -> None:
    tracker = build_tracker(TrackingSettings(backends=backends), FakeClock())
    assert tracker.backends == expected


def test_default_config_never_imports_backend_libraries() -> None:
    code = (
        "import sys\n"
        "from vms.common import build_tracker\n"
        "from vms.common.settings import TrackingSettings\n"
        "t = build_tracker(TrackingSettings())\n"
        "t.start_run('r'); t.end_run()\n"
        "print(sorted(m for m in ('mlflow', 'wandb') if m in sys.modules))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=60
    )
    assert out.stdout.strip() == "[]"


def test_flatten_params() -> None:
    nested = {"a": {"b": 1, "c": {"d": "x"}}, "e": [1, 2], "f": None, "g": True}
    assert flatten_params(nested) == {"a.b": 1, "a.c.d": "x", "e": "[1, 2]", "f": None, "g": True}


# --- W&B ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("configured", "env", "expected"),
    [
        (None, {}, "disabled"),
        (None, {"WANDB_MODE": "offline"}, "offline"),
        (None, {"WANDB_MODE": " Online "}, "online"),
        ("disabled", {"WANDB_MODE": "online"}, "disabled"),
    ],
)
def test_wandb_mode_resolution(configured: Any, env: dict[str, str], expected: str) -> None:
    assert resolve_mode(WandbSettings(mode=configured), env) == expected


def test_wandb_invalid_env_mode_fails() -> None:
    with pytest.raises(ValueError, match="WANDB_MODE"):
        resolve_mode(WandbSettings(), {"WANDB_MODE": "dryrun"})


class FakeWandbRun:
    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.id = "wb-1"
        self.url: str | None = None if kwargs["mode"] != "online" else "https://wandb.ai/x/wb-1"
        self.config = SimpleNamespace(
            data=dict(kwargs["config"]),
            update=lambda d, allow_val_change: self.config.data.update(d),
        )
        self.summary: dict[str, Any] = {}
        self.logged: list[tuple[dict[str, Any], int | None]] = []
        self.exit_code: int | None = None

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        self.logged.append((data, step))

    def finish(self, exit_code: int | None = None) -> None:
        self.exit_code = exit_code


def test_wandb_tracker_maps_calls_to_run() -> None:
    runs: list[FakeWandbRun] = []

    def fake_init(**kwargs: Any) -> FakeWandbRun:
        runs.append(FakeWandbRun(**kwargs))
        return runs[-1]

    tracker = WandbTracker(WandbSettings(project="p"), init=fake_init, environ={})
    ref = tracker.start_run("lora-r16", job_type="train", tags={"phase": "8"}, params={"r": 16})
    tracker.log_params({"lr": 1e-4})
    tracker.log_metrics({"loss": 1}, step=10)
    tracker.log_artifact_ref("adapter", "training/adapters/r16.dvc")
    tracker.end_run("failed")

    [run] = runs
    assert ref == RunRef("wandb", "wb-1", None)
    assert run.init_kwargs["mode"] == "disabled"
    assert run.init_kwargs["project"] == "p"
    assert run.init_kwargs["tags"] == ["phase=8"]
    assert run.config.data == {"r": 16, "lr": 1e-4}
    assert run.logged == [({"loss": 1.0}, 10)]
    assert run.summary == {"artifact/adapter": "training/adapters/r16.dvc"}
    assert run.exit_code == 1
    with pytest.raises(TrackerError):
        tracker.log_metrics({"x": 1.0})


# --- MLflow ------------------------------------------------------------------------------------


class FakeMlflowClient:
    def __init__(self, tracking_uri: str, experiments: dict[str, str] | None = None) -> None:
        self.tracking_uri = tracking_uri
        self.experiments = dict(experiments or {})
        self.runs: dict[str, dict[str, Any]] = {}
        self.batches: list[tuple[str, list[Any], list[Any]]] = []

    def get_experiment_by_name(self, name: str) -> Any:
        exp_id = self.experiments.get(name)
        return None if exp_id is None else SimpleNamespace(experiment_id=exp_id)

    def create_experiment(self, name: str) -> str:
        self.experiments[name] = str(len(self.experiments) + 1)
        return self.experiments[name]

    def create_run(
        self, experiment_id: str, start_time: int | None, tags: dict[str, Any], run_name: str
    ) -> Any:
        run_id = f"run{len(self.runs) + 1}"
        self.runs[run_id] = {"exp": experiment_id, "start": start_time, "tags": dict(tags)}
        self.runs[run_id]["name"] = run_name
        return SimpleNamespace(info=SimpleNamespace(run_id=run_id))

    def log_batch(self, run_id: str, metrics: Any = (), params: Any = ()) -> None:
        self.batches.append((run_id, list(metrics), list(params)))

    def set_tag(self, run_id: str, key: str, value: Any) -> None:
        self.runs[run_id]["tags"][key] = value

    def set_terminated(self, run_id: str, status: str, end_time: int | None) -> None:
        self.runs[run_id].update(status=status, end=end_time)


T0 = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
T0_MS = int(T0.timestamp() * 1000)


def test_mlflow_tracker_maps_calls_to_client() -> None:
    pytest.importorskip("mlflow")
    from vms.common.settings import MlflowSettings
    from vms.common.tracking.mlflow_tracker import MlflowTracker

    clock = FakeClock(T0)
    client = FakeMlflowClient("http://mlflow:5000/", experiments={"vms": "7"})
    tracker = MlflowTracker(MlflowSettings(experiment="vms"), clock, client=client)

    ref = tracker.start_run("eval", job_type="eval", tags={"split": "dev"}, params={"k": 10})
    clock.advance(2)
    tracker.log_metrics({"exec_acc": 0.8}, step=None)
    tracker.log_artifact_ref("results", "eval/results/x.json")
    tracker.end_run("failed")

    assert ref == RunRef("mlflow", "run1", "http://mlflow:5000/#/experiments/7/runs/run1")
    run = client.runs["run1"]
    assert run["exp"] == "7"
    assert run["start"] == T0_MS
    assert run["end"] == T0_MS + 2000
    assert run["status"] == "FAILED"
    assert run["tags"] == {
        "job_type": "eval",
        "split": "dev",
        "artifact.results": "eval/results/x.json",
    }
    _, _, params = client.batches[0]
    assert [(p.key, p.value) for p in params] == [("k", "10")]
    _, metrics, _ = client.batches[1]
    assert [(m.key, m.value, m.timestamp, m.step) for m in metrics] == [
        ("exec_acc", 0.8, T0_MS + 2000, 0)
    ]


def test_mlflow_creates_experiment_chunks_params_and_has_no_url_for_files() -> None:
    pytest.importorskip("mlflow")
    from vms.common.settings import MlflowSettings
    from vms.common.tracking.mlflow_tracker import MAX_PARAMS_PER_BATCH, MlflowTracker

    client = FakeMlflowClient("file:///tmp/mlruns")
    tracker = MlflowTracker(MlflowSettings(experiment="new"), FakeClock(T0), client=client)
    ref = tracker.start_run("eval")
    tracker.log_params({f"p{i}": i for i in range(250)})
    tracker.end_run()

    assert client.experiments == {"new": "1"}
    assert ref.url is None
    sizes = [len(params) for _, _, params in client.batches]
    assert sizes == [MAX_PARAMS_PER_BATCH, MAX_PARAMS_PER_BATCH, 50]
    assert client.runs["run1"]["status"] == "FINISHED"
