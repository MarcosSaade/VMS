"""Experiment tracking interface.

`Tracker` is the per-backend protocol: start a run, log params, metrics and artifact references,
end the run. `RunTracker` fans calls out to every backend enabled in config and returns one
`RunRef` per backend, which the run's results JSON records. That JSON stays the source of truth
for reported numbers; trackers are for browsing, comparing and the model registry.

Roles are fixed by convention: W&B for training runs, sweeps and reports; MLflow for eval results,
the model registry and champion tracking. Both may be enabled for one run.

Artifacts are logged by reference (a DVC path, S3 URI or results file path), never uploaded:
DVC owns large artifacts, and the reference is enough to find them.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from vms.common.clock import Clock, SystemClock
from vms.common.ids import UlidGenerator
from vms.common.settings import TrackingSettings

ParamValue = str | int | float | bool | None
RunStatus = Literal["finished", "failed"]


@dataclass(frozen=True)
class RunRef:
    """Where a run lives in one tracking backend."""

    backend: str
    run_id: str
    url: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {"backend": self.backend, "run_id": self.run_id, "url": self.url}


class TrackerError(RuntimeError):
    """Misuse of the run lifecycle, e.g. logging without an active run."""


class Tracker(Protocol):
    backend: str

    def start_run(
        self, name: str, *, job_type: str, tags: Mapping[str, str], params: Mapping[str, ParamValue]
    ) -> RunRef: ...

    def log_params(self, params: Mapping[str, ParamValue]) -> None: ...

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None: ...

    def log_artifact_ref(self, name: str, uri: str) -> None: ...

    def end_run(self, status: RunStatus = "finished") -> RunRef: ...


class _RunState:
    """Shared lifecycle bookkeeping so every backend rejects the same misuse."""

    def __init__(self) -> None:
        self._ref: RunRef | None = None

    def require(self) -> RunRef:
        """The active run; raises if there is none."""
        if self._ref is None:
            raise TrackerError("no active run; call start_run first")
        return self._ref

    def start(self, ref: RunRef) -> None:
        if self._ref is not None:
            raise TrackerError(f"run {self._ref.run_id} is still active; end it first")
        self._ref = ref

    def ensure_idle(self) -> None:
        if self._ref is not None:
            raise TrackerError(f"run {self._ref.run_id} is still active; end it first")

    def end(self) -> RunRef:
        ref = self.require()
        self._ref = None
        return ref


class NoopTracker:
    """Records nothing; enforces the lifecycle so misuse fails in tests too."""

    backend = "noop"

    def __init__(self, clock: Clock | None = None) -> None:
        self._ids = UlidGenerator(clock or SystemClock())
        self._state = _RunState()

    def start_run(
        self,
        name: str,
        *,
        job_type: str = "eval",
        tags: Mapping[str, str] | None = None,
        params: Mapping[str, ParamValue] | None = None,
    ) -> RunRef:
        self._state.ensure_idle()
        ref = RunRef(self.backend, self._ids.new())
        self._state.start(ref)
        return ref

    def log_params(self, params: Mapping[str, ParamValue]) -> None:
        self._state.require()

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        self._state.require()

    def log_artifact_ref(self, name: str, uri: str) -> None:
        self._state.require()

    def end_run(self, status: RunStatus = "finished") -> RunRef:
        return self._state.end()


class RunTracker:
    """Fans each call out to all configured backends.

    If a backend fails to start, backends already started are ended as `failed` before the error
    propagates, so no run is left dangling.
    """

    def __init__(self, trackers: Sequence[Tracker]) -> None:
        if not trackers:
            raise ValueError("RunTracker needs at least one tracker (use NoopTracker)")
        self.trackers = list(trackers)

    @property
    def backends(self) -> list[str]:
        return [t.backend for t in self.trackers]

    def start_run(
        self,
        name: str,
        *,
        job_type: str = "eval",
        tags: Mapping[str, str] | None = None,
        params: Mapping[str, ParamValue] | None = None,
    ) -> list[RunRef]:
        refs: list[RunRef] = []
        started: list[Tracker] = []
        for tracker in self.trackers:
            try:
                refs.append(
                    tracker.start_run(name, job_type=job_type, tags=tags or {}, params=params or {})
                )
            except Exception:
                for done in started:
                    done.end_run("failed")
                raise
            started.append(tracker)
        return refs

    def log_params(self, params: Mapping[str, ParamValue]) -> None:
        for tracker in self.trackers:
            tracker.log_params(params)

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        for tracker in self.trackers:
            tracker.log_metrics(metrics, step)

    def log_artifact_ref(self, name: str, uri: str) -> None:
        for tracker in self.trackers:
            tracker.log_artifact_ref(name, uri)

    def end_run(self, status: RunStatus = "finished") -> list[RunRef]:
        return [tracker.end_run(status) for tracker in self.trackers]

    @contextmanager
    def run(
        self,
        name: str,
        *,
        job_type: str = "eval",
        tags: Mapping[str, str] | None = None,
        params: Mapping[str, ParamValue] | None = None,
    ) -> Iterator[list[RunRef]]:
        """Start a run, end it as `finished`, or as `failed` if the block raises."""
        refs = self.start_run(name, job_type=job_type, tags=tags, params=params)
        try:
            yield refs
        except BaseException:
            self.end_run("failed")
            raise
        self.end_run("finished")


def build_tracker(settings: TrackingSettings, clock: Clock | None = None) -> RunTracker:
    """Create the trackers named in `settings.backends`. Switching backends is a config change.

    Backend libraries are imported only when their backend is enabled.
    """
    clock = clock or SystemClock()
    trackers: list[Tracker] = []
    for backend in dict.fromkeys(settings.backends):  # dedupe, keep order
        if backend == "noop":
            trackers.append(NoopTracker(clock))
        elif backend == "mlflow":
            from vms.common.tracking.mlflow_tracker import MlflowTracker

            trackers.append(MlflowTracker(settings.mlflow, clock))
        elif backend == "wandb":
            from vms.common.tracking.wandb_tracker import WandbTracker

            trackers.append(WandbTracker(settings.wandb))
    return RunTracker(trackers or [NoopTracker(clock)])


def flatten_params(params: Mapping[str, Any], prefix: str = "") -> dict[str, ParamValue]:
    """Flatten nested config (e.g. `settings.model_dump()`) into dotted keys for logging.

    Lists and other non-scalar values become their string form.
    """
    flat: dict[str, ParamValue] = {}
    for key, value in params.items():
        full = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten_params(value, f"{full}."))
        elif value is None or isinstance(value, str | int | float | bool):
            flat[full] = value
        else:
            flat[full] = str(value)
    return flat
