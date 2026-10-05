"""MLflow tracker: eval results, model registry and champion tracking.

Uses `MlflowClient` with an explicit run ID rather than the `mlflow.start_run()` fluent API, which
keeps an implicit global "active run" that is easy to leak between runs in one process. The
client is injectable, so tests use a fake and never talk to a server.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Final, Protocol

from vms.common.clock import Clock
from vms.common.settings import MlflowSettings
from vms.common.tracking.base import ParamValue, RunRef, RunStatus, _RunState

# MLflow's server-side limits per log_batch request.
MAX_PARAMS_PER_BATCH: Final = 100
MAX_METRICS_PER_BATCH: Final = 1000
ARTIFACT_TAG_PREFIX: Final = "artifact."
_STATUS = {"finished": "FINISHED", "failed": "FAILED"}


class MlflowClientLike(Protocol):
    """The subset of `mlflow.tracking.MlflowClient` this tracker uses."""

    @property
    def tracking_uri(self) -> str: ...

    def get_experiment_by_name(self, name: str) -> Any: ...

    def create_experiment(self, name: str) -> str: ...

    def create_run(
        self, experiment_id: str, start_time: int | None, tags: dict[str, Any], run_name: str
    ) -> Any: ...

    def log_batch(self, run_id: str, metrics: Sequence[Any], params: Sequence[Any]) -> Any: ...

    def set_tag(self, run_id: str, key: str, value: Any) -> Any: ...

    def set_terminated(self, run_id: str, status: str, end_time: int | None) -> None: ...


class MlflowTracker:
    backend = "mlflow"

    def __init__(
        self, settings: MlflowSettings, clock: Clock, client: MlflowClientLike | None = None
    ) -> None:
        self._settings = settings
        self._clock = clock
        self._client = client
        self._state = _RunState()
        self._experiment_id: str | None = None

    def start_run(
        self,
        name: str,
        *,
        job_type: str = "eval",
        tags: Mapping[str, str] | None = None,
        params: Mapping[str, ParamValue] | None = None,
    ) -> RunRef:
        self._state.ensure_idle()
        client = self._get_client()
        experiment_id = self._experiment(client)
        all_tags = {"job_type": job_type, **(tags or {})}
        run = client.create_run(experiment_id, self._now_ms(), all_tags, name)
        run_id: str = run.info.run_id
        self._state.start(
            RunRef(self.backend, run_id, self._run_url(client, experiment_id, run_id))
        )
        if params:
            self.log_params(params)
        return self._state.require()

    def log_params(self, params: Mapping[str, ParamValue]) -> None:
        from mlflow.entities import Param

        run_id = self._state.require().run_id
        items = [Param(key, str(value)) for key, value in params.items()]
        for chunk in _chunks(items, MAX_PARAMS_PER_BATCH):
            self._get_client().log_batch(run_id, metrics=[], params=chunk)

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        from mlflow.entities import Metric

        run_id = self._state.require().run_id
        now = self._now_ms()
        items = [Metric(key, float(value), now, step or 0) for key, value in metrics.items()]
        for chunk in _chunks(items, MAX_METRICS_PER_BATCH):
            self._get_client().log_batch(run_id, metrics=chunk, params=[])

    def log_artifact_ref(self, name: str, uri: str) -> None:
        run_id = self._state.require().run_id
        self._get_client().set_tag(run_id, f"{ARTIFACT_TAG_PREFIX}{name}", uri)

    def end_run(self, status: RunStatus = "finished") -> RunRef:
        ref = self._state.require()
        self._get_client().set_terminated(ref.run_id, _STATUS[status], self._now_ms())
        return self._state.end()

    def _get_client(self) -> MlflowClientLike:
        if self._client is None:
            from mlflow.tracking import MlflowClient

            # tracking_uri=None makes MLflow use MLFLOW_TRACKING_URI, then its local default.
            self._client = MlflowClient(tracking_uri=self._settings.tracking_uri)
        return self._client

    def _experiment(self, client: MlflowClientLike) -> str:
        if self._experiment_id is None:
            existing = client.get_experiment_by_name(self._settings.experiment)
            self._experiment_id = (
                str(existing.experiment_id)
                if existing is not None
                else client.create_experiment(self._settings.experiment)
            )
        return self._experiment_id

    def _now_ms(self) -> int:
        return int(self._clock.now().timestamp() * 1000)

    @staticmethod
    def _run_url(client: MlflowClientLike, experiment_id: str, run_id: str) -> str | None:
        uri = client.tracking_uri
        if uri.startswith(("http://", "https://")):
            return f"{uri.rstrip('/')}/#/experiments/{experiment_id}/runs/{run_id}"
        return None


def _chunks[T](items: list[T], size: int) -> Iterator[list[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]
