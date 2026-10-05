"""Weights & Biases tracker: training runs, sweeps and reports.

`wandb` is imported only when a run starts, and the mode defaults to `disabled`: the run object
works but nothing is sent or written, so no W&B account is needed unless a config or `WANDB_MODE`
asks for `offline` (local files, sync later) or `online`.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from typing import Any, Final, Literal, Protocol

from vms.common.settings import WandbSettings
from vms.common.tracking.base import ParamValue, RunRef, RunStatus, _RunState

ARTIFACT_SUMMARY_PREFIX: Final = "artifact/"
WandbMode = Literal["disabled", "offline", "online"]
_VALID_MODES: Final = ("disabled", "offline", "online")


class WandbRunLike(Protocol):
    """The subset of `wandb.Run` this tracker uses."""

    @property
    def id(self) -> str: ...

    @property
    def url(self) -> str | None: ...

    @property
    def config(self) -> Any: ...

    @property
    def summary(self) -> Any: ...

    def log(self, data: dict[str, Any], step: int | None = None) -> None: ...

    def finish(self, exit_code: int | None = None) -> None: ...


def resolve_mode(settings: WandbSettings, environ: Mapping[str, str] | None = None) -> WandbMode:
    """Config wins, then `WANDB_MODE`, then `disabled`."""
    environ = os.environ if environ is None else environ
    if settings.mode is not None:
        return settings.mode
    from_env = environ.get("WANDB_MODE", "").strip().lower()
    if not from_env:
        return "disabled"
    if from_env not in _VALID_MODES:
        raise ValueError(f"WANDB_MODE must be one of {_VALID_MODES}, got {from_env!r}")
    return from_env  # type: ignore[return-value]


class WandbTracker:
    backend = "wandb"

    def __init__(
        self,
        settings: WandbSettings,
        init: Callable[..., WandbRunLike] | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self._settings = settings
        self._init = init
        self.mode = resolve_mode(settings, environ)
        self._state = _RunState()
        self._run: WandbRunLike | None = None

    def start_run(
        self,
        name: str,
        *,
        job_type: str = "train",
        tags: Mapping[str, str] | None = None,
        params: Mapping[str, ParamValue] | None = None,
    ) -> RunRef:
        self._state.ensure_idle()
        init = self._init or _lazy_wandb_init()
        run = init(
            project=self._settings.project,
            entity=self._settings.entity,
            name=name,
            job_type=job_type,
            mode=self.mode,
            # W&B tags are plain strings; keep key=value so they stay searchable.
            tags=[f"{key}={value}" for key, value in (tags or {}).items()],
            config=dict(params or {}),
            reinit="create_new",
        )
        self._run = run
        self._state.start(RunRef(self.backend, run.id, run.url or None))
        return self._state.require()

    def log_params(self, params: Mapping[str, ParamValue]) -> None:
        self._active().config.update(dict(params), allow_val_change=True)

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        self._active().log({key: float(value) for key, value in metrics.items()}, step=step)

    def log_artifact_ref(self, name: str, uri: str) -> None:
        self._active().summary[f"{ARTIFACT_SUMMARY_PREFIX}{name}"] = uri

    def end_run(self, status: RunStatus = "finished") -> RunRef:
        self._active().finish(exit_code=0 if status == "finished" else 1)
        self._run = None
        return self._state.end()

    def _active(self) -> WandbRunLike:
        self._state.require()
        assert self._run is not None
        return self._run


def _lazy_wandb_init() -> Callable[..., WandbRunLike]:
    import wandb

    init: Callable[..., WandbRunLike] = wandb.init
    return init
