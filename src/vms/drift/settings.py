"""Settings for the drift monitor, loaded from `configs/drift.yaml` (see vms.common.settings).

Connection identity and secrets come from POSTGRES_* (vms.storage.config).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vms.common.settings import ServiceSettings
from vms.drift.detect import Calibration, DetectorSettings
from vms.drift.signals import SignalSettings
from vms.storage.frames import PoolTuning


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HttpServerSettings(_Section):
    host: str = "127.0.0.1"
    port: int = Field(default=8095, ge=1, le=65535)


class MonitorSettings(_Section):
    """Windows are in capture time and end `settle_s` before now, so frames still in the
    worker's queue (freshness p95 is well under a second at the rated load) are counted."""

    window_s: float = Field(default=30.0, gt=0)
    reference_s: float = Field(default=60.0, gt=0)  # calibration period of a new camera
    update_every_s: float = Field(default=5.0, gt=0)
    settle_s: float = Field(default=3.0, ge=0)
    discover_every_s: float = Field(default=30.0, gt=0)  # look for new cameras
    shutdown_grace_s: float = Field(default=5.0, gt=0)
    # The detector only sees windows whose frames span at least this share of window_s: the
    # first windows after a camera starts or resumes (and the last as it stops) hold a few
    # seconds of frames, unlike every window the clean-run calibration saw (task 9.3, E-055).
    detect_min_coverage: float = Field(default=0.9, ge=0, le=1)

    @model_validator(mode="after")
    def _check(self) -> MonitorSettings:
        if self.update_every_s > self.window_s:
            raise ValueError("update_every_s must not exceed window_s (frames would be skipped)")
        return self


class DriftSettings(ServiceSettings):
    http: HttpServerSettings = Field(default_factory=HttpServerSettings)
    monitor: MonitorSettings = Field(default_factory=MonitorSettings)
    signals: SignalSettings = Field(default_factory=SignalSettings)
    db: PoolTuning = Field(default_factory=PoolTuning)
    # Alerting (task 9.3): the chosen operating point and the clean-run calibration it was
    # chosen with (from `python -m vms.drift score`). Without a calibration, no detection.
    detector: DetectorSettings = Field(default_factory=DetectorSettings)
    calibration: Calibration | None = None
