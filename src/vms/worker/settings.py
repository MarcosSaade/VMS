"""Settings for the inference worker, loaded from `configs/worker.yaml` (see vms.common.settings).

Connection identity and secrets come from POSTGRES_* and S3_* (vms.storage.config); this file
holds the worker's behavior and tuning.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vms.common.settings import ServiceSettings
from vms.storage.frames import PoolTuning
from vms.storage.objects import ObjectStoreTuning
from vms.worker.detector import DetectorSettings
from vms.worker.tracking import TrackerSettings
from vms.worker.zones import ZoneSettings


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HttpServerSettings(_Section):
    host: str = "127.0.0.1"
    port: int = Field(default=8090, ge=1, le=65535)


class QueueSettings(_Section):
    """How the worker takes work from the Postgres queue (D3, D20)."""

    max_cameras: int = Field(default=16, ge=1)  # cameras leased at once by one worker
    lease_ttl_s: float = Field(default=30.0, gt=0)  # a dead worker's cameras move after this
    renew_every_s: float = Field(default=5.0, gt=0)
    # A leased camera with nothing claimed for this long and no pending frames is handed back
    # at the next renewal, so a worker at max_cameras makes room for cameras that start later.
    idle_release_s: float = Field(default=60.0, gt=0)
    claim_batch: int = Field(default=4, ge=1)  # frames per claim query
    # Frames per detect_batch call (task 5.5): claimed frames, of any leased cameras, are
    # detected together up to this many; 1 is one frame at a time. At most claim_batch.
    infer_batch: int = Field(default=1, ge=1)
    max_attempts: int = Field(default=3, ge=1)  # then the frame is marked failed
    # Idle polling backs off from min to max while no frames arrive.
    poll_min_s: float = Field(default=0.05, gt=0)
    poll_max_s: float = Field(default=1.0, gt=0)
    backlog_every_s: float = Field(default=5.0, gt=0)  # pending-frame count for the gauge
    # On SIGTERM: finish the current frame, then hand back the rest, within this budget.
    shutdown_grace_s: float = Field(default=10.0, gt=0)

    @model_validator(mode="after")
    def _check(self) -> QueueSettings:
        if self.poll_min_s > self.poll_max_s:
            raise ValueError("poll_min_s must not exceed poll_max_s")
        if self.renew_every_s >= self.lease_ttl_s:
            raise ValueError("renew_every_s must be shorter than lease_ttl_s")
        if self.infer_batch > self.claim_batch:
            raise ValueError("infer_batch must not exceed claim_batch")
        return self


class WorkerSettings(ServiceSettings):
    http: HttpServerSettings = Field(default_factory=HttpServerSettings)
    queue: QueueSettings = Field(default_factory=QueueSettings)
    detector: DetectorSettings = Field(default_factory=DetectorSettings)
    tracker: TrackerSettings = Field(default_factory=TrackerSettings)
    zones: ZoneSettings = Field(default_factory=ZoneSettings)
    db: PoolTuning = Field(default_factory=PoolTuning)
    objects: ObjectStoreTuning = Field(default_factory=ObjectStoreTuning)

    @model_validator(mode="after")
    def _detector_feeds_tracker(self) -> WorkerSettings:
        # Boxes the detector drops never reach ByteTrack's low-confidence round.
        if self.detector.conf > self.tracker.low_thresh:
            raise ValueError("detector.conf must not exceed tracker.low_thresh")
        return self
