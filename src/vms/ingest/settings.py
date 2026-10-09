"""Settings for the ingest service, loaded from `configs/ingest.yaml` (see vms.common.settings).

Connection identity and secrets come from POSTGRES_* and S3_* (vms.storage.config); this file
holds the service's own behavior and tuning.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from vms.common.settings import ServiceSettings
from vms.storage.frames import PoolTuning
from vms.storage.objects import ObjectStoreTuning


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HttpServerSettings(_Section):
    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    # Seconds to finish in-flight requests after SIGTERM before connections are closed.
    graceful_shutdown_s: float = Field(default=10.0, gt=0)


class FrameLimits(_Section):
    max_jpeg_bytes: int = Field(default=4 * 1024 * 1024, ge=1)
    # Reject capture timestamps this far ahead of the server clock (a broken camera clock).
    max_future_skew_s: float = Field(default=300.0, ge=0)


class IngestSettings(ServiceSettings):
    http: HttpServerSettings = Field(default_factory=HttpServerSettings)
    limits: FrameLimits = Field(default_factory=FrameLimits)
    db: PoolTuning = Field(default_factory=PoolTuning)
    objects: ObjectStoreTuning = Field(default_factory=ObjectStoreTuning)
