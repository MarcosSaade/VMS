"""Settings for the load generator, loaded from `configs/loadgen.yaml` (see vms.common.settings).

A benchmark scenario (`bench/scenarios/*.yaml`) overrides fields under its `config:` key.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vms.common.settings import ServiceSettings
from vms.loadgen.drift import DriftInjectionSettings


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FrameSettings(_Section):
    """What each sent frame looks like. Frames are prepared once at startup, not per send."""

    # Resize to exactly width x height; both null keeps each clip's own size.
    width: int | None = Field(default=1280, ge=16)
    height: int | None = Field(default=720, ge=16)
    # Re-encode at this quality; null sends the clip's stored JPEG bytes when no resize is needed.
    jpeg_quality: int | None = Field(default=85, ge=1, le=100)

    @model_validator(mode="after")
    def _both_or_neither(self) -> FrameSettings:
        if (self.width is None) != (self.height is None):
            raise ValueError("set both frame.width and frame.height, or neither")
        return self


class ClipSettings(_Section):
    data_dir: Path = Path("data")
    # Clip ids under data/clips (make data / make data-mini). Cameras are assigned clips from
    # this list: round_robin (camera i gets ids[i % n]) or random (seeded).
    ids: list[str] = Field(default_factory=lambda: ["mot17-09-mini"], min_length=1)
    assignment: Literal["round_robin", "random"] = "round_robin"
    # Cap frames held in memory per clip (null: whole clip).
    max_frames: int | None = Field(default=None, ge=1)


class HttpSettings(_Section):
    request_timeout_s: float = Field(default=5.0, gt=0)
    connect_timeout_s: float = Field(default=2.0, gt=0)
    # Open-loop pacing: a frame due while this many sends of its camera are still in flight is
    # dropped (counted), like a camera buffer overflowing. Above 1, a camera's frames may
    # arrive out of order.
    max_in_flight_per_camera: int = Field(default=4, ge=1)
    # Retries on timeouts, connection errors and 5xx other than 503. Never on 429/503: a shed
    # frame stays shed. Exponential backoff with full jitter, capped.
    max_retries: int = Field(default=0, ge=0, le=10)
    retry_base_s: float = Field(default=0.05, gt=0)
    retry_max_s: float = Field(default=1.0, gt=0)


class StubSettings(_Section):
    """The stub ingest server (`python -m vms.loadgen stub`) used to test loadgen alone."""

    host: str = "127.0.0.1"
    port: int = Field(default=8080, ge=1, le=65535)
    latency_ms: float = Field(default=0.0, ge=0)
    reject_ratio: float = Field(default=0.0, ge=0, le=1)
    # Processes sharing the port (SO_REUSEPORT); one parses ~600 multipart frames/s (E-010).
    workers: int = Field(default=1, ge=1)


class LoadgenSettings(ServiceSettings):
    target_url: str = "http://127.0.0.1:8080"
    # paced: each camera sends at `fps` on a fixed schedule (open loop), the normal mode.
    # unpaced: each camera sends as fast as its in-flight limit allows; measures max rate.
    mode: Literal["paced", "unpaced"] = "paced"
    cameras: int = Field(default=20, ge=1)
    fps: float = Field(default=5.0, gt=0, le=120)
    duration_s: float = Field(default=60.0, gt=0)
    # Samples whose due time falls in the first warmup_s are excluded from the summary.
    warmup_s: float = Field(default=5.0, ge=0)
    camera_prefix: str = "cam-"
    # A run whose schedule lag p99 exceeds this is flagged `generator_ok: false`: the generator
    # fell behind, so the load it actually offered was lower than configured.
    max_schedule_lag_p99_ms: float = Field(default=10.0, gt=0)
    frame: FrameSettings = Field(default_factory=FrameSettings)
    clips: ClipSettings = Field(default_factory=ClipSettings)
    http: HttpSettings = Field(default_factory=HttpSettings)
    stub: StubSettings = Field(default_factory=StubSettings)
    # Scheduled, seeded drift on some cameras (task 9.2, vms.loadgen.drift); none by default.
    drift: DriftInjectionSettings = Field(default_factory=DriftInjectionSettings)
    # Prometheus /metrics while running; null disables.
    metrics_port: int | None = Field(default=9105, ge=1, le=65535)
    results_dir: Path = Path("bench/results")

    @model_validator(mode="after")
    def _check(self) -> LoadgenSettings:
        if self.warmup_s >= self.duration_s:
            raise ValueError("warmup_s must be shorter than duration_s")
        return self
