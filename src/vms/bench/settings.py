"""Settings for the end-to-end benchmark runner, loaded from `configs/bench.yaml`.

Scenarios describe the load (`vms.bench.scenario`); these settings say where the system under
test is, how each step's window is read from Prometheus and how a step is judged against the
SLOs.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from vms.common.settings import ServiceSettings

SERVICE = "bench"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DrainSettings(_Section):
    enabled: bool = True
    timeout_s: float = Field(default=900.0, gt=0)
    interval_s: float = Field(default=2.0, gt=0)


class SloSettings(_Section):
    freshness_p95_s: float = Field(default=2.0, gt=0)
    max_lost_frames: int = Field(default=0, ge=0)
    max_backlog_slope_fps: float = Field(default=0.5, ge=0)
    query_p95_s: float = Field(default=1.5, gt=0)


class PromQuery(_Section):
    """A PromQL template; `$window` becomes the step's window (e.g. `30s`)."""

    query: str
    by: str | None = None  # split the result into one number per value of this label


class BenchSettings(ServiceSettings):
    ingest_url: str = "http://127.0.0.1:8080"
    query_url: str = "http://127.0.0.1:8070"
    prometheus_url: str = "http://127.0.0.1:9090"
    results_dir: Path = Path("bench/results")
    http_timeout_s: float = Field(default=10.0, gt=0)
    query_timeout_s: float = Field(default=30.0, gt=0)
    loadgen_timeout_margin_s: float = Field(default=120.0, gt=0)
    min_window_s: float = Field(default=15.0, gt=0)
    scrape_settle_s: float = Field(default=7.0, ge=0)
    drain: DrainSettings = Field(default_factory=DrainSettings)
    max_client_cpu_cores: float = Field(default=0.8, gt=0)
    slo: SloSettings = Field(default_factory=SloSettings)
    queries: dict[str, PromQuery] = Field(default_factory=dict)
