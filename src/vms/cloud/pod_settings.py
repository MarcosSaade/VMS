"""Settings for GPU sessions on Runpod pods (D37), loaded from `configs/runpod.yaml`."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from vms.cloud.settings import RemoteSettings
from vms.common.settings import ServiceSettings

SERVICE = "runpod"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RunpodCredentials(BaseSettings):
    """`RUNPOD_API_KEY` from the shell or `.env` (the same variable runpodctl reads)."""

    model_config = SettingsConfigDict(env_prefix="RUNPOD_", env_file=".env", extra="ignore")

    api_key: SecretStr | None = None

    @field_validator("api_key", mode="before")
    @classmethod
    def _empty_is_none(cls, value: object) -> object:
        return None if value == "" else value


class PodPrices(_Section):
    # Container disk while the pod runs; the GPU's hourly price comes from the live catalog.
    container_disk_gb_month_usd: float = Field(default=0.10, ge=0)
    hours_per_month: float = Field(default=730.0, gt=0)


class PodTimeouts(_Section):
    api_s: float = Field(default=30.0, gt=0)
    api_max_attempts: int = Field(default=5, ge=1)
    running_s: float = Field(default=900.0, gt=0)  # placement, image pull, sshd up
    ssh_ready_s: float = Field(default=300.0, gt=0)
    ssh_connect_s: int = Field(default=10, gt=0)
    sync_s: float = Field(default=1200.0, gt=0)
    setup_s: float = Field(default=3600.0, gt=0)
    scenario_s: float = Field(default=3600.0, gt=0)
    collect_s: float = Field(default=600.0, gt=0)
    terminate_s: float = Field(default=300.0, gt=0)
    poll_s: float = Field(default=5.0, gt=0)


class PodSettings(ServiceSettings):
    gpu_type: str = "NVIDIA L4"  # a catalog id (GET /v2/catalog/gpus), VMS_GPU_TYPE
    gpu_count: int = Field(default=1, ge=1)
    cloud: Literal["SECURE", "COMMUNITY"] = "SECURE"
    min_cuda_version: str = "12.8"  # the host driver's CUDA floor
    data_center_ids: list[str] = Field(default_factory=list)  # empty: any
    image: str  # pinned by digest
    container_disk_gb: int = Field(default=40, ge=10)
    ssh_user: str = "root"
    name_prefix: str = "vms"  # pods are named <prefix>-gpu-<session id>; the leftover check
    max_session_minutes: int = Field(default=150, ge=10)
    # The pod terminates itself after max_session_minutes only if it holds a pod-scoped API key
    # (RUNPOD_API_KEY inside the pod). Without one the session stops before running anything.
    require_self_terminate: bool = True
    # Shell commands run once over SSH before anything else (tools the image lacks).
    host_setup: list[str] = Field(default_factory=list)
    prices: PodPrices = Field(default_factory=PodPrices)
    remote: RemoteSettings = Field(default_factory=RemoteSettings)
    timeouts: PodTimeouts = Field(default_factory=PodTimeouts)

    def hourly_usd(self, gpu_hourly_usd: float) -> float:
        """Expected cost of one session hour: the GPUs at the catalog price plus container disk."""
        disk = (
            self.container_disk_gb
            * self.prices.container_disk_gb_month_usd
            / self.prices.hours_per_month
        )
        return gpu_hourly_usd * self.gpu_count + disk
