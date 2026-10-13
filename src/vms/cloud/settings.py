"""Settings for GPU sessions on AWS, loaded from `configs/cloud.yaml`."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from vms.common.settings import ServiceSettings

SERVICE = "cloud"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Prices(_Section):
    hourly_usd: dict[str, float] = Field(default_factory=dict)
    gp3_gb_month_usd: float = Field(default=0.08, ge=0)
    hours_per_month: float = Field(default=730.0, gt=0)
    public_ipv4_hour_usd: float = Field(default=0.005, ge=0)


class RemoteSettings(_Section):
    dir: str = "vms"
    clips: list[str] = Field(default_factory=list)


class Timeouts(_Section):
    aws_connect_s: float = Field(default=10.0, gt=0)
    aws_read_s: float = Field(default=60.0, gt=0)
    aws_max_attempts: int = Field(default=5, ge=1)
    running_s: float = Field(default=600.0, gt=0)
    ssh_ready_s: float = Field(default=600.0, gt=0)
    ssh_connect_s: int = Field(default=10, gt=0)
    sync_s: float = Field(default=1200.0, gt=0)
    setup_s: float = Field(default=3600.0, gt=0)
    scenario_s: float = Field(default=3600.0, gt=0)
    collect_s: float = Field(default=600.0, gt=0)
    terminate_s: float = Field(default=600.0, gt=0)
    delete_retry_s: float = Field(default=300.0, gt=0)
    poll_s: float = Field(default=5.0, gt=0)


class BudgetSettings(_Section):
    name: str = "vms-monthly"
    amount_usd: float = Field(default=20.0, gt=0)
    actual_thresholds_pct: list[float] = Field(default_factory=lambda: [50.0, 80.0, 100.0])
    forecast_thresholds_pct: list[float] = Field(default_factory=lambda: [100.0])


class CloudSettings(ServiceSettings):
    region: str | None = None
    instance_type: str = "g6.xlarge"
    vcpu_quota_code: str = "L-DB2E81BA"
    ami_parameter: str
    root_volume_gb: int = Field(default=100, ge=8)
    ssh_user: str = "ubuntu"
    ssh_cidr: str | None = None
    max_session_minutes: int = Field(default=150, ge=10)
    tags: dict[str, str] = Field(default_factory=lambda: {"project": "vms"})
    prices: Prices = Field(default_factory=Prices)
    remote: RemoteSettings = Field(default_factory=RemoteSettings)
    timeouts: Timeouts = Field(default_factory=Timeouts)
    budget: BudgetSettings = Field(default_factory=BudgetSettings)

    def hourly_usd(self) -> float:
        """Expected cost of one session hour: instance, root volume and public IPv4."""
        try:
            instance = self.prices.hourly_usd[self.instance_type]
        except KeyError:
            raise ValueError(
                f"no price for {self.instance_type!r} in configs/cloud.yaml prices.hourly_usd"
            ) from None
        volume = self.root_volume_gb * self.prices.gp3_gb_month_usd / self.prices.hours_per_month
        return instance + volume + self.prices.public_ipv4_hour_usd
