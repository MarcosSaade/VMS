"""Settings for cloud sessions on AWS (GPU or CPU), loaded from `configs/cloud.yaml`."""

from __future__ import annotations

from typing import Literal

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
    # Untracked files or directories pushed when present (repo-relative): DVC-tracked training
    # data and adapters the git sync leaves out (task 8.3).
    data: list[str] = Field(default_factory=list)
    # Directories copied back after the scenarios besides bench/results (repo-relative).
    collect: list[str] = Field(default_factory=list)


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


class Machine(_Section):
    """One kind of session host: what to launch and how to prepare it."""

    accelerator: Literal["gpu", "cpu"]
    instance_type: str
    vcpu_quota_code: str
    quota_name: str
    ami_parameter: str
    root_volume_gb: int = Field(default=100, ge=8)
    # Shell commands run once over SSH before anything else (e.g. installing Docker).
    host_setup: list[str] = Field(default_factory=list)


class CloudSettings(ServiceSettings):
    region: str | None = None
    machine: str = "gpu"  # a key of `machines` (VMS_MACHINE=cpu)
    instance_type: str | None = None  # overrides the machine's (VMS_INSTANCE_TYPE)
    machines: dict[str, Machine] = Field(default_factory=dict)
    ssh_user: str = "ubuntu"
    ssh_cidr: str | None = None
    max_session_minutes: int = Field(default=150, ge=10)
    tags: dict[str, str] = Field(default_factory=lambda: {"project": "vms"})
    prices: Prices = Field(default_factory=Prices)
    remote: RemoteSettings = Field(default_factory=RemoteSettings)
    timeouts: Timeouts = Field(default_factory=Timeouts)
    budget: BudgetSettings = Field(default_factory=BudgetSettings)

    @property
    def target(self) -> Machine:
        """The selected machine, with `instance_type` applied when set."""
        try:
            machine = self.machines[self.machine]
        except KeyError:
            raise ValueError(
                f"unknown machine {self.machine!r}; configs/cloud.yaml has {sorted(self.machines)}"
            ) from None
        if self.instance_type:
            machine = machine.model_copy(update={"instance_type": self.instance_type})
        return machine

    def hourly_usd(self) -> float:
        """Expected cost of one session hour: instance, root volume and public IPv4."""
        m = self.target
        try:
            instance = self.prices.hourly_usd[m.instance_type]
        except KeyError:
            raise ValueError(
                f"no price for {m.instance_type!r} in configs/cloud.yaml prices.hourly_usd"
            ) from None
        volume = m.root_volume_gb * self.prices.gp3_gb_month_usd / self.prices.hours_per_month
        return instance + volume + self.prices.public_ipv4_hour_usd
