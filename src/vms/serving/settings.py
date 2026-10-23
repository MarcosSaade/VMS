"""Settings for LLM serving benchmarks, loaded from `configs/serving.yaml`."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from vms.common.settings import ServiceSettings

SERVICE = "serving"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EngineSettings(_Section):
    bin: str
    host: str = "127.0.0.1"
    port: int = Field(ge=1, le=65535)
    pull_timeout_s: float = Field(default=1800.0, gt=0)  # Ollama only

    @property
    def root_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class ServingSettings(ServiceSettings):
    vllm: EngineSettings
    ollama: EngineSettings
    poll_s: float = Field(default=2.0, gt=0)
    stop_timeout_s: float = Field(default=60.0, gt=0)
    request_timeout_s: float = Field(default=120.0, gt=0)
    log_tail_lines: int = Field(default=40, ge=0)
