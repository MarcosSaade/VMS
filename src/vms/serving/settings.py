"""Settings for LLM serving benchmarks, loaded from `configs/serving.yaml`."""

from __future__ import annotations

from pathlib import Path

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
    env: dict[str, str] = Field(default_factory=dict)  # for every server of this engine

    @property
    def root_url(self) -> str:
        return f"http://{self.host}:{self.port}"


class LoraSettings(_Section):
    """LoRA training on a pod (task 8.3, D42)."""

    # The trainer runs in vLLM's virtualenv (its CUDA torch and transformers, plus peft).
    python: str = "/root/vllm-env/bin/python"
    script: Path = Path("scripts/lora_train.py")
    unpack_dir: Path = Path("data/serving/lora")  # adapter .tar files unpacked for vLLM
    work_dir: Path = Path("data/lora")  # rendered examples and run specs


class ServingSettings(ServiceSettings):
    vllm: EngineSettings
    ollama: EngineSettings
    lora: LoraSettings = Field(default_factory=LoraSettings)
    poll_s: float = Field(default=2.0, gt=0)
    stop_timeout_s: float = Field(default=60.0, gt=0)
    request_timeout_s: float = Field(default=120.0, gt=0)
    log_tail_lines: int = Field(default=40, ge=0)
