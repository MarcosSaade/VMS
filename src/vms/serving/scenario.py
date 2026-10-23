"""`kind: llm` scenarios (task 8.1, D40): servers started one at a time, each with request-side
variants that are load tested and, optionally, recorded for the replayed dev eval.

A server is one engine process with one model and fixed server flags (prefix caching on or
off is a server flag, so "off" is its own server entry). A variant is what changes per request
on the same server: the response format (structured outputs on or off) and extra body fields.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Engine = Literal["vllm", "ollama", "external"]
NAME = r"^[a-z0-9][a-z0-9.-]*$"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoadSpec(_Model):
    """Closed-loop load: `concurrency` requests in flight, warmup excluded, repeated."""

    concurrency: list[int] = Field(min_length=1)
    requests_per_slot: int = Field(default=4, ge=1)  # requests per rep = max(min, slots * c)
    min_requests: int = Field(default=16, ge=1)
    warmup: int = Field(default=4, ge=0)  # requests at concurrency 1 before each level
    repetitions: int = Field(default=3, ge=1)
    timeout_s: float = Field(default=120.0, gt=0)

    def requests(self, concurrency: int) -> int:
        return max(self.min_requests, self.requests_per_slot * concurrency)


class RecordSpec(_Model):
    concurrency: int = Field(default=4, ge=1)  # as the eval's run.concurrency


class Variant(_Model):
    name: str = Field(pattern=NAME)
    response_format: Literal["json_schema", "json_object", "none"] = "json_schema"
    load: bool = True
    record: bool = False  # plan every question of the split with repair; replies kept


class Server(_Model):
    name: str = Field(pattern=NAME)
    engine: Engine
    model: str  # Hugging Face repo id (vLLM) or Ollama tag
    revision: str | None = None  # Hugging Face commit, pinned (vLLM)
    served_name: str | None = None  # the `model` field in requests; default: `name`
    args: list[str] = Field(default_factory=list)  # extra server flags
    env: dict[str, str] = Field(default_factory=dict)
    base_url: str | None = None  # engine external only
    use_llm_api_key: bool = False  # external only: send LLM_API_KEY (env or .env) as bearer
    request_extra: dict[str, Any] = Field(default_factory=dict)  # merged into every body
    start_timeout_s: float = Field(default=1200.0, gt=0)  # download, load, CUDA graphs
    variants: list[Variant] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> Server:
        if self.engine == "vllm" and not self.revision:
            raise ValueError(f"{self.name}: a vLLM model must be pinned by revision")
        if self.engine == "external" and not self.base_url:
            raise ValueError(f"{self.name}: an external server needs base_url")
        names = [v.name for v in self.variants]
        if len(set(names)) != len(names):
            raise ValueError(f"{self.name}: variant names must be unique")
        return self

    @property
    def request_model(self) -> str:
        if self.engine == "ollama":
            return self.model
        return self.served_name or self.name


class LlmScenario(_Model):
    name: str
    description: str = ""
    kind: Literal["llm"] = "llm"
    seed: int
    split: Literal["dev"] = "dev"  # tuning split only (AGENTS 8); test is never served here
    catalog: Path
    max_items: int | None = Field(default=None, ge=1)  # smoke runs only: not reportable
    max_tokens: int = Field(default=1024, ge=16)
    temperature: float = Field(default=0.0, ge=0.0)
    results_dir: Path = Path("bench/results")
    logs_dir: Path = Path("data/serving/logs")
    load: LoadSpec
    record: RecordSpec = Field(default_factory=RecordSpec)
    servers: list[Server] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> LlmScenario:
        names = [s.name for s in self.servers]
        if len(set(names)) != len(names):
            raise ValueError("server names must be unique")
        return self
