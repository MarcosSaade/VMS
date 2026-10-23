"""`kind: lora` scenarios (task 8.3, D42): train LoRA adapters on the planner's training pairs,
then serve them on the base with vLLM and record the dev split for the replayed eval.

The runs are an ablation against one named baseline run, one variable at a time (data size or
rank). They train in the listed order until the training time budget would be exceeded; a run
that would not fit is recorded as skipped, never cut short.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vms.serving.scenario import NAME, LoadSpec, RecordSpec


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Base(_Model):
    model: str  # Hugging Face repo id
    revision: str  # commit, pinned
    served_name: str = "base"


class Data(_Model):
    pairs: Path  # train/plans.v1/pairs.jsonl (DVC)
    catalogs: Path  # each training state's catalog and clock (configs/traindata)
    prompt: Path  # planner prompt the adapters are trained and served with
    val_fraction: float = Field(default=0.05, gt=0, lt=0.5)  # held out for eval loss


class Train(_Model):
    """Hyperparameters every run shares unless the run overrides rank."""

    epochs: int = Field(default=2, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    warmup_ratio: float = Field(default=0.03, ge=0, lt=1)
    weight_decay: float = Field(default=0.0, ge=0)
    micro_batch: int = Field(default=4, ge=1)
    grad_accum: int = Field(default=4, ge=1)
    max_len: int = Field(default=2048, ge=64)  # longer examples are dropped and counted
    alpha_per_rank: float = Field(default=2.0, gt=0)  # lora_alpha = alpha_per_rank * rank
    dropout: float = Field(default=0.05, ge=0, lt=1)
    # PEFT regex: attention and MLP projections of the language model only (not the vision
    # tower, not the linear-attention projections), the modules vLLM serves LoRA on.
    target_modules: str = (
        r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    )
    gradient_checkpointing: bool = False
    max_grad_norm: float = Field(default=1.0, gt=0)


class Run(_Model):
    name: str = Field(pattern=NAME)
    pairs: int | None = Field(default=None, ge=1)  # training pairs; None: all
    rank: int = Field(ge=1, le=256)


class Serve(_Model):
    """The vLLM server for the base with every trained adapter, and what is recorded."""

    args: list[str] = Field(default_factory=list)
    request_extra: dict[str, object] = Field(default_factory=dict)
    start_timeout_s: float = Field(default=1200.0, gt=0)
    catalog: Path  # the eval catalog file
    max_tokens: int = Field(default=1024, ge=16)
    record: RecordSpec = Field(default_factory=RecordSpec)
    # Controls recorded on the same server: the base with the service's prompt, and the base
    # with the adapters' prompt (so prompt and weights are separate variables).
    base_prompts: dict[str, Path] = Field(default_factory=dict)  # variant name -> prompt


class LoraScenario(_Model):
    name: str
    description: str = ""
    kind: Literal["lora"] = "lora"
    seed: int
    base: Base
    data: Data
    train: Train = Field(default_factory=Train)
    baseline: str  # the run the ablations change one variable of
    runs: list[Run] = Field(min_length=1)
    train_budget_minutes: float = Field(gt=0)
    adapters_dir: Path = Path("models/lora")  # one .tar per run (collected, DVC)
    results_dir: Path = Path("bench/results")
    logs_dir: Path = Path("data/serving/logs")
    serve: Serve

    @model_validator(mode="after")
    def _check(self) -> LoraScenario:
        names = [r.name for r in self.runs]
        if len(set(names)) != len(names):
            raise ValueError("run names must be unique")
        if self.baseline not in names:
            raise ValueError(f"baseline {self.baseline!r} is not a run")
        clash = set(self.serve.base_prompts) & set(names)
        if clash:
            raise ValueError(f"control variants clash with run names: {sorted(clash)}")
        return self


def load_spec_load() -> LoadSpec:
    """Recording only: the serving scenario's load section is required but unused here."""
    return LoadSpec(concurrency=[1])
