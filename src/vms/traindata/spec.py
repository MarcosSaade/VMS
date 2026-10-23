"""The training-data spec (`configs/traindata.v1.yaml`, task 8.2)."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class State(_Model):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    spec: Path
    db: str = Field(pattern=r"^vms_train_[a-z0-9_]+$")


class Generator(_Model):
    prompt: Path
    questions_per_call: int = Field(ge=1, le=50)
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(ge=256)
    response_format: str = "json_object"
    timeout_s: float = Field(gt=0)
    focuses: list[str] = Field(min_length=1)
    styles: list[str] = Field(min_length=1)


class Budget(_Model):
    max_cost_usd: float = Field(gt=0)
    max_candidates: int = Field(ge=1)
    call_concurrency: int = Field(ge=1)
    model_price: str


class Leakage(_Model):
    min_ratio: float = Field(gt=0, le=1)
    min_jaccard: float = Field(gt=0, le=1)
    min_cosine: float = Field(gt=0, le=1)


class TrainDataSpec(_Model):
    version: int
    seed: int
    states: list[State] = Field(min_length=1)
    states_dir: Path
    states_manifest: Path
    catalogs: Path  # in git: each state's catalog and clock, for prompts without a database
    generator: Generator
    planner_concurrency: int = Field(ge=1)
    budget: Budget
    candidates: Path
    dataset_dir: Path
    results_dir: Path
    statement_timeout_ms: int = Field(ge=100)
    leakage: Leakage


def load_spec(path: Path) -> TrainDataSpec:
    return TrainDataSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
