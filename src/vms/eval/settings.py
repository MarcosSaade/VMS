"""Settings for the evaluation tooling, loaded from `configs/eval.yaml`."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from vms.common.settings import ServiceSettings


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SnapshotPaths(_Section):
    spec: Path = Path("configs/eval/snapshot.v1.yaml")
    dir: Path = Path("eval/structured/snapshot.v1")
    manifest: Path = Path("eval/structured/snapshot.v1.json")


class StructuredPaths(_Section):
    items: Path = Path("eval/structured/items.v1.jsonl")
    dev: Path = Path("eval/structured/dev.v1.jsonl")
    test: Path = Path("eval/structured/test.v1.jsonl")
    manifest: Path = Path("eval/structured/dataset.v1.json")
    split_seed: int = 31
    test_fraction: float = Field(default=0.5, gt=0, lt=1)


class LeakageSettings(_Section):
    min_ratio: float = Field(default=0.85, gt=0, le=1)
    min_jaccard: float = Field(default=0.8, gt=0, le=1)


class CompareSettings(_Section):
    rel_tol: float = Field(default=1e-6, ge=0)
    abs_tol: float = Field(default=1e-6, ge=0)


class RunSettings(_Section):
    """How an eval run asks and scores (task 3.2)."""

    concurrency: int = Field(default=4, ge=1, le=8)  # at most the reader pool's size
    resamples: int = Field(default=2000, ge=1000)
    confidence: float = Field(default=0.95, gt=0, lt=1)
    seed: int = 2026  # bootstrap resampling
    results_dir: Path = Path("eval/results")


class Price(_Section):
    """USD per million tokens."""

    prompt_per_mtok: float = Field(ge=0)
    completion_per_mtok: float = Field(ge=0)


class EvalSettings(ServiceSettings):
    db: str = "vms_eval"
    statement_timeout_ms: int = Field(default=10_000, ge=1)
    snapshot: SnapshotPaths = Field(default_factory=SnapshotPaths)
    structured: StructuredPaths = Field(default_factory=StructuredPaths)
    leakage: LeakageSettings = Field(default_factory=LeakageSettings)
    compare: CompareSettings = Field(default_factory=CompareSettings)
    run: RunSettings = Field(default_factory=RunSettings)
    # LLM prices by model name (LLM_MODEL); runs with a model not listed report no cost.
    prices: dict[str, Price] = Field(default_factory=dict)
