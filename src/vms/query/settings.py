"""Settings for the query service, loaded from `configs/query.yaml` (see vms.common.settings)."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vms.common.settings import ServiceSettings


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SqlValidatorSettings(_Section):
    """Policy for the sqlglot validator that guards every SQL statement before execution."""

    dialect: Literal["postgres", "clickhouse"] = "postgres"
    default_limit: int = Field(default=100, ge=1)
    max_limit: int = Field(default=1000, ge=1)
    max_joins: int = Field(default=4, ge=0)
    max_sql_chars: int = Field(default=20_000, ge=1)
    # Queryable tables -> column -> SQL type. The allowlist; anything absent is rejected.
    tables: dict[str, dict[str, str]] = Field(default_factory=dict)
    denied_functions: list[str] = Field(default_factory=list)
    denied_function_prefixes: list[str] = Field(default_factory=list)

    @field_validator("denied_functions", "denied_function_prefixes")
    @classmethod
    def _lower(cls, values: list[str]) -> list[str]:
        return [v.lower() for v in values]

    @model_validator(mode="after")
    def _check(self) -> SqlValidatorSettings:
        if self.default_limit > self.max_limit:
            raise ValueError("default_limit must not exceed max_limit")
        return self


class QueryAssets(_Section):
    """Versioned data files the query service reads. Paths are relative to the repo root."""

    fewshots: Path = Path("eval/fewshot/structured.v1.jsonl")
    planner_prompt: Path = Path("configs/prompts/planner.v1.txt")
    sql_prompt: Path = Path("configs/prompts/sql.v1.txt")


class QuerySettings(ServiceSettings):
    sql: SqlValidatorSettings = Field(default_factory=SqlValidatorSettings)
    assets: QueryAssets = Field(default_factory=QueryAssets)
