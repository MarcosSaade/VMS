"""Versioned prompt templates (`configs/prompts/*.vN.txt`) with `$name` placeholders.

Templates are data, so a prompt change is a reviewed diff with a new version number, and eval
results can record which version produced them. Rendering is strict: every placeholder must be
filled and no unknown value may be passed, so a template and its caller cannot drift apart
silently. Context building (schema summary, retrieved examples) is task 1.7.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from string import Template

PLANNER_FIELDS = frozenset(
    {"schema_summary", "cameras", "classes", "now", "plan_schema", "examples"}
)
# A planner prompt may leave out the plan schema and the few-shot bank (the compact prompt a
# fine-tuned planner is trained and served with, D42), never the catalog or the clock.
PLANNER_REQUIRED = frozenset({"schema_summary", "cameras", "classes", "now"})
SQL_FIELDS = frozenset({"schema_ddl", "cameras", "classes", "max_limit", "examples"})


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    template: Template

    @classmethod
    def load(cls, path: Path, expected: frozenset[str] | None = None) -> PromptTemplate:
        prompt = cls(name=path.name, template=Template(path.read_text(encoding="utf-8")))
        if not prompt.template.is_valid():
            raise ValueError(f"{path}: invalid placeholder syntax (escape a literal $ as $$)")
        if expected is not None and prompt.fields != expected:
            raise ValueError(
                f"{path}: placeholders {sorted(prompt.fields)} != expected {sorted(expected)}"
            )
        return prompt

    @property
    def fields(self) -> frozenset[str]:
        return frozenset(self.template.get_identifiers())

    @classmethod
    def load_planner(cls, path: Path) -> PromptTemplate:
        """A planner prompt: PLANNER_REQUIRED <= placeholders <= PLANNER_FIELDS."""
        prompt = cls.load(path)
        if not PLANNER_REQUIRED <= prompt.fields <= PLANNER_FIELDS:
            raise ValueError(
                f"{path}: planner placeholders {sorted(prompt.fields)} must include "
                f"{sorted(PLANNER_REQUIRED)} and stay within {sorted(PLANNER_FIELDS)}"
            )
        return prompt

    def render(self, values: Mapping[str, object]) -> str:
        missing, unknown = self.fields - values.keys(), values.keys() - self.fields
        if missing or unknown:
            raise ValueError(f"{self.name}: missing {sorted(missing)}, unknown {sorted(unknown)}")
        return self.template.substitute({k: str(v) for k, v in values.items()})
