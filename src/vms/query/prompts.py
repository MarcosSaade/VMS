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

    def render(self, values: Mapping[str, object]) -> str:
        missing, unknown = self.fields - values.keys(), values.keys() - self.fields
        if missing or unknown:
            raise ValueError(f"{self.name}: missing {sorted(missing)}, unknown {sorted(unknown)}")
        return self.template.substitute({k: str(v) for k, v in values.items()})
