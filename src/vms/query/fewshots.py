"""Few-shot bank: (question, gold `QueryPlan`, gold SQL) triples stored as versioned JSONL.

The plan is the target for the typed-plan planner; the SQL serves the free-form SQL engine
(task 7.2) and result checks once the schema exists. Questions the plan schema cannot express
carry an `abstain`/`unsupported` plan plus their real SQL, so both engines learn the boundary.

These examples are prompt material, never eval data: task 3.1 must keep the test split free of
exact and near duplicates of every question here.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vms.query.plan import AbstainReason, Intent, QueryPlan

# Abstentions that have no SQL answer at all.
_NO_SQL_REASONS = frozenset(
    {AbstainReason.AMBIGUOUS, AbstainReason.UNANSWERABLE, AbstainReason.DISALLOWED}
)


class Category(StrEnum):
    COUNT = "count"
    AGGREGATE = "aggregate"
    LIST = "list"
    TREND = "trend"
    TOP_K = "top_k"
    DWELL = "dwell"
    FRAMES = "frames"
    HYBRID = "hybrid"
    AMBIGUOUS = "ambiguous"
    UNANSWERABLE = "unanswerable"
    UNSUPPORTED = "unsupported"
    ADVERSARIAL = "adversarial"


_CATEGORY_REASON = {
    Category.AMBIGUOUS: AbstainReason.AMBIGUOUS,
    Category.UNANSWERABLE: AbstainReason.UNANSWERABLE,
    Category.UNSUPPORTED: AbstainReason.UNSUPPORTED,
    Category.ADVERSARIAL: AbstainReason.DISALLOWED,
}


class FewShotExample(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^fs-\d{3}$")
    category: Category
    question: str = Field(min_length=3)
    plan: QueryPlan
    expressible_by_plan: bool
    sql: str | None = None
    notes: str | None = None

    @model_validator(mode="after")
    def _check(self) -> FewShotExample:
        plan, reason = self.plan, self.plan.abstain_reason
        expected_reason = _CATEGORY_REASON.get(self.category)
        if expected_reason is not None and reason is not expected_reason:
            raise ValueError(f"category {self.category} needs abstain_reason {expected_reason}")
        if expected_reason is None and plan.intent is Intent.ABSTAIN:
            raise ValueError(f"category {self.category} must not abstain")
        if self.category is Category.HYBRID and plan.visual_query is None:
            raise ValueError("category hybrid needs plan.visual_query")
        if self.expressible_by_plan == (reason is AbstainReason.UNSUPPORTED):
            raise ValueError("expressible_by_plan must be false exactly for unsupported questions")
        if (self.sql is None) != (reason in _NO_SQL_REASONS):
            raise ValueError("sql is required unless the plan abstains with no SQL answer")
        return self


def normalize_question(question: str) -> str:
    """Key for duplicate checks (here and against eval splits in task 3.1)."""
    return " ".join(question.lower().replace("?", " ").split())


def load_fewshots(path: Path) -> list[FewShotExample]:
    examples: list[FewShotExample] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                examples.append(FewShotExample.model_validate(json.loads(line)))
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    ids = [e.id for e in examples]
    if len(set(ids)) != len(ids):
        raise ValueError(f"{path}: duplicate ids")
    questions = [normalize_question(e.question) for e in examples]
    if len(set(questions)) != len(questions):
        raise ValueError(f"{path}: duplicate questions")
    return examples
