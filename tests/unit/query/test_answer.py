"""Templated answers: derived from the rows and the plan only."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from vms.query.answer import render_answer
from vms.query.plan import QueryPlan, ResultKind, result_kind


def answer(
    plan: dict[str, Any], columns: list[str], rows: list[tuple[Any, ...]], truncated: bool = False
) -> str:
    p = QueryPlan.model_validate(plan)
    return render_answer(p, result_kind(p), columns, rows, truncated=truncated)


@pytest.mark.parametrize(
    ("classes", "n", "expected"),
    [
        (["person"], 1, "1 person."),
        (["person"], 12, "12 people."),
        (["car"], 3, "3 cars."),
        ([], 1234, "1,234 tracked objects."),
    ],
)
def test_counts_name_what_was_counted(classes: list[str], n: int, expected: str) -> None:
    plan = {"intent": "count", "target": "tracks", "filters": {"classes": classes}}
    assert answer(plan, ["n"], [(n,)]) == expected


def test_aggregates_and_empty_results() -> None:
    dwell = {
        "intent": "dwell",
        "target": "tracks",
        "measure": {"op": "avg", "field": "dwell_seconds"},
    }
    assert answer(dwell, ["value"], [(42.256,)]) == "avg dwell_seconds: 42.26 s."
    assert answer(dwell, ["value"], [(None,)]) == "No matching data."
    assert answer({"intent": "list", "target": "events"}, ["event_id"], []) == "No matching data."


def test_time_series_singular_and_plural() -> None:
    plan = {"intent": "trend", "target": "tracks", "bucket": "hour"}
    assert answer(plan, ["bucket", "n"], [("t1", 5)]) == "1 hour bucket, 5 in total."
    assert answer(plan, ["bucket", "n"], [("t1", 5), ("t2", 7)]) == "2 hour buckets, 12 in total."


def test_groups_name_the_first() -> None:
    plan = {"intent": "count", "target": "tracks", "group_by": ["camera"]}
    assert answer(plan, ["camera", "n"], [("entrance", 9), ("checkout", 2)]) == (
        "2 groups; first: entrance (9)."
    )


def test_truncation_is_stated() -> None:
    plan = {"intent": "frames", "target": "frames"}
    p = QueryPlan.model_validate(plan)
    assert result_kind(p) is ResultKind.FRAMES
    assert answer(plan, ["frame_id"], [("a",), ("b",)], truncated=True) == (
        "2 frames (showing the first 2)."
    )


def test_numeric_values_are_rounded() -> None:
    dwell = {
        "intent": "dwell",
        "target": "tracks",
        "measure": {"op": "avg", "field": "dwell_seconds"},
        "group_by": ["zone"],
    }
    assert answer(dwell, ["zone", "value"], [("register_2", Decimal("11.6000000000000000"))]) == (
        "1 group; first: register_2 (11.60)."
    )
    scalar = {
        "intent": "dwell",
        "target": "tracks",
        "measure": {"op": "max", "field": "dwell_seconds"},
    }
    assert answer(scalar, ["value"], [(Decimal("3.5"),)]) == "max dwell_seconds: 3.50 s."
