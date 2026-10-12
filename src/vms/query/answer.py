"""Short natural-language answers from results, by template (no second LLM call).

The answer is derived from the rows and the plan only, so it is deterministic, costs no tokens
and cannot state anything the rows do not show. The rows themselves, the plan and the SQL are
returned next to it for anyone who wants the detail.
"""

from __future__ import annotations

from typing import Any

from vms.query.plan import Intent, QueryPlan, ResultKind, Target

_NOUNS = {
    Target.TRACKS: ("tracked object", "tracked objects"),
    Target.DETECTIONS: ("detection", "detections"),
    Target.EVENTS: ("event", "events"),
    Target.FRAMES: ("frame", "frames"),
}
_PEOPLE = ("person", "people")


def _noun(plan: QueryPlan, n: int) -> str:
    assert plan.target is not None
    singular, plural = _NOUNS[plan.target]
    if plan.target is Target.TRACKS and plan.filters.classes == ["person"]:
        singular, plural = _PEOPLE
    elif plan.target is Target.TRACKS and len(plan.filters.classes) == 1:
        cls = plan.filters.classes[0]
        singular, plural = cls, f"{cls}s"
    return singular if n == 1 else plural


def _number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def render_answer(
    plan: QueryPlan,
    kind: ResultKind,
    columns: list[str],
    rows: list[tuple[Any, ...]],
    *,
    truncated: bool,
) -> str:
    more = f" (showing the first {len(rows)})" if truncated else ""
    if kind is ResultKind.SCALAR:
        value = rows[0][0] if rows else None
        if plan.intent is Intent.COUNT:
            n = int(value or 0)
            return f"{n:,} {_noun(plan, n)}."
        if value is None:
            return "No matching data."
        assert plan.measure is not None
        unit = " s" if plan.intent is Intent.DWELL else ""
        return f"{plan.measure.op.value} {plan.measure.field.value}: {_number(value)}{unit}."
    if not rows:
        return "No matching data."
    if kind is ResultKind.TIME_SERIES:
        total = sum(int(r[columns.index("n")]) for r in rows)
        assert plan.bucket is not None
        return f"{len(rows)} {plan.bucket.value} buckets, {total:,} in total{more}."
    if kind is ResultKind.FRAMES:
        return f"{len(rows)} frame{'s' if len(rows) != 1 else ''}{more}."
    if plan.intent is Intent.LIST:
        return f"{len(rows)} {_noun(plan, len(rows))}{more}."
    # Grouped count, aggregate, dwell or top_k: name the first group.
    keys = [c for c in columns if c not in ("n", "value")]
    metric = "n" if "n" in columns else "value"
    first = rows[0]
    label = ", ".join(str(first[columns.index(k)]) for k in keys)
    value = _number(first[columns.index(metric)])
    return f"{len(rows)} groups; first: {label} ({value}){more}."
