"""Filter candidates into the training set (task 8.2).

A candidate is kept only if, in order (each stage's drop count is reported):
1. plan A parses as a QueryPlan and passes the planner's semantic validation for its state;
2. the planner (generation B) produced a valid plan for the question alone;
3. A and B agree on the kind of answer: both abstain for the same reason, or neither abstains;
4. for answering plans: both compile, both execute on the state's database as the read-only
   login, the two result sets are equal (the scorer's comparison, row order ignored), and the
   result is not empty (two plans that both select nothing agree without evidence);
5. its question is not a duplicate of an earlier kept one (normalized text);
6. it does not leak (`vms.traindata.leakage`) against any eval question or few-shot example.
The kept pair's plan is B's (what the service's prompt produces); its compiled SQL and
parameters are kept as well, as SQL targets for a D11 adapter.
"""

from __future__ import annotations

import datetime as dt
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import psycopg
from pydantic import ValidationError

from vms.eval.compare import ResultSet, Tolerance, canonical, compare, result_set
from vms.query.compiler import CompileError, PostgresCompiler
from vms.query.fewshots import normalize_question
from vms.query.plan import Intent, QueryPlan
from vms.query.semantic import validate_plan
from vms.query.settings import QuerySettings
from vms.query.sql_validator import SqlValidationError, SqlValidator
from vms.traindata.states import LoadedState

Execute = Callable[[str, dict[str, Any]], ResultSet]

STAGES = (
    "a_invalid",
    "b_failed",
    "abstain_mismatch",
    "compile_failed",
    "execute_failed",
    "results_differ",
    "empty_result",
    "duplicate",
    "leakage",
)


@dataclass
class Checked:
    row: dict[str, Any]
    drop: str | None = None
    plan: QueryPlan | None = None
    sql: str | None = None
    params: dict[str, Any] | None = None
    rows: int | None = None
    detail: str = ""


def _plan(raw: Any) -> QueryPlan | None:
    try:
        return QueryPlan.model_validate(raw)
    except ValidationError:
        return None


class PairChecker:
    def __init__(
        self,
        qs: QuerySettings,
        state: LoadedState,
        execute: Execute,
        tolerance: Tolerance,
    ) -> None:
        self._qs = qs
        self._state = state
        self._execute = execute
        self._tol = tolerance
        self._validator = SqlValidator(qs.sql)
        self._compiler = PostgresCompiler(
            state.catalog, default_limit=qs.sql.default_limit, max_limit=qs.sql.max_limit
        )

    def _valid(self, plan: QueryPlan) -> list[str]:
        return validate_plan(
            plan,
            self._state.catalog,
            self._state.now,
            self._qs.planner,
            max_limit=self._qs.sql.max_limit,
        )

    def _run(self, plan: QueryPlan) -> tuple[str, dict[str, Any], ResultSet]:
        compiled = self._compiler.compile(plan, self._state.now)
        sql = self._validator.validate(compiled.sql).sql
        return sql, compiled.params, self._execute(sql, compiled.params)

    def check(self, row: dict[str, Any]) -> Checked:
        out = Checked(row)
        a = _plan(row.get("plan_a"))
        if a is None or self._valid(a):
            out.drop = "a_invalid"
            return out
        b = _plan(row.get("plan_b")) if row.get("b_status") == "planned" else None
        if b is None:
            out.drop = "b_failed"
            return out
        out.plan = b
        abstain_a, abstain_b = a.intent is Intent.ABSTAIN, b.intent is Intent.ABSTAIN
        if abstain_a or abstain_b:
            if not (abstain_a and abstain_b and a.abstain_reason == b.abstain_reason):
                out.drop = "abstain_mismatch"
            return out
        try:
            self._compiler.compile(a, self._state.now)
            self._compiler.compile(b, self._state.now)
        except CompileError as exc:
            out.drop, out.detail = "compile_failed", str(exc)
            return out
        try:
            _, _, result_a = self._run(a)
            out.sql, out.params, result_b = self._run(b)
        except (CompileError, SqlValidationError, psycopg.Error) as exc:
            out.drop, out.detail = "execute_failed", str(exc).strip()[:200]
            return out
        match = compare(canonical(result_a, ordered=False), result_b, ordered=False, tol=self._tol)
        if not match.ok:
            out.drop, out.detail = "results_differ", match.reason
            return out
        out.rows = len(result_b.rows)
        if not result_b.rows or all(all(v in (0, None) for v in r) for r in result_b.rows):
            out.drop = "empty_result"
        return out


def connection_executor(conn: psycopg.Connection[Any]) -> Execute:
    def execute(sql: str, params: dict[str, Any]) -> ResultSet:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return result_set([d.name for d in cur.description or []], cur.fetchall())

    return execute


def dedupe(checked: Sequence[Checked]) -> None:
    seen: set[tuple[str, str]] = set()
    for c in checked:
        if c.drop is not None:
            continue
        key = (c.row["state"], normalize_question(c.row["question"]))
        if key in seen:
            c.drop = "duplicate"
        seen.add(key)


def pair_record(c: Checked, state: LoadedState) -> dict[str, Any]:
    assert c.plan is not None
    return {
        "id": c.row["id"],
        "state": c.row["state"],
        "now": state.now.isoformat(),
        "question": c.row["question"],
        "plan": c.plan.model_dump(mode="json", exclude_none=True),
        "abstain": c.plan.intent is Intent.ABSTAIN,
        "sql": c.sql,
        "params": {k: _json_value(v) for k, v in (c.params or {}).items()},
        "result_rows": c.rows,
        "plan_a_equal": c.row.get("plan_a") is not None and _plan(c.row["plan_a"]) == c.plan,
        "focus": c.row.get("focus"),
        "style": c.row.get("style"),
    }


def _json_value(value: Any) -> Any:
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    if isinstance(value, list | tuple):
        return [_json_value(v) for v in value]
    return value


def stage_counts(checked: Sequence[Checked]) -> dict[str, Any]:
    drops = Counter(c.drop for c in checked if c.drop is not None)
    kept = sum(c.drop is None for c in checked)
    remaining = len(checked)
    stages = []
    for stage in STAGES:
        dropped = drops.get(stage, 0)
        stages.append(
            {
                "stage": stage,
                "dropped": dropped,
                "pass_rate": round(1 - dropped / remaining, 4) if remaining else None,
            }
        )
        remaining -= dropped
    return {"candidates": len(checked), "kept": kept, "stages": stages}
