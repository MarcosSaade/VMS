"""Execute an eval item's golds on the snapshot and check they agree (task 3.1).

For every item with an answer, the gold SQL runs through the SQL validator and then as the
read-only login, exactly as model-written SQL would (so a free-form engine could in principle
produce it). For every expressible item, the gold plan goes through the query service's own
path: semantic validation against the snapshot's catalog, the Postgres compiler with the eval
clock, the validator, the read-only login. The two results must match under the scorer's
comparison (`vms.eval.compare`); the SQL result becomes the item's `expected`.

Abstaining items are checked too: an unsupported item's gold plan abstains but its SQL must
run; ambiguous, unanswerable and adversarial items have nothing to run.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import psycopg
import sqlglot
from sqlglot import exp

from vms.eval.compare import ResultSet, Tolerance, canonical, compare, result_set
from vms.eval.dataset import EvalItem
from vms.query.catalog import CAMERAS_SQL, ZONES_SQL, QueryCatalog, catalog_from_rows
from vms.query.compiler import CompileError, PostgresCompiler
from vms.query.plan import Intent
from vms.query.semantic import validate_plan
from vms.query.settings import QuerySettings
from vms.query.sql_validator import SqlValidationError, SqlValidator

# Rows beyond the validator's cap would be cut off silently; gold results must be complete.
Execute = Callable[[str, dict[str, Any]], ResultSet]


@dataclass
class GoldOutcome:
    item_id: str
    expected: ResultSet | None = None
    plan_rows: int | None = None
    problems: list[str] = field(default_factory=list)


def explicit_limit(sql: str) -> int | None:
    """The outer LIMIT written in `sql`, if any (the validator injects one otherwise)."""
    limit = sqlglot.parse_one(sql, read="postgres").args.get("limit")
    expression = limit.expression if limit is not None else None
    if isinstance(expression, exp.Literal) and expression.is_int:
        return int(expression.name)
    return None


def connection_executor(conn: psycopg.Connection[Any]) -> Execute:
    def execute(sql: str, params: dict[str, Any]) -> ResultSet:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            columns = [d.name for d in cur.description or []]
            return result_set(columns, cur.fetchall())

    return execute


def load_catalog_rows(conn: psycopg.Connection[Any], classes: list[str]) -> QueryCatalog:
    return catalog_from_rows(
        conn.execute(CAMERAS_SQL).fetchall(), conn.execute(ZONES_SQL).fetchall(), classes
    )


class GoldChecker:
    def __init__(
        self,
        settings: QuerySettings,
        catalog: QueryCatalog,
        now: dt.datetime,
        execute: Execute,
        tolerance: Tolerance,
    ) -> None:
        self._settings = settings
        self._catalog = catalog
        self._now = now
        self._execute = execute
        self._tol = tolerance
        self._validator = SqlValidator(settings.sql)
        self._compiler = PostgresCompiler(
            catalog, default_limit=settings.sql.default_limit, max_limit=settings.sql.max_limit
        )

    def check(self, item: EvalItem) -> GoldOutcome:
        out = GoldOutcome(item.id)
        if item.sql is not None:
            result = self._run_sql(item, out)
            out.expected = None if result is None else canonical(result, ordered=item.ordered)
        if item.plan.intent is not Intent.ABSTAIN:
            self._check_plan(item, out)
        return out

    def _run_sql(self, item: EvalItem, out: GoldOutcome) -> ResultSet | None:
        assert item.sql is not None
        try:
            validated = self._validator.validate(item.sql)
        except SqlValidationError as exc:
            out.problems.append(f"gold SQL rejected by the validator: {exc}")
            return None
        try:
            result = self._execute(validated.sql, {"now": self._now})
        except psycopg.Error as exc:
            out.problems.append(f"gold SQL failed: {exc}".strip())
            return None
        if len(result.rows) >= validated.limit and explicit_limit(item.sql) != validated.limit:
            out.problems.append(
                f"gold SQL returned {len(result.rows)} rows, the validator's limit "
                f"({validated.limit}): the result may be truncated; narrow the question"
            )
        return result

    def _check_plan(self, item: EvalItem, out: GoldOutcome) -> None:
        errors = validate_plan(
            item.plan,
            self._catalog,
            self._now,
            self._settings.planner,
            max_limit=self._settings.sql.max_limit,
        )
        if errors:
            out.problems.append(f"gold plan fails semantic validation: {'; '.join(errors)}")
            return
        try:
            compiled = self._compiler.compile(item.plan, self._now)
            sql = self._validator.validate(compiled.sql).sql
            result = self._execute(sql, compiled.params)
        except (CompileError, SqlValidationError, psycopg.Error) as exc:
            out.problems.append(f"gold plan failed: {exc}".strip())
            return
        out.plan_rows = len(result.rows)
        if out.expected is None:
            return
        match = compare(out.expected, result, ordered=item.ordered, tol=self._tol)
        if not match.ok:
            out.problems.append(f"gold plan and gold SQL disagree: {match.reason}")
