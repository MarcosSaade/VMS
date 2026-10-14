"""Gold checks with a fake executor: the SQL goes through the validator, the plan through
semantic validation and the compiler, and the two results must agree."""

from __future__ import annotations

import datetime as dt
from typing import Any

from vms.eval.compare import ResultSet, Tolerance
from vms.eval.dataset import EvalItem
from vms.eval.gold import GoldChecker, explicit_limit
from vms.query.catalog import QueryCatalog
from vms.query.settings import QuerySettings

NOW = dt.datetime(2026, 3, 31, 14, 30, tzinfo=dt.UTC)


class FakeDb:
    """Answers the gold SQL (recognized by its text) and anything else (the compiled plan)."""

    def __init__(self, gold: ResultSet, plan: ResultSet) -> None:
        self.gold, self.plan = gold, plan
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, sql: str, params: dict[str, Any]) -> ResultSet:
        self.calls.append((sql, params))
        return self.gold if "gold_marker" in sql else self.plan


GOLD_SQL = "SELECT count(*) AS gold_marker FROM tracks t WHERE t.class_name = 'person'"
PLAN = {"intent": "count", "target": "tracks", "filters": {"cameras": ["lobby"]}}


def item(**fields: Any) -> EvalItem:
    data: dict[str, Any] = {
        "id": "sq-001",
        "category": "count",
        "question": "How many objects did the lobby camera track?",
        "plan": PLAN,
        "expressible_by_plan": True,
        "sql": GOLD_SQL,
    }
    data.update(fields)
    return EvalItem.model_validate(data)


def checker(settings: QuerySettings, catalog: QueryCatalog, db: FakeDb) -> GoldChecker:
    return GoldChecker(settings, catalog, NOW, db, Tolerance())


def n(value: int) -> ResultSet:
    return ResultSet(columns=["n"], rows=[[value]])


def test_agreeing_golds_give_the_sql_result(settings: QuerySettings, catalog: QueryCatalog) -> None:
    db = FakeDb(gold=n(42), plan=n(42))
    out = checker(settings, catalog, db).check(item())
    assert out.problems == []
    assert out.expected == n(42)
    assert out.plan_rows == 1
    gold_sql, gold_params = db.calls[0]
    assert gold_params == {"now": NOW}
    assert "LIMIT 100" in gold_sql  # validated SQL, with the injected limit
    _, plan_params = db.calls[1]
    assert "cam-005" in plan_params.values()  # compiled with the eval catalog


def test_disagreement_is_reported(settings: QuerySettings, catalog: QueryCatalog) -> None:
    out = checker(settings, catalog, FakeDb(gold=n(42), plan=n(41))).check(item())
    assert out.problems == [
        "gold plan and gold SQL disagree: no column matches expected column 'n'"
    ]


def test_gold_sql_must_pass_the_validator(settings: QuerySettings, catalog: QueryCatalog) -> None:
    out = checker(settings, catalog, FakeDb(n(1), n(1))).check(
        item(sql="SELECT pg_sleep(1) AS gold_marker FROM tracks t")
    )
    assert len(out.problems) == 1
    assert out.problems[0].startswith("gold SQL rejected by the validator: forbidden_function")
    assert out.expected is None


def test_gold_plan_must_pass_semantic_validation(
    settings: QuerySettings, catalog: QueryCatalog
) -> None:
    plan = {"intent": "count", "target": "tracks", "filters": {"cameras": ["lobby_2"]}}
    out = checker(settings, catalog, FakeDb(n(1), n(1))).check(item(plan=plan))
    assert len(out.problems) == 1
    assert out.problems[0].startswith("gold plan fails semantic validation: unknown camera(s)")


def test_results_at_the_injected_limit_may_be_truncated(
    settings: QuerySettings, catalog: QueryCatalog
) -> None:
    many = ResultSet(columns=["track_id"], rows=[[str(i)] for i in range(100)])
    sql = "SELECT t.track_id AS gold_marker FROM tracks t"
    plan = {"intent": "list", "target": "tracks"}
    out = checker(settings, catalog, FakeDb(many, many)).check(
        item(category="list", sql=sql, plan=plan)
    )
    assert out.problems == [
        "gold SQL returned 100 rows, the validator's limit (100): the result may be "
        "truncated; narrow the question"
    ]
    # An explicit LIMIT of the same size is the question's own k.
    out = checker(settings, catalog, FakeDb(many, many)).check(
        item(category="list", sql=f"{sql} LIMIT 100", plan={**plan, "limit": 100})
    )
    assert out.problems == []


def test_unsupported_items_run_sql_only(settings: QuerySettings, catalog: QueryCatalog) -> None:
    db = FakeDb(n(3), n(0))
    out = checker(settings, catalog, db).check(
        item(
            category="unsupported",
            plan={"intent": "abstain", "abstain_reason": "unsupported"},
            expressible_by_plan=False,
        )
    )
    assert out.problems == []
    assert out.expected == n(3)
    assert len(db.calls) == 1


def test_explicit_limit() -> None:
    assert explicit_limit("SELECT 1 FROM t LIMIT 5") == 5
    assert explicit_limit("SELECT 1 FROM t") is None
    assert explicit_limit("SELECT 1 FROM (SELECT 1 FROM t LIMIT 3) s") is None
