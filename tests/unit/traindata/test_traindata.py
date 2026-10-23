"""Training data (task 8.2): the budget, batch assignment, reply parsing, the pair filter with
a fake database, stage accounting, masking and the leakage checks. No network, no Postgres."""

from __future__ import annotations

import asyncio
import datetime as dt
from pathlib import Path
from typing import Any

import pytest

from vms.common.settings import load_settings
from vms.eval.compare import ResultSet, Tolerance, result_set
from vms.eval.settings import Price
from vms.query.catalog import catalog_from_rows
from vms.query.llm import LlmReply
from vms.query.settings import QuerySettings
from vms.traindata.filter import Checked, PairChecker, dedupe, stage_counts
from vms.traindata.generate import (
    Budget,
    BudgetedLlm,
    BudgetExhaustedError,
    parse_pairs,
    plan_batches,
)
from vms.traindata.leakage import leakage_hits, mask, max_cosines
from vms.traindata.spec import load_spec
from vms.traindata.states import LoadedState

REPO_ROOT = Path(__file__).resolve().parents[3]
PRICE = Price(prompt_per_mtok=0.30, completion_per_mtok=1.20, prompt_cache_hit_per_mtok=0.006)
NOW = dt.datetime(2026, 5, 14, 3, 15, tzinfo=dt.UTC)
CATALOG = catalog_from_rows(
    [("cam-w01", "main_gate", "Asia/Tokyo"), ("cam-w02", "aisle_7", "Asia/Tokyo")],
    [("cam-w02:picking_area", "cam-w02", "picking_area")],
    ["person", "car", "cell phone"],
)
STATE = LoadedState("warehouse", "vms_train_warehouse", NOW, CATALOG)
COUNT_PLAN = {
    "intent": "count",
    "target": "tracks",
    "filters": {"cameras": ["main_gate"], "classes": ["car"]},
}


def test_price_counts_cache_hits_at_the_hit_rate() -> None:
    assert PRICE.cost_usd(5000, 100, cached=4000) == pytest.approx(
        (1000 * 0.30 + 4000 * 0.006 + 100 * 1.20) / 1e6
    )
    assert Price(prompt_per_mtok=1.0, completion_per_mtok=0.0).cost_usd(10, 0, cached=10) == (
        pytest.approx(10 / 1e6)  # no hit price: hits cost as misses
    )


class Fixed:
    name = "fixed"

    def __init__(self, reply: LlmReply | Exception) -> None:
        self.reply = reply

    async def complete(self, messages: Any, schema: Any) -> LlmReply:
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply

    async def aclose(self) -> None:
        return None


def test_budget_reserves_the_worst_case_and_settles_the_actual_cost() -> None:
    budget = Budget(PRICE, cap_usd=0.01)
    reply = LlmReply("{}", prompt_tokens=4700, completion_tokens=150, cached_prompt_tokens=4600)
    llm = BudgetedLlm(Fixed(reply), budget, max_tokens=1024)
    asyncio.run(llm.complete([], {}))
    assert budget.reserved_usd == pytest.approx(0.0)
    assert budget.spent_usd == pytest.approx(PRICE.cost_usd(4700, 150, 4600))
    assert budget.calls == 1


def test_budget_refuses_a_call_that_could_pass_the_cap() -> None:
    budget = Budget(PRICE, cap_usd=0.002)  # less than one worst case
    with pytest.raises(BudgetExhaustedError):
        asyncio.run(BudgetedLlm(Fixed(LlmReply("{}")), budget, 1024).complete([], {}))
    assert budget.spent_usd == 0.0


def test_a_failed_call_is_charged_its_worst_case() -> None:
    budget = Budget(PRICE, cap_usd=1.0)
    with pytest.raises(RuntimeError):
        asyncio.run(BudgetedLlm(Fixed(RuntimeError("boom")), budget, 1024).complete([], {}))
    assert budget.spent_usd == pytest.approx(budget.worst_case(1024))
    assert budget.reserved_usd == pytest.approx(0.0)


def test_batches_are_seeded_per_index_so_a_resumed_run_continues_the_same_sequence() -> None:
    spec = load_spec(REPO_ROOT / "configs/traindata.v1.yaml")
    whole = plan_batches(spec, 0, 8)
    assert plan_batches(spec, 4, 4) == whole[4:]
    assert [b.state for b in whole[:3]] == [s.name for s in spec.states]


def test_parse_pairs_keeps_well_formed_items() -> None:
    text = '{"pairs": [{"question": " a? ", "plan": {}}, {"question": ""}, "x", {"plan": {}}]}'
    pairs, error = parse_pairs(text)
    assert error is None
    assert pairs == [{"question": "a?", "plan": {}}]
    assert parse_pairs("[1]")[1] == "no pairs list"
    assert parse_pairs("{")[1] is not None


@pytest.fixture(scope="module")
def qs() -> QuerySettings:
    return load_settings(QuerySettings, "query", config_dir=REPO_ROOT / "configs")


def executor(results: dict[str, ResultSet]) -> Any:
    calls: list[str] = []

    def execute(sql: str, params: dict[str, Any]) -> ResultSet:
        calls.append(sql)
        return results.get("b" if len(calls) % 2 == 0 else "a", result_set(["n"], [[3]]))

    return execute


def row(plan_a: Any, plan_b: Any, status: str = "planned", **kw: Any) -> dict[str, Any]:
    return {
        "id": kw.get("id", "c00000-00"),
        "state": "warehouse",
        "question": kw.get("question", "how many cars at the main gate?"),
        "plan_a": plan_a,
        "plan_b": plan_b,
        "b_status": status,
    }


def check(
    qs: QuerySettings, r: dict[str, Any], results: dict[str, ResultSet] | None = None
) -> Checked:
    checker = PairChecker(qs, STATE, executor(results or {}), Tolerance(1e-6, 1e-6))
    return checker.check(r)


def test_agreeing_plans_with_a_result_are_kept(qs: QuerySettings) -> None:
    c = check(qs, row(COUNT_PLAN, COUNT_PLAN))
    assert c.drop is None
    assert c.sql is not None
    assert c.rows == 1


@pytest.mark.parametrize(
    ("r", "results", "drop"),
    [
        (row({"intent": "count"}, COUNT_PLAN), None, "a_invalid"),
        (row({**COUNT_PLAN, "filters": {"cameras": ["lobby"]}}, COUNT_PLAN), None, "a_invalid"),
        (row(COUNT_PLAN, None, status="failed_validation"), None, "b_failed"),
        (
            row(COUNT_PLAN, {"intent": "abstain", "abstain_reason": "unsupported"}),
            None,
            "abstain_mismatch",
        ),
        (row(COUNT_PLAN, COUNT_PLAN), {"b": result_set(["n"], [[4]])}, "results_differ"),
        (
            row(COUNT_PLAN, COUNT_PLAN),
            {"a": result_set(["n"], [[0]]), "b": result_set(["n"], [[0]])},
            "empty_result",
        ),
    ],
)
def test_each_filter_stage_drops_what_it_should(
    qs: QuerySettings, r: dict[str, Any], results: Any, drop: str
) -> None:
    assert check(qs, r, results).drop == drop


def test_abstentions_agreeing_on_the_reason_are_kept(qs: QuerySettings) -> None:
    abstain = {"intent": "abstain", "abstain_reason": "unanswerable"}
    assert check(qs, row(abstain, abstain)).drop is None
    other = {"intent": "abstain", "abstain_reason": "ambiguous"}
    assert check(qs, row(abstain, other)).drop == "abstain_mismatch"


def test_dedupe_and_stage_counts(qs: QuerySettings) -> None:
    a = check(qs, row(COUNT_PLAN, COUNT_PLAN, id="c1", question="How many cars?"))
    b = check(qs, row(COUNT_PLAN, COUNT_PLAN, id="c2", question="how many  cars"))
    bad = check(qs, row({"intent": "count"}, COUNT_PLAN, id="c3"))
    dedupe([a, b, bad])
    assert [a.drop, b.drop, bad.drop] == [None, "duplicate", "a_invalid"]
    counts = stage_counts([a, b, bad])
    assert counts["kept"] == 1
    first = counts["stages"][0]
    assert (first["stage"], first["dropped"], first["pass_rate"]) == ("a_invalid", 1, 0.6667)
    dup = next(s for s in counts["stages"] if s["stage"] == "duplicate")
    assert dup["pass_rate"] == 0.5  # of the two that reached it


def test_mask_replaces_names_in_both_spellings_numbers_and_months() -> None:
    masked = mask("How many cell phones at Aisle 7 picking area on March 3, after 10:30?", CATALOG)
    assert masked == "how many <class> at <camera> <zone> on <month> <n>, after <n>"


def test_a_reworded_eval_question_for_another_catalog_is_caught() -> None:
    eval_catalog = catalog_from_rows(
        [("cam-001", "entrance", "Europe/London")],
        [("cam-001:front_door", "cam-001", "front_door")],
        ["person", "car"],
    )
    ref = [
        (
            "sq-001",
            mask("How many people entered the front door at the entrance today?", eval_catalog),
        )
    ]
    train = [
        ("t1", mask("How many people entered the picking area at aisle 7 today?", CATALOG)),
        ("t2", mask("Show the hourly trend of cars at the main gate last week", CATALOG)),
    ]
    hits = leakage_hits(train, ref, min_ratio=0.85, min_jaccard=0.8, min_cosine=0.9)
    assert {h.train_id for h in hits} == {"t1"}  # "the <camera>" vs "<camera>": not a template
    same = [("t3", mask("How many people entered the picking area at the aisle 7 today?", CATALOG))]
    hits = leakage_hits(same, ref, min_ratio=0.85, min_jaccard=0.8, min_cosine=0.9)
    assert "template" in {h.check for h in hits}


def test_max_cosines_finds_the_closest_reference() -> None:
    sims = max_cosines(["count cars today", "zzz qqq"], ["list trucks", "count cars today please"])
    assert sims[0][1] == 1
    assert sims[0][0] > 0.7
    assert sims[1][0] < 0.2
