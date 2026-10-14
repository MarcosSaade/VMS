"""The harness with a fake engine: concurrency limit, red-team questions one at a time between
database fingerprints, breaches, engine exceptions, cost, and the gold planner."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from vms.eval.compare import ResultSet, Tolerance
from vms.eval.dataset import EvalItem
from vms.eval.harness import breach_of, cost_usd, gold_llm, run_items, self_test_failures
from vms.eval.scorer import EngineAnswer
from vms.eval.settings import Price
from vms.query.engine import QueryResponse, QueryStatus
from vms.query.llm import ChatMessage

PLAN = {"intent": "count", "target": "tracks"}


def item(n: int, *, redteam: bool = False) -> EvalItem:
    if redteam:
        return EvalItem.model_validate(
            {
                "id": f"sq-{n:03d}",
                "category": "adversarial",
                "question": f"attack {n}",
                "plan": {"intent": "abstain", "abstain_reason": "disallowed"},
                "expressible_by_plan": True,
                "tags": ["redteam:dml"],
            }
        )
    return EvalItem.model_validate(
        {
            "id": f"sq-{n:03d}",
            "category": "count",
            "question": f"question {n}",
            "plan": PLAN,
            "expressible_by_plan": True,
            "sql": "SELECT 1",
            "expected": ResultSet(columns=["n"], rows=[[n]]),
        }
    )


class FakeEngine:
    """Answers question N with N rows-worth of count; tracks how many calls overlap."""

    name = "fake"

    def __init__(self, events: list[str] | None = None, fail_on: str | None = None) -> None:
        self.in_flight = 0
        self.max_in_flight = 0
        self.events = events if events is not None else []
        self.fail_on = fail_on

    async def answer(self, question: str) -> QueryResponse:
        self.in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self.in_flight)
        self.events.append(f"ask {question}")
        for _ in range(3):
            await asyncio.sleep(0)  # yield, so other questions can start
        self.in_flight -= 1
        if question == self.fail_on:
            raise RuntimeError("boom")
        base: dict[str, Any] = {"query_id": "q", "question": question, "llm": "fake"}
        if question.startswith("attack"):
            return QueryResponse(
                **base, status=QueryStatus.ABSTAINED, answer="no", abstain_reason="disallowed"
            )
        n = int(question.split()[1])
        return QueryResponse(
            **base, status=QueryStatus.OK, answer="", columns=["n"], rows=[[n]], plan=PLAN
        )


def run(engine: FakeEngine, items: list[EvalItem], **kw: Any) -> list[Any]:
    return asyncio.run(run_items(engine, items, tolerance=Tolerance(), **kw))


def test_scores_come_back_in_item_order_within_the_concurrency_limit() -> None:
    engine = FakeEngine()
    items = [item(n) for n in range(1, 11)]
    scores = run(engine, items, concurrency=3)
    assert [s.id for s in scores] == [i.id for i in items]
    assert all(s.correct for s in scores)
    assert engine.max_in_flight == 3


def test_redteam_questions_run_alone_between_fingerprints() -> None:
    events: list[str] = []
    engine = FakeEngine(events)
    states = iter(["a", "a", "a", "b"])  # the second attack changes the database

    async def fingerprint() -> str:
        events.append("fingerprint")
        return next(states)

    items = [item(1), item(2, redteam=True), item(3, redteam=True)]
    scores = run(engine, items, concurrency=4, fingerprint=fingerprint)
    assert events == [
        "ask question 1",
        "fingerprint", "ask attack 2", "fingerprint",
        "fingerprint", "ask attack 3", "fingerprint",
    ]  # fmt: skip
    assert [s.breach for s in scores] == [None, None, "database changed"]
    assert scores[2].correct  # refused, yet harm was observed: reported separately


def test_an_engine_exception_costs_one_question() -> None:
    scores = run(FakeEngine(fail_on="question 2"), [item(1), item(2)], concurrency=2)
    assert scores[0].correct
    assert scores[1].answer.error_type == "exception"
    assert scores[1].error_class == "execution_error"
    assert "RuntimeError: boom" in (scores[1].answer.error_detail or "")


def test_breach_rules() -> None:
    ok = EngineAnswer(status="ok")
    assert breach_of(ok, "a", "a") is None
    assert breach_of(ok, None, None) is None
    assert breach_of(ok, "a", "b") == "database changed"
    timeout = EngineAnswer(status="error", error_type="timeout")
    assert (breach_of(timeout, "a", "a") or "").startswith("statement timeout")


def test_cost_from_configured_prices() -> None:
    prices = {"m": Price(prompt_per_mtok=0.5, completion_per_mtok=2.0)}
    assert cost_usd(prices, "m", 1_000_000, 500_000) == pytest.approx(1.5)
    assert cost_usd(prices, "other", 10, 10) is None


def test_gold_llm_answers_each_items_gold_plan() -> None:
    llm = gold_llm([item(4)])
    reply = asyncio.run(llm.complete([ChatMessage("user", "Question 4?")], {}))
    assert '"intent": "count"' in reply.text


def test_self_test_failures_name_every_wrong_or_breached_item() -> None:
    events: list[str] = []
    states = iter(["a", "a", "a", "b"])

    async def fingerprint() -> str:
        return next(states)

    items = [item(1), item(2), item(3, redteam=True), item(4, redteam=True)]
    scores = run(FakeEngine(events, fail_on="question 2"), items, concurrency=2,
                 fingerprint=fingerprint)  # fmt: skip
    assert self_test_failures(scores) == [
        "sq-002: execution_error",
        "sq-004: database changed",
    ]
    assert self_test_failures(scores[:1]) == []
    assert self_test_failures([]) == ["no items scored"]
