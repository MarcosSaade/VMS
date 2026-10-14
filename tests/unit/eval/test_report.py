"""The Markdown report renders every headline metric with its interval."""

from __future__ import annotations

from typing import Any

from vms.eval.report import render
from vms.eval.scorer import EngineAnswer, ItemScore, summarize


def result(cost: float | None) -> dict[str, Any]:
    scores = [
        ItemScore(
            id=f"sq-00{n}",
            category="count",
            split="dev",
            expressible=True,
            must_abstain=False,
            correct=n % 2 == 0,
            error_class=None if n % 2 == 0 else "wrong_filter",
            answer=EngineAnswer(status="ok", latency_ms=100.0 * n),
        )
        for n in range(1, 5)
    ]
    return {
        "run_id": "01ABCDEFGHJKMNPQRSTVWXYZ00",
        "metrics": {
            "run": {"split": "dev", "llm": "m", "cost_usd": cost},
            "summary": summarize(scores, resamples=1000, seed=1, confidence=0.95),
        },
    }


def test_render() -> None:
    text = render([result(0.0123), result(None)])
    assert "| accuracy (all) | 50.0% [" in text
    assert "(n=4)" in text
    assert "| cost (USD) | 0.0123 | n/a |" in text
    assert "| count | 50.0%" in text
    assert "| wrong_filter | 2 | 2 |" in text
    assert render([]) == ""
