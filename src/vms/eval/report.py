"""Markdown tables of eval results files for BENCHMARKS.md (`python -m vms.eval report`)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def _pct(estimate: dict[str, Any] | None) -> str:
    if estimate is None:
        return "n/a"
    return (
        f"{100 * estimate['mean']:.1f}% [{100 * estimate['lo']:.1f}, {100 * estimate['hi']:.1f}]"
        f" (n={estimate['n']})"
    )


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:,.0f}"


def render(results: Sequence[dict[str, Any]]) -> str:
    """One column per results file: headline metrics, then categories, errors and red team."""
    if not results:
        return ""
    heads = [
        f"{r['metrics']['run']['split']} / {r['metrics']['run']['llm']} ({r['run_id'][-6:]})"
        for r in results
    ]
    s = [r["metrics"]["summary"] for r in results]
    lines = [
        "| Metric (95% bootstrap CI) | " + " | ".join(heads) + " |",
        "|---|" + "---|" * len(results),
    ]

    def row(name: str, cells: list[str]) -> None:
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    row("accuracy (all)", [_pct(x["accuracy"]) for x in s])
    row("accuracy (plan-expressible)", [_pct(x["accuracy_expressible"]) for x in s])
    row(
        "plan coverage",
        [f"{100 * x['plan_coverage']['value']:.1f}%" if x["plan_coverage"] else "n/a" for x in s],
    )
    row("abstention correct", [_pct(x["abstention_correct"]) for x in s])
    row("abstained (any reason)", [_pct(x["abstained_any"]) for x in s])
    row("false abstention", [_pct(x["false_abstention"]) for x in s])
    row("red team refused", [_pct(x["redteam"]["refusal"]) for x in s])
    row("red team attack success", [_pct(x["redteam"]["attack_success"]) for x in s])
    row("latency p50 / p95 ms", [f"{_ms(x['latency_ms']['p50'])} / {_ms(x['latency_ms']['p95'])}"
                                  for x in s])  # fmt: skip
    row("tokens per question", [f"{x['tokens']['per_question']:,.0f}" for x in s])
    row(
        "cost (USD)",
        [
            "n/a" if r["metrics"]["run"]["cost_usd"] is None
            else f"{r['metrics']['run']['cost_usd']:.4f}"
            for r in results
        ],
    )  # fmt: skip

    categories = sorted({c for x in s for c in x["by_category"]})
    lines += ["", "| Category | " + " | ".join(heads) + " |", "|---|" + "---|" * len(results)]
    for c in categories:
        row(c, [_pct(x["by_category"].get(c)) for x in s])

    classes = sorted({c for x in s for c in x["errors"]})
    lines += ["", "| Error class | " + " | ".join(heads) + " |", "|---|" + "---|" * len(results)]
    for c in classes:
        row(c, [str(x["errors"].get(c, 0)) for x in s])
    return "\n".join(lines) + "\n"
