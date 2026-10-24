"""The D7 comparison (task 8.5): planner options on test accuracy, latency under load and cost
per 1,000 questions, from results files already written (`kind: llm-compare`).

Each option names three inputs: its one test-split eval (accuracy with its bootstrap interval,
and how many LLM calls a question took: one, plus one per repair), a load test of the same
model and prompt (`kind: llm` results: closed-loop levels of concurrency) and how it is paid
for. A GPU option pays the pod's hourly price whether busy or idle; a hosted option pays per
token at the provider's prices (configs/eval.yaml), with cached prompt tokens at the cache-hit
rate when the load test saw them.

Per option: the latency floor (p95 at concurrency 1), the highest load level whose p95 meets
the SLO (worst repetition), and the saturation level (most requests/s). Cost per 1,000
questions for a GPU is its hourly price over the questions it answers in an hour at that level,
so it is the cost at full utilization, the best case; the monthly cost of keeping one GPU up is
the fixed cost a deployment pays at any volume. The break-even volume is that monthly cost over
the hosted cost per question: below it, the hosted model is cheaper. Accuracy differences
between options are paired bootstrap intervals over the questions both answered (same split).

What this does not measure: the network between the query service and a self-hosted server
(load tests ran with client and server on the pod), a hosted provider's rate limits and price
changes, engineering and on-call time, and GPU prices other than the pod's.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vms.bench.results import RunResult, git_info, hardware, package_versions, write_result
from vms.common.ids import new_ulid
from vms.common.logs import get_logger
from vms.common.settings import load_settings
from vms.eval.settings import EvalSettings, Price
from vms.eval.stats import percentile

log = get_logger(__name__)

KIND = "bench.llm-compare"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LoadRef(_Model):
    results: Path  # a `bench.llm` results JSON
    server: str
    variant: str


class Cost(_Model):
    kind: Literal["gpu", "hosted"]
    session: Path | None = None  # gpu: the pod session's results JSON (hourly_usd)
    price: str | None = None  # hosted: a model in configs/eval.yaml prices

    @model_validator(mode="after")
    def _check(self) -> Cost:
        if self.kind == "gpu" and self.session is None:
            raise ValueError("a GPU option needs its session results (hourly price)")
        if self.kind == "hosted" and self.price is None:
            raise ValueError("a hosted option needs a price name")
        return self


class Option(_Model):
    name: str
    description: str = ""
    accuracy: Path  # the option's test-split eval results JSON
    load: LoadRef
    cost: Cost


class CompareScenario(_Model):
    name: str
    description: str = ""
    kind: Literal["llm-compare"] = "llm-compare"
    seed: int
    slo_p95_s: float = Field(gt=0)  # LLM call p95 a load level must meet (worst repetition)
    hours_per_month: float = Field(default=730.0, gt=0)
    resamples: int = Field(default=2000, ge=1)
    confidence: float = Field(default=0.95, gt=0, lt=1)
    results_dir: Path = Path("bench/results")
    options: list[Option] = Field(min_length=1)
    paired: list[tuple[str, str]] = Field(default_factory=list)  # (a, b): accuracy a - b

    @model_validator(mode="after")
    def _check(self) -> CompareScenario:
        names = [o.name for o in self.options]
        if len(set(names)) != len(names):
            raise ValueError("option names must be unique")
        unknown = {n for pair in self.paired for n in pair} - set(names)
        if unknown:
            raise ValueError(f"paired names unknown options {sorted(unknown)}")
        return self


def read_json(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def accuracy_of(result: Mapping[str, Any]) -> dict[str, Any]:
    """Accuracy with its interval, the split's hash, and LLM calls per question."""
    metrics = result["metrics"]
    items = metrics["items"]
    calls = sum(1 + bool(i.get("repaired")) for i in items) / len(items)
    return {
        "split": metrics["run"]["split"],
        "split_sha256": metrics["run"]["split_sha256"],
        "accuracy": metrics["summary"]["accuracy"],
        "calls_per_question": calls,
        "correct": {i["id"]: bool(i["correct"]) for i in items},
    }


def levels_of(result: Mapping[str, Any], ref: LoadRef) -> list[dict[str, Any]]:
    """Each load level of one variant: concurrency, mean requests/s, p50 and p95 (worst rep,
    seconds) and per-request tokens."""
    server = result["metrics"]["servers"][ref.server]
    if not server.get("ok"):
        raise ValueError(f"{ref.results}: server {ref.server} did not run: {server.get('error')}")
    load = server["variants"][ref.variant]["load"]
    out = []
    for conc, level in sorted(load.items(), key=lambda kv: int(kv[0])):
        reps = level["reps"]
        ok = sum(r["requests"] - r["errors"] for r in reps)
        totals = [r.get("tokens_total") or {} for r in reps]
        cached = [t.get("cached") for t in totals]
        out.append(
            {
                "concurrency": int(conc),
                "requests_per_s": level["spread"]["requests_per_s"]["mean"],
                "p50_s": max(r["latency_ms"]["p50"] for r in reps) / 1000.0,
                "p95_s": max(r["latency_ms"]["p95"] for r in reps) / 1000.0,
                "error_rate": max(r["error_rate"] for r in reps),
                "tokens_per_request": {
                    "prompt": sum(t.get("prompt", 0) for t in totals) / ok if ok else None,
                    "completion": sum(t.get("completion", 0) for t in totals) / ok if ok else None,
                    "cached": (
                        sum(c for c in cached if c is not None) / ok
                        if ok and all(c is not None for c in cached)
                        else None
                    ),
                },
            }
        )
    return out


def operating_points(levels: Sequence[Mapping[str, Any]], slo_s: float) -> dict[str, Any]:
    """Latency floor (lowest concurrency), the best level meeting the SLO with no errors (or
    None), and saturation (most requests/s)."""
    meeting = [lv for lv in levels if lv["p95_s"] <= slo_s and lv["error_rate"] == 0]
    return {
        "floor": min(levels, key=lambda lv: lv["concurrency"]),
        "slo": max(meeting, key=lambda lv: lv["requests_per_s"]) if meeting else None,
        "saturation": max(levels, key=lambda lv: lv["requests_per_s"]),
    }


def gpu_cost(
    hourly_usd: float, level: Mapping[str, Any] | None, calls: float, hours_per_month: float
) -> dict[str, Any] | None:
    """USD per 1,000 questions at a level's throughput (full utilization) and the questions
    one GPU answers in a month there."""
    if level is None or not level["requests_per_s"]:
        return None
    questions_per_hour = level["requests_per_s"] * 3600.0 / calls
    return {
        "usd_per_1k": round(hourly_usd / questions_per_hour * 1000.0, 4),
        "questions_per_month": round(questions_per_hour * hours_per_month),
    }


def hosted_cost(price: Price, level: Mapping[str, Any], calls: float) -> dict[str, Any]:
    """USD per 1,000 questions from a level's mean tokens per request: with the cached prompt
    tokens the provider reported, and with every prompt token priced as a miss (upper bound)."""
    t = level["tokens_per_request"]
    prompt, completion, cached = t["prompt"] or 0.0, t["completion"] or 0.0, t["cached"]
    per_call_miss = price.cost_usd(round(prompt), round(completion))
    per_call = (
        per_call_miss
        if cached is None
        else price.cost_usd(round(prompt), round(completion), round(cached))
    )
    return {
        "usd_per_1k": round(per_call * calls * 1000.0, 4),
        "usd_per_1k_all_miss": round(per_call_miss * calls * 1000.0, 4),
        "cached_share": (cached / prompt) if cached is not None and prompt else None,
    }


def paired_difference(
    a: Mapping[str, bool], b: Mapping[str, bool], resamples: int, seed: int, confidence: float
) -> dict[str, Any]:
    """Mean of (a correct - b correct) over the questions both answered, with a percentile
    bootstrap interval over questions (paired: one resample of ids serves both)."""
    ids = sorted(set(a) & set(b))
    if not ids:
        raise ValueError("no common questions")
    diffs = [int(a[i]) - int(b[i]) for i in ids]
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(resamples))
    tail = (1 - confidence) / 2
    return {
        "mean": sum(diffs) / n,
        "lo": percentile(means, tail),
        "hi": percentile(means, 1 - tail),
        "n": n,
        "a_only": sum(d == 1 for d in diffs),  # questions only a got right
        "b_only": sum(d == -1 for d in diffs),
    }


def evaluate(scenario: CompareScenario, root: Path, prices: Mapping[str, Price]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    correct: dict[str, dict[str, bool]] = {}
    hosted_per_q: dict[str, float] = {}
    splits = set()
    for opt in scenario.options:
        acc = accuracy_of(read_json(root / opt.accuracy))
        if acc["split"] != "test":
            raise ValueError(f"{opt.name}: {opt.accuracy} is not a test-split run")
        splits.add(acc["split_sha256"])
        correct[opt.name] = acc.pop("correct")
        levels = levels_of(read_json(root / opt.load.results), opt.load)
        points = operating_points(levels, scenario.slo_p95_s)
        calls = acc["calls_per_question"]
        entry: dict[str, Any] = {
            "description": opt.description,
            **acc,
            "levels": levels,
            "floor_p95_s": points["floor"]["p95_s"],
            "slo_level": points["slo"]["concurrency"] if points["slo"] else None,
            "saturation_level": points["saturation"]["concurrency"],
            "saturation_p95_s": points["saturation"]["p95_s"],
            "cost_kind": opt.cost.kind,
        }
        if opt.cost.kind == "gpu":
            assert opt.cost.session is not None
            hourly = float(read_json(root / opt.cost.session)["metrics"]["hourly_usd"])
            entry["hourly_usd"] = hourly
            entry["monthly_usd"] = round(hourly * scenario.hours_per_month, 2)
            entry["cost_at_slo"] = gpu_cost(hourly, points["slo"], calls, scenario.hours_per_month)
            entry["cost_at_saturation"] = gpu_cost(
                hourly, points["saturation"], calls, scenario.hours_per_month
            )
        else:
            assert opt.cost.price is not None
            if opt.cost.price not in prices:
                raise ValueError(f"{opt.name}: no price for {opt.cost.price!r}")
            # Per-token pricing: the cost per question does not depend on the load level.
            cost = hosted_cost(prices[opt.cost.price], points["saturation"], calls)
            entry["cost"] = cost
            hosted_per_q[opt.name] = cost["usd_per_1k"] / 1000.0
        options[opt.name] = entry
    if len(splits) != 1:
        raise ValueError(f"options were evaluated on different test files: {sorted(splits)}")
    for entry in options.values():
        if entry["cost_kind"] == "gpu":
            entry["break_even_questions_per_month"] = {
                name: round(entry["monthly_usd"] / per_q) if per_q > 0 else None
                for name, per_q in hosted_per_q.items()
            }
    paired = {
        f"{a} - {b}": paired_difference(
            correct[a], correct[b], scenario.resamples, scenario.seed, scenario.confidence
        )
        for a, b in scenario.paired
    }
    return {"split_sha256": splits.pop(), "options": options, "paired": paired}


def pct(e: Mapping[str, Any]) -> str:
    return f"{100 * e['mean']:.1f}% [{100 * e['lo']:.1f}, {100 * e['hi']:.1f}]"


def markdown(metrics: Mapping[str, Any], slo_s: float) -> str:
    rows = [
        "| Option | Test accuracy | p95 at c=1 | Best level within "
        f"{slo_s:g} s p95 | Saturation (req/s, p95) | USD per 1k questions | Fixed USD/month |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, o in metrics["options"].items():
        sat = next(lv for lv in o["levels"] if lv["concurrency"] == o["saturation_level"])
        slo = "none"
        if o["slo_level"] is not None:
            slo = f"c={o['slo_level']}"
            if o["cost_kind"] == "gpu" and o["cost_at_slo"]:
                slo += f" (USD {o['cost_at_slo']['usd_per_1k']:.3f}/1k there)"
        if o["cost_kind"] == "gpu":
            # Saturation for every GPU option, so the column compares like with like.
            cost = f"{o['cost_at_saturation']['usd_per_1k']:.3f} (saturation, full use)"
            fixed = f"{o['monthly_usd']:.0f}"
        else:
            c = o["cost"]
            cost = f"{c['usd_per_1k']:.3f} (all-miss {c['usd_per_1k_all_miss']:.3f})"
            fixed = "0"
        rows.append(
            f"| {name} | {pct(o['accuracy'])} | {o['floor_p95_s']:.2f} s | {slo} | "
            f"{sat['requests_per_s']:.2f} at c={sat['concurrency']}, {sat['p95_s']:.2f} s | "
            f"{cost} | {fixed} |"
        )
    if metrics["paired"]:
        rows += ["", "| Paired difference (test) | Points [95% CI] | Only a / only b |"]
        rows += ["|---|---|---|"]
        for pair, d in metrics["paired"].items():
            rows.append(
                f"| {pair} | {100 * d['mean']:+.1f} [{100 * d['lo']:+.1f}, {100 * d['hi']:+.1f}]"
                f" | {d['a_only']} / {d['b_only']} |"
            )
    return "\n".join(rows)


def run_compare(scenario_path: Path, root: Path = Path()) -> Path:
    raw = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    scenario = CompareScenario.model_validate(raw)
    prices = load_settings(EvalSettings, "eval", config_dir=root / "configs").prices
    started = datetime.now(UTC)
    metrics = evaluate(scenario, root, prices)
    metrics["markdown"] = markdown(metrics, scenario.slo_p95_s)
    result = RunResult(
        run_id=new_ulid(),
        name=scenario.name,
        kind=KIND,
        started_at=started,
        ended_at=datetime.now(UTC),
        seed=scenario.seed,
        git=git_info(root),
        hardware=hardware(),
        packages=package_versions(("pydantic",)),
        config={"prices": {k: v.model_dump() for k, v in prices.items()}},
        scenario=raw,
        metrics=metrics,
        notes="Computed from the results files the scenario names, no model calls. GPU cost per "
        "1,000 questions assumes full utilization at the stated level; the fixed monthly cost "
        "is one GPU kept up. Hosted cost uses the load test's tokens per request.",
    )
    path = write_result(root / scenario.results_dir, result)
    log.info("llm comparison written", extra={"path": str(path)})
    return path
