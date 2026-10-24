"""The D7 comparison (task 8.5): accuracy and calls per question from eval results, load levels
and operating points, GPU and hosted cost per 1,000 questions, break-even volume, the paired
bootstrap, and the refusals (a dev run, two different test files). Results files are written
by the tests; no model is called."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from vms.eval.settings import Price
from vms.serving import compare
from vms.serving.compare import CompareScenario, LoadRef

SHA = "a" * 64


def eval_file(
    path: Path, correct: dict[str, bool], repaired: set[str], split: str = "test"
) -> Path:
    items = [{"id": i, "correct": c, "repaired": i in repaired} for i, c in correct.items()]
    mean = sum(correct.values()) / len(correct)
    data = {
        "metrics": {
            "run": {"split": split, "split_sha256": SHA},
            "summary": {"accuracy": {"mean": mean, "lo": mean - 0.1, "hi": mean + 0.1, "n": 4}},
            "items": items,
        }
    }
    path.write_text(json.dumps(data))
    return path


def rep(rps: float, p95_ms: float, tokens: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "requests": 10,
        "errors": 0,
        "error_rate": 0.0,
        "requests_per_s": rps,
        "latency_ms": {"p50": p95_ms / 2, "p95": p95_ms},
        "tokens_total": tokens or {"prompt": 1000, "completion": 100, "cached": None},
    }


def load_file(path: Path, levels: dict[int, list[dict[str, Any]]]) -> Path:
    load = {
        str(c): {
            "reps": reps,
            "spread": {
                "requests_per_s": {"mean": sum(r["requests_per_s"] for r in reps) / len(reps)}
            },
        }
        for c, reps in levels.items()
    }
    data = {"metrics": {"servers": {"s": {"ok": True, "variants": {"v": {"load": load}}}}}}
    path.write_text(json.dumps(data))
    return path


def test_levels_use_the_worst_repetitions_p95_and_mean_throughput(tmp_path: Path) -> None:
    path = load_file(tmp_path / "l.json", {4: [rep(2.0, 1000), rep(4.0, 1400)], 1: [rep(1.0, 900)]})
    levels = compare.levels_of(
        json.loads(path.read_text()), LoadRef(results=path, server="s", variant="v")
    )
    assert [lv["concurrency"] for lv in levels] == [1, 4]
    assert levels[1]["requests_per_s"] == 3.0
    assert levels[1]["p95_s"] == 1.4
    assert levels[1]["tokens_per_request"] == {"prompt": 100.0, "completion": 10.0, "cached": None}
    points = compare.operating_points(levels, slo_s=1.0)
    assert points["floor"]["concurrency"] == 1
    assert points["slo"]["concurrency"] == 1  # c=4 misses 1.0 s in its worst rep
    assert points["saturation"]["concurrency"] == 4
    assert compare.operating_points(levels, slo_s=0.5)["slo"] is None


def test_a_server_that_did_not_run_has_no_levels(tmp_path: Path) -> None:
    path = tmp_path / "l.json"
    path.write_text(json.dumps({"metrics": {"servers": {"s": {"ok": False, "error": "OOM"}}}}))
    with pytest.raises(ValueError, match="OOM"):
        compare.levels_of(
            json.loads(path.read_text()), LoadRef(results=path, server="s", variant="v")
        )


def test_gpu_cost_is_the_hourly_price_over_questions_answered_at_full_use() -> None:
    level = {"requests_per_s": 2.0}
    cost = compare.gpu_cost(0.72, level, calls=1.2, hours_per_month=730)
    # 2 calls/s = 7,200 calls/h = 6,000 questions/h; $0.72 / 6,000 * 1,000 = $0.12
    assert cost == {"usd_per_1k": 0.12, "questions_per_month": 6000 * 730}
    assert compare.gpu_cost(0.72, None, 1.0, 730) is None


def test_hosted_cost_prices_cache_hits_and_reports_the_all_miss_bound() -> None:
    price = Price(prompt_per_mtok=0.30, completion_per_mtok=1.20, prompt_cache_hit_per_mtok=0.006)
    level = {"tokens_per_request": {"prompt": 5000.0, "completion": 100.0, "cached": 4000.0}}
    cost = compare.hosted_cost(price, level, calls=1.0)
    miss = (5000 * 0.30 + 100 * 1.20) / 1e6 * 1000
    hit = (1000 * 0.30 + 4000 * 0.006 + 100 * 1.20) / 1e6 * 1000
    assert cost["usd_per_1k_all_miss"] == pytest.approx(miss, abs=1e-4)
    assert cost["usd_per_1k"] == pytest.approx(hit, abs=1e-4)
    assert cost["cached_share"] == 0.8
    unknown = compare.hosted_cost(
        price, {"tokens_per_request": {**level["tokens_per_request"], "cached": None}}, 2.0
    )
    assert (
        unknown["usd_per_1k"] == unknown["usd_per_1k_all_miss"] == pytest.approx(2 * miss, abs=1e-4)
    )


def test_paired_difference_counts_discordant_questions_and_is_seeded() -> None:
    a = {"q1": True, "q2": True, "q3": False, "q4": True, "only-a": True}
    b = {"q1": True, "q2": False, "q3": False, "q4": False}
    d = compare.paired_difference(a, b, resamples=500, seed=7, confidence=0.95)
    assert d["n"] == 4  # questions both answered
    assert d["mean"] == 0.5
    assert (d["a_only"], d["b_only"]) == (2, 0)
    assert 0.0 <= d["lo"] <= 0.5 <= d["hi"] <= 1.0
    assert compare.paired_difference(a, b, 500, 7, 0.95) == d
    same = compare.paired_difference(a, a, 100, 1, 0.95)
    assert (same["mean"], same["lo"], same["hi"]) == (0.0, 0.0, 0.0)


def write_option_files(tmp_path: Path) -> dict[str, Any]:
    eval_file(tmp_path / "gpu.json", {"q1": True, "q2": False, "q3": True, "q4": False}, set())
    eval_file(tmp_path / "api.json", {"q1": True, "q2": True, "q3": True, "q4": True}, {"q1", "q2"})
    load_file(tmp_path / "gpu-load.json", {1: [rep(1.0, 900)], 16: [rep(8.0, 3000)]})
    tokens = {"prompt": 50000, "completion": 1000, "cached": 40000}
    load_file(
        tmp_path / "api-load.json", {1: [rep(0.5, 2000, tokens)], 16: [rep(6.0, 5000, tokens)]}
    )
    (tmp_path / "session.json").write_text(json.dumps({"metrics": {"hourly_usd": 0.6}}))
    return {
        "name": "d7",
        "seed": 1,
        "slo_p95_s": 1.5,
        "results_dir": str(tmp_path / "out"),
        "options": [
            {
                "name": "lora",
                "accuracy": "gpu.json",
                "load": {"results": "gpu-load.json", "server": "s", "variant": "v"},
                "cost": {"kind": "gpu", "session": "session.json"},
            },
            {
                "name": "hosted",
                "accuracy": "api.json",
                "load": {"results": "api-load.json", "server": "s", "variant": "v"},
                "cost": {"kind": "hosted", "price": "p"},
            },
        ],
        "paired": [["hosted", "lora"]],
    }


def test_evaluate_prices_each_option_and_the_break_even_volume(tmp_path: Path) -> None:
    raw = write_option_files(tmp_path)
    price = Price(prompt_per_mtok=0.30, completion_per_mtok=1.20, prompt_cache_hit_per_mtok=0.006)
    out = compare.evaluate(CompareScenario.model_validate(raw), tmp_path, {"p": price})
    lora, hosted = out["options"]["lora"], out["options"]["hosted"]
    assert (lora["slo_level"], lora["saturation_level"], lora["floor_p95_s"]) == (1, 16, 0.9)
    assert lora["calls_per_question"] == 1.0
    assert lora["monthly_usd"] == 0.6 * 730
    assert lora["cost_at_slo"]["usd_per_1k"] == pytest.approx(0.6 / 3600 * 1000, abs=1e-4)
    assert hosted["calls_per_question"] == 1.5  # two of four questions needed a repair
    assert hosted["slo_level"] is None
    per_q = hosted["cost"]["usd_per_1k"] / 1000
    assert lora["break_even_questions_per_month"] == {"hosted": round(0.6 * 730 / per_q)}
    assert out["paired"]["hosted - lora"]["mean"] == 0.5
    assert "| lora |" in compare.markdown(out, 1.5)


def test_options_must_be_test_runs_of_one_test_file(tmp_path: Path) -> None:
    raw = write_option_files(tmp_path)
    prices = {"p": Price(prompt_per_mtok=1, completion_per_mtok=1)}
    eval_file(tmp_path / "gpu.json", {"q1": True}, set(), split="dev")
    with pytest.raises(ValueError, match="not a test-split"):
        compare.evaluate(CompareScenario.model_validate(raw), tmp_path, prices)
    eval_file(tmp_path / "gpu.json", {"q1": True}, set())
    data = json.loads((tmp_path / "gpu.json").read_text())
    data["metrics"]["run"]["split_sha256"] = "b" * 64
    (tmp_path / "gpu.json").write_text(json.dumps(data))
    with pytest.raises(ValueError, match="different test files"):
        compare.evaluate(CompareScenario.model_validate(raw), tmp_path, prices)


def test_scenario_refusals() -> None:
    base = {"name": "d7", "seed": 1, "slo_p95_s": 1.5}
    gpu = {
        "name": "a",
        "accuracy": "a.json",
        "load": {"results": "l.json", "server": "s", "variant": "v"},
        "cost": {"kind": "gpu", "session": "s.json"},
    }
    with pytest.raises(ValueError, match="unique"):
        CompareScenario.model_validate({**base, "options": [gpu, gpu]})
    with pytest.raises(ValueError, match="unknown"):
        CompareScenario.model_validate({**base, "options": [gpu], "paired": [["a", "x"]]})
    with pytest.raises(ValueError, match="hourly"):
        CompareScenario.model_validate({**base, "options": [{**gpu, "cost": {"kind": "gpu"}}]})
    with pytest.raises(ValueError, match="price"):
        CompareScenario.model_validate({**base, "options": [{**gpu, "cost": {"kind": "hosted"}}]})
