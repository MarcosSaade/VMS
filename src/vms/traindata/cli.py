"""`python -m vms.traindata`: synthetic (question, QueryPlan) training data (task 8.2).

states     generate the training states' snapshots and load each into its own database on the
           stack's Postgres; writes the states manifest (make traindata-states)
generate   append candidates from the hosted LLM (LLM_* in .env) until the candidate target or
           the cost cap in configs/traindata.v1.yaml; resumes from the candidates file
filter     check every candidate (validity, agreement of two generations on the executed
           result, duplicates, leakage), write the dataset and the report (results JSON)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg

from vms.bench.results import RunResult, git_info, hardware, package_versions, write_result
from vms.common.ids import new_ulid
from vms.common.logs import configure_logging, get_logger
from vms.common.settings import load_settings
from vms.eval.compare import Tolerance
from vms.eval.dataset import load_items
from vms.eval.settings import EvalSettings
from vms.query.fewshots import load_fewshots
from vms.query.llm import LlmEndpointSettings
from vms.query.settings import QuerySettings
from vms.serving.catalog_file import CATALOG_FILE, load_catalog_file
from vms.traindata.filter import (
    PairChecker,
    connection_executor,
    dedupe,
    pair_record,
    stage_counts,
)
from vms.traindata.generate import Budget, Generator, read_candidates
from vms.traindata.leakage import leakage_hits, masked
from vms.traindata.spec import TrainDataSpec, load_spec
from vms.traindata.states import (
    LoadedState,
    build_states,
    check_states,
    load_state,
    reader_conninfo,
)

log = get_logger(__name__)

SPEC = Path("configs/traindata.v1.yaml")
REPO_ROOT = Path()
KIND = "data.traindata"


def _settings() -> tuple[QuerySettings, EvalSettings]:
    configs = REPO_ROOT / "configs"
    # The planner runs as the service does, with .env's VMS_* overrides (the hosted model's
    # response_format), as `python -m vms.eval run` does.
    dotenv = REPO_ROOT / ".env"
    return (
        load_settings(
            QuerySettings,
            "query",
            config_dir=configs,
            env_file=dotenv if dotenv.is_file() else None,
        ),
        load_settings(EvalSettings, "eval", config_dir=configs),
    )


def _states(spec: TrainDataSpec, qs: QuerySettings) -> dict[str, LoadedState]:
    check_states(spec, REPO_ROOT)
    return {s.name: load_state(s, REPO_ROOT, list(qs.classes)) for s in spec.states}


def cmd_states(spec: TrainDataSpec) -> int:
    manifest = build_states(spec, REPO_ROOT)
    path = REPO_ROOT / spec.states_manifest
    path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    sys.stdout.write(f"wrote {path}\n")
    return 0


def cmd_generate(spec: TrainDataSpec) -> int:
    qs, es = _settings()
    states = _states(spec, qs)
    rows = read_candidates(REPO_ROOT / spec.candidates)
    budgets = [r["budget"] for r in rows if "budget" in r]
    budget = Budget(es.prices[spec.budget.model_price], spec.budget.max_cost_usd)
    if budgets:
        last = budgets[-1]
        budget.spent_usd, budget.calls = last["spent_usd"], last["calls"]
        budget.prompt_tokens = last["prompt_tokens"]
        budget.cached_tokens = last["cached_prompt_tokens"]
        budget.completion_tokens = last["completion_tokens"]
    start = 1 + max((r["call"] for r in rows), default=-1)
    endpoint = LlmEndpointSettings()
    if endpoint.model != spec.budget.model_price:
        raise SystemExit(
            f"LLM_MODEL is {endpoint.model!r}, the spec prices {spec.budget.model_price!r}"
        )
    gen = Generator(spec, qs, states, budget, REPO_ROOT, endpoint, REPO_ROOT / spec.candidates)
    gen.written = sum("question" in r for r in rows)
    next_call = asyncio.run(gen.run(start))
    sys.stdout.write(
        f"{gen.written} candidates after {next_call} calls; spent ${budget.spent_usd:.3f}\n"
    )
    return 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def cmd_filter(spec: TrainDataSpec) -> int:
    qs, es = _settings()
    states = _states(spec, qs)
    started = datetime.now(UTC)
    rows = read_candidates(REPO_ROOT / spec.candidates)
    candidates = [r for r in rows if "question" in r]
    errors = [r for r in rows if "error" in r]
    budgets = [r["budget"] for r in rows if "budget" in r]
    tol = Tolerance(es.compare.rel_tol, es.compare.abs_tol)
    checked = []
    by_state = {s.name: s for s in spec.states}
    for name, state in states.items():
        with psycopg.connect(
            reader_conninfo(by_state[name], spec.statement_timeout_ms), autocommit=True
        ) as conn:
            checker = PairChecker(qs, state, connection_executor(conn), tol)
            checked += [checker.check(r) for r in candidates if r["state"] == name]
    checked.sort(key=lambda c: c.row["id"])
    dedupe(checked)

    eval_catalog, _ = load_catalog_file(REPO_ROOT / CATALOG_FILE)
    items = load_items(REPO_ROOT / es.structured.items)
    fewshots = load_fewshots(REPO_ROOT / qs.assets.fewshots)
    ref = masked(
        [(i.id, i.question, eval_catalog) for i in items]
        + [(f"fewshot:{k}", f.question, eval_catalog) for k, f in enumerate(fewshots)]
    )
    kept = [c for c in checked if c.drop is None]
    train = masked([(c.row["id"], c.row["question"], states[c.row["state"]].catalog) for c in kept])
    lk = spec.leakage
    hits = leakage_hits(
        train, ref, min_ratio=lk.min_ratio, min_jaccard=lk.min_jaccard, min_cosine=lk.min_cosine
    )
    leaked = {h.train_id for h in hits}
    for c in kept:
        if c.row["id"] in leaked:
            c.drop = "leakage"
    final = [c for c in checked if c.drop is None]
    after = leakage_hits(
        [t for t in train if t[0] not in leaked],
        ref,
        min_ratio=lk.min_ratio,
        min_jaccard=lk.min_jaccard,
        min_cosine=lk.min_cosine,
    )

    out_dir = REPO_ROOT / spec.dataset_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = out_dir / "pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8") as fh:
        for c in final:
            fh.write(json.dumps(pair_record(c, states[c.row["state"]]), ensure_ascii=False) + "\n")
    records = [pair_record(c, states[c.row["state"]]) for c in final]
    result = RunResult(
        run_id=new_ulid(),
        name="traindata-v1",
        kind=KIND,
        started_at=started,
        ended_at=datetime.now(UTC),
        seed=spec.seed,
        git=git_info(REPO_ROOT),
        hardware=hardware(),
        packages=package_versions(("numpy", "psycopg", "pydantic", "sqlglot")),
        config={"spec": json.loads(spec.model_dump_json()), "spec_sha256": _sha256(SPEC)},
        metrics={
            "generation": {
                "candidates": len(candidates),
                "failed_calls": len(errors),
                "call_errors": Counter(e["error"][:60] for e in errors).most_common(5),
                "budget": budgets[-1] if budgets else None,
                "b_status": Counter(r.get("b_status") for r in candidates),
                "candidates_sha256": _sha256(REPO_ROOT / spec.candidates),
            },
            "filter": stage_counts(checked),
            "drop_details": Counter(
                f"{c.drop}: {c.detail[:80]}" for c in checked if c.detail
            ).most_common(10),
            "leakage": {
                "reference": len(ref),
                "hits": Counter(h.check for h in hits),
                "examples": [h.__dict__ for h in hits[:20]],
                "hits_after_drop": len(after),
            },
            "dataset": {
                "pairs": len(records),
                "file": str(spec.dataset_dir / "pairs.jsonl"),
                "sha256": _sha256(pairs_path),
                "by_state": Counter(r["state"] for r in records),
                "by_intent": Counter(r["plan"]["intent"] for r in records),
                "abstain": sum(r["abstain"] for r in records),
                "plan_a_equal": sum(r["plan_a_equal"] for r in records),
            },
        },
        cost_usd=budgets[-1]["spent_usd"] if budgets else 0.0,
        notes="SYNTHETIC DATA: questions and plans by a hosted LLM over synthetic training "
        "states. Leakage is checked on masked questions against every eval question (dev "
        "and test) and the few-shot bank; the lexical embedding is char 3-gram TF-IDF.",
    )
    path = write_result(REPO_ROOT / spec.results_dir, result)
    sys.stdout.write(f"{len(records)} pairs -> {pairs_path}; report {path}\n")
    return 0 if not after else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vms.traindata",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--spec", type=Path, default=SPEC)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("command", choices=["states", "generate", "filter"])
    args = parser.parse_args(argv)
    configure_logging("traindata", args.log_level)
    spec = load_spec(args.spec)
    commands: dict[str, Any] = {
        "states": cmd_states,
        "generate": cmd_generate,
        "filter": cmd_filter,
    }
    return int(commands[args.command](spec))
