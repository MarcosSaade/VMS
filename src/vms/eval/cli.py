"""`python -m vms.eval`: the structured-query eval dataset (task 3.1).

snapshot   generate the synthetic snapshot from its spec and seed into the DVC-tracked
           directory and write its manifest (seed, eval clock, file hashes). With --check,
           generate into a temporary directory and compare with the manifest instead
           (determinism check; writes nothing).
load       (re)create the eval database (configs/eval.yaml `db`): migrate, read-only login,
           eval catalog, COPY the snapshot. Files missing (no `make data-pull`) are generated
           first; either way they must match the manifest's hashes. Drops the eval database
           if it exists; it holds nothing but the snapshot.
build      run every item's gold SQL and gold plan on the eval database, require them to
           agree, and write the dev and test files and the dataset manifest.
check      offline consistency: items valid, no leakage into the few-shot bank, split files
           match the items, the seed and the manifest's hashes (no database needed).
review     print a split for human review: question, gold plan, gold SQL, expected rows.
run        ask every question of a split, score it and write a results JSON to eval/results/
           (task 3.2; see vms.eval.harness). --planner gold answers with the gold plans: a
           self-test that must score 100%.
report     Markdown tables of results files, for BENCHMARKS.md.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import psycopg
from alembic import command
from alembic.config import Config as AlembicConfig
from psycopg import sql

from vms.bench.results import RunResult, git_info, hardware, package_versions, write_result
from vms.common.ids import new_ulid
from vms.common.logs import configure_logging, get_logger
from vms.common.settings import load_settings
from vms.eval import snapshot as snap
from vms.eval.compare import Tolerance
from vms.eval.dataset import (
    DatasetManifest,
    EvalItem,
    Split,
    assign_splits,
    built_problems,
    leakage_problems,
    load_items,
    read_dataset_manifest,
    sha256_file,
    split_counts,
    structure_problems,
    write_dataset_manifest,
    write_items,
)
from vms.eval.gold import GoldChecker, connection_executor, load_catalog_rows
from vms.eval.harness import (
    PlannerChoice,
    build_llm_for,
    cost_usd,
    db_fingerprint,
    eval_reader_conninfo,
    plan_engine,
    run_items,
)
from vms.eval.report import render
from vms.eval.scorer import METRIC_NOTES, ItemScore, summarize
from vms.eval.settings import EvalSettings
from vms.query.fewshots import load_fewshots
from vms.query.llm import LlmEndpointSettings
from vms.query.settings import QuerySettings
from vms.storage.catalog import load_catalog, sync_catalog
from vms.storage.config import PostgresSettings
from vms.storage.roles import ReadonlyUserSettings, ensure_query_login

log = get_logger(__name__)
REPO_ROOT = Path.cwd()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vms.eval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log-level", default="INFO")
    commands = parser.add_subparsers(dest="command", required=True)
    snapshot = commands.add_parser("snapshot", help="generate the synthetic snapshot")
    snapshot.add_argument("--check", action="store_true", help="compare with the manifest only")
    commands.add_parser("load", help="(re)create the eval database from the snapshot")
    commands.add_parser("build", help="run golds, write dev/test files and the manifest")
    commands.add_parser("check", help="offline consistency checks")
    review = commands.add_parser("review", help="print a split for human review")
    review.add_argument("--split", choices=[s.value for s in Split], default=Split.TEST.value)
    review.add_argument("--max-rows", type=int, default=8)
    run = commands.add_parser("run", help="evaluate an engine on a split; write results JSON")
    run.add_argument("--suite", choices=["structured", "detection"], default="structured")
    run.add_argument("--split", choices=[s.value for s in Split], default=Split.DEV.value)
    run.add_argument("--planner", choices=["config", "gold"], default="config")
    run.add_argument("--limit", type=int, help="first N items only (smoke runs; not reportable)")
    run.add_argument("--ids", nargs="*", help="these item ids only (debugging; not reportable)")
    report = commands.add_parser("report", help="Markdown tables of results files")
    report.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)
    configure_logging("eval", args.log_level)
    settings = load_settings(EvalSettings, "eval")

    if args.command == "snapshot":
        return cmd_snapshot(settings, check=args.check)
    if args.command == "load":
        return cmd_load(settings)
    if args.command == "build":
        return cmd_build(settings)
    if args.command == "run":
        if args.suite == "detection":  # no split or planner: one frozen subset (D31)
            from vms.eval.detection.cli import main as detection_main

            return detection_main(["--log-level", args.log_level, "run"])
        return cmd_run(settings, Split(args.split), args.planner, args.limit, args.ids)
    if args.command == "report":
        results = [json.loads(p.read_text(encoding="utf-8")) for p in args.files]
        sys.stdout.write(render(results))
        return 0
    if args.command == "check":
        problems = check_dataset(settings, REPO_ROOT)
        return _report(problems, "dataset consistent")
    return cmd_review(settings, Split(args.split), args.max_rows)


def _report(problems: list[str], ok_message: str) -> int:
    for problem in problems:
        sys.stdout.write(f"  - {problem}\n")
    if problems:
        log.error("problems found", extra={"count": len(problems)})
        return 1
    log.info(ok_message)
    return 0


# --- snapshot --------------------------------------------------------------------------------


def cmd_snapshot(settings: EvalSettings, *, check: bool) -> int:
    paths = settings.snapshot
    if check:
        manifest = snap.read_manifest(paths.manifest)
        with tempfile.TemporaryDirectory() as tmp:
            fresh = snap.build(paths.spec, Path(tmp), REPO_ROOT)
        problems = [
            f"{name}: regenerated sha256 differs from the manifest"
            for name, digest in manifest.files.items()
            if fresh.files.get(name) != digest
        ]
        return _report(problems, "snapshot regenerates byte for byte")
    manifest = snap.build(paths.spec, paths.dir, REPO_ROOT)
    snap.write_manifest(paths.manifest, manifest)
    log.info("snapshot written", extra={"dir": str(paths.dir), "rows": manifest.rows})
    return 0


def ensure_snapshot_files(settings: EvalSettings) -> snap.SnapshotManifest:
    """The manifest, after making sure the files exist and match it (generating if missing)."""
    paths = settings.snapshot
    manifest = snap.read_manifest(paths.manifest)
    if any(not (paths.dir / name).is_file() for name in manifest.files):
        log.info("snapshot files missing (no dvc pull); generating", extra={"dir": str(paths.dir)})
        snap.build(paths.spec, paths.dir, REPO_ROOT)
    problems = snap.verify_files(paths.dir, manifest)
    if problems:
        raise SystemExit("snapshot does not match its manifest:\n  " + "\n  ".join(problems))
    return manifest


# --- load ------------------------------------------------------------------------------------


def alembic_config(pg: PostgresSettings) -> AlembicConfig:
    config = AlembicConfig(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("sqlalchemy.url", pg.sqlalchemy_url().replace("%", "%%"))
    return config


def create_eval_db(admin: PostgresSettings, db: str) -> PostgresSettings:
    """Drop and recreate `db`, migrated to head. Returns settings pointing at it."""
    if db == admin.db:
        raise SystemExit(f"the eval database must not be the dev database ({db!r})")
    maintenance = admin.model_copy(update={"db": "postgres"})
    with psycopg.connect(maintenance.conninfo(application_name="vms-eval"), autocommit=True) as c:
        c.execute(sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(sql.Identifier(db)))
        c.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db)))
    target = admin.model_copy(update={"db": db})
    command.upgrade(alembic_config(target), "head")
    return target


def load_snapshot(target: PostgresSettings, settings: EvalSettings, catalog_path: Path) -> None:
    manifest = snap.read_manifest(settings.snapshot.manifest)
    readonly = ReadonlyUserSettings()
    with psycopg.connect(target.conninfo(application_name="vms-eval")) as conn:
        ensure_query_login(conn, readonly.user, readonly.password)
        sync_catalog(conn, load_catalog(catalog_path))
        conn.commit()
        conn.autocommit = True
        counts = snap.load_tables(conn, settings.snapshot.dir)
    if counts != manifest.rows:
        raise SystemExit(f"loaded row counts {counts} differ from the manifest {manifest.rows}")
    log.info("eval database loaded", extra={"db": target.db, "rows": counts})


def cmd_load(settings: EvalSettings) -> int:
    manifest = ensure_snapshot_files(settings)
    target = create_eval_db(PostgresSettings(), settings.db)
    load_snapshot(target, settings, REPO_ROOT / manifest.catalog)
    return 0


# --- build and check -------------------------------------------------------------------------


def query_settings(root: Path, *, env_file: Path | None = None) -> QuerySettings:
    """The query service's settings: the SQL policy, classes and few-shot bank it runs with.
    `env_file` adds the VMS_* overrides from a dotenv file, as compose gives the service."""
    return load_settings(QuerySettings, "query", config_dir=root / "configs", env_file=env_file)


def offline_problems(settings: EvalSettings, items: list[EvalItem], root: Path) -> list[str]:
    qs = query_settings(root)
    fewshots = [(e.id, e.question) for e in load_fewshots(root / qs.assets.fewshots)]
    return [
        *structure_problems(items),
        *leakage_problems(
            items,
            fewshots,
            min_ratio=settings.leakage.min_ratio,
            min_jaccard=settings.leakage.min_jaccard,
        ),
    ]


def reader_conninfo(settings: EvalSettings) -> str:
    pg, readonly = PostgresSettings(), ReadonlyUserSettings()
    return (
        f"host={pg.host} port={pg.port} dbname={settings.db} user={readonly.user} "
        f"password={readonly.password.get_secret_value()} connect_timeout=5 "
        f"application_name=vms-eval options='-c statement_timeout={settings.statement_timeout_ms}'"
    )


def run_golds(
    settings: EvalSettings, items: list[EvalItem], conn: psycopg.Connection[Any], now: dt.datetime
) -> tuple[list[EvalItem], list[str]]:
    qs = query_settings(REPO_ROOT)
    checker = GoldChecker(
        qs,
        load_catalog_rows(conn, qs.classes),
        now,
        connection_executor(conn),
        Tolerance(settings.compare.rel_tol, settings.compare.abs_tol),
    )
    built: list[EvalItem] = []
    problems: list[str] = []
    for item in items:
        outcome = checker.check(item)
        problems.extend(f"{item.id}: {p}" for p in outcome.problems)
        built.append(item.model_copy(update={"expected": outcome.expected}))
    return built, problems


def cmd_build(settings: EvalSettings) -> int:
    s = settings.structured
    items = load_items(s.items)
    problems = offline_problems(settings, items, REPO_ROOT)
    if problems:
        return _report(problems, "")
    snapshot_manifest = snap.read_manifest(settings.snapshot.manifest)
    with psycopg.connect(reader_conninfo(settings), autocommit=True) as conn:
        built, problems = run_golds(settings, items, conn, snapshot_manifest.now)
    if problems:
        return _report(problems, "")
    splits = assign_splits(items, seed=s.split_seed, test_fraction=s.test_fraction)
    built = [b.model_copy(update={"split": splits[b.id]}) for b in built]
    dev_sha = write_items(s.dev, [b for b in built if b.split is Split.DEV])
    test_sha = write_items(s.test, [b for b in built if b.split is Split.TEST])
    qs = query_settings(REPO_ROOT)
    manifest = DatasetManifest(
        version=snapshot_manifest.version,
        items=str(s.items),
        items_sha256=sha256_file(s.items),
        snapshot_manifest=str(settings.snapshot.manifest),
        snapshot_files=snapshot_manifest.files,
        now=snapshot_manifest.now.isoformat(),
        split_seed=s.split_seed,
        test_fraction=s.test_fraction,
        dev_sha256=dev_sha,
        test_sha256=test_sha,
        counts=split_counts(built),
        fewshots=str(qs.assets.fewshots),
        fewshots_sha256=sha256_file(REPO_ROOT / qs.assets.fewshots),
    )
    write_dataset_manifest(s.manifest, manifest)
    log.info("dataset built", extra={"test_sha256": test_sha, "counts": manifest.counts})
    return 0


def check_dataset(settings: EvalSettings, root: Path) -> list[str]:
    """Everything about the dataset that can be checked without the database."""
    s = settings.structured
    items = load_items(root / s.items)
    problems = offline_problems(settings, items, root)
    manifest = read_dataset_manifest(root / s.manifest)
    snapshot_manifest = snap.read_manifest(root / settings.snapshot.manifest)
    built = [*load_items(root / s.dev), *load_items(root / s.test)]
    splits = assign_splits(items, seed=manifest.split_seed, test_fraction=manifest.test_fraction)
    problems.extend(built_problems(items, built, splits))
    for item in load_items(root / s.dev):
        if item.split is not Split.DEV:
            problems.append(f"{item.id}: in the dev file with split {item.split}")
    for item in load_items(root / s.test):
        if item.split is not Split.TEST:
            problems.append(f"{item.id}: in the test file with split {item.split}")
    pins = {
        "items_sha256": (manifest.items_sha256, sha256_file(root / s.items)),
        "dev_sha256": (manifest.dev_sha256, sha256_file(root / s.dev)),
        "test_sha256": (manifest.test_sha256, sha256_file(root / s.test)),
        "snapshot_files": (manifest.snapshot_files, snapshot_manifest.files),
        "now": (manifest.now, snapshot_manifest.now.isoformat()),
    }
    qs = query_settings(root)
    pins["fewshots_sha256"] = (
        manifest.fewshots_sha256,
        sha256_file(root / qs.assets.fewshots),
    )
    for name, (recorded, actual) in pins.items():
        if recorded != actual:
            problems.append(f"dataset manifest {name} is stale; rebuild (make eval-build)")
    return problems


# --- review ----------------------------------------------------------------------------------


def format_item(item: EvalItem, max_rows: int) -> str:
    lines = [
        f"### {item.id} [{item.category}] {item.question}",
        f"plan: {item.plan.compact_json()}",
        f"expressible_by_plan: {item.expressible_by_plan}   ordered: {item.ordered}"
        + (f"   group: {item.group}" if item.group else "")
        + (f"   tags: {', '.join(item.tags)}" if item.tags else ""),
    ]
    if item.notes:
        lines.append(f"notes: {item.notes}")
    if item.sql:
        lines.append(f"sql: {item.sql}")
    if item.expected is not None:
        rows = item.expected.rows
        lines.append(f"expected ({len(rows)} rows): columns {item.expected.columns}")
        lines.extend(f"  {json.dumps(r)}" for r in rows[:max_rows])
        if len(rows) > max_rows:
            lines.append(f"  ... {len(rows) - max_rows} more")
    return "\n".join(lines) + "\n"


def cmd_review(settings: EvalSettings, split: Split, max_rows: int) -> int:
    s = settings.structured
    items = load_items(s.test if split is Split.TEST else s.dev)
    manifest = snap.read_manifest(settings.snapshot.manifest)
    sys.stdout.write(
        f"{split} split: {len(items)} items. Eval clock (now): {manifest.now.isoformat()}. "
        "Relative periods are camera-local. Data is synthetic (D29).\n\n"
    )
    for item in items:
        sys.stdout.write(format_item(item, max_rows) + "\n")
    return 0


# --- run -------------------------------------------------------------------------------------


def owner_conninfo(settings: EvalSettings) -> str:
    pg = PostgresSettings().model_copy(update={"db": settings.db})
    return pg.conninfo(application_name="vms-eval")


def check_eval_db(settings: EvalSettings, manifest: snap.SnapshotManifest) -> None:
    """The eval database must hold the snapshot the gold answers were computed on."""
    try:
        with psycopg.connect(owner_conninfo(settings)) as conn:
            counts = {
                t: int(conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0])  # type: ignore[index]
                for t in snap.TABLES
            }
    except psycopg.Error as exc:
        raise SystemExit(
            f"eval database {settings.db!r} unavailable ({exc}); run make eval-db"
        ) from exc
    if counts != manifest.rows:
        raise SystemExit(
            f"eval database holds {counts}, not snapshot v{manifest.version}; run make eval-db"
        )


def select_items(
    settings: EvalSettings, split: Split, limit: int | None, ids: list[str] | None
) -> tuple[list[EvalItem], str]:
    """The split's items, after checking the file is the one the manifest pins."""
    s = settings.structured
    path = s.test if split is Split.TEST else s.dev
    manifest = read_dataset_manifest(s.manifest)
    recorded = manifest.test_sha256 if split is Split.TEST else manifest.dev_sha256
    actual = sha256_file(path)
    if actual != recorded:
        raise SystemExit(f"{path} does not match {s.manifest}; run make eval-build")
    items = load_items(path)
    if ids:
        items = [i for i in items if i.id in set(ids)]
    if limit is not None:
        items = items[:limit]
    return items, actual


def cmd_run(
    settings: EvalSettings,
    split: Split,
    planner: PlannerChoice,
    limit: int | None,
    ids: list[str] | None,
) -> int:
    items, split_sha = select_items(settings, split, limit, ids)
    if split is Split.TEST:
        log.warning("test split: use only for reported numbers, never for tuning (AGENTS 8)")
    manifest = snap.read_manifest(settings.snapshot.manifest)
    check_eval_db(settings, manifest)
    # The service runs with .env's VMS_* overrides (compose env_file), e.g. the LLM's
    # response_format; evaluate the same configuration.
    dotenv = REPO_ROOT / ".env"
    qs = query_settings(REPO_ROOT, env_file=dotenv if dotenv.is_file() else None)
    endpoint = LlmEndpointSettings()
    llm = build_llm_for(planner, qs, load_items(settings.structured.items), REPO_ROOT)
    model = "gold" if planner == "gold" else (endpoint.model or "canned")
    started = dt.datetime.now(dt.UTC)
    log.info("eval run", extra={"split": split.value, "items": len(items), "llm": model})

    def progress(score: ItemScore) -> None:
        log.info(
            "scored",
            extra={"id": score.id, "correct": score.correct, "error": score.error_class},
        )

    async def go() -> list[ItemScore]:
        async with plan_engine(
            eval_reader_conninfo(settings, qs), qs, manifest.now, llm, REPO_ROOT
        ) as engine:
            return await run_items(
                engine,
                items,
                concurrency=settings.run.concurrency,
                tolerance=Tolerance(settings.compare.rel_tol, settings.compare.abs_tol),
                fingerprint=lambda: db_fingerprint(owner_conninfo(settings)),
                progress=progress,
            )

    scores = asyncio.run(go())
    ended = dt.datetime.now(dt.UTC)
    summary = summarize(
        scores,
        resamples=settings.run.resamples,
        seed=settings.run.seed,
        confidence=settings.run.confidence,
    )
    cost = cost_usd(
        settings.prices, model, summary["tokens"]["prompt"], summary["tokens"]["completion"]
    )
    partial = limit is not None or bool(ids)
    result = RunResult(
        run_id=new_ulid(),
        name=f"structured-{split.value}-{_slug(model)}",
        kind="eval.structured",
        started_at=started,
        ended_at=ended,
        seed=settings.run.seed,
        git=git_info(REPO_ROOT),
        hardware=hardware(),
        packages=package_versions(("aiohttp", "psycopg", "pydantic", "sqlglot")),
        config={
            "eval": settings.model_dump(mode="json"),
            "query": {
                "llm": qs.llm.model_dump(mode="json"),
                "planner": qs.planner.model_dump(mode="json"),
                "assets": qs.assets.model_dump(mode="json"),
            },
            "llm_endpoint": {"base_url": endpoint.base_url, "model": endpoint.model},
        },
        metrics={
            "run": {
                "suite": "structured",
                "split": split.value,
                "split_sha256": split_sha,
                "planner": planner,
                "llm": model,
                "eval_clock": manifest.now.isoformat(),
                "snapshot_files": manifest.files,
                "partial": partial,
                "cost_usd": cost,
            },
            "summary": summary,
            "notes": {
                k: {"measures": m, "does_not_measure": d} for k, (m, d) in METRIC_NOTES.items()
            },
            "items": [_item_record(s) for s in scores],
        },
        cost_usd=cost or 0.0,
        notes="partial run (--limit/--ids): not reportable" if partial else "",
    )
    path = write_result(REPO_ROOT / settings.run.results_dir, result)
    sys.stdout.write(render([result.model_dump(mode="json")]))
    log.info("results written", extra={"path": str(path)})
    return 0


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "-." else "-" for c in text.lower()).strip("-")


def _item_record(score: ItemScore) -> dict[str, Any]:
    a = score.answer
    return {
        "id": score.id,
        "category": score.category.value,
        "correct": score.correct,
        "error_class": score.error_class.value if score.error_class else None,
        "detail": score.detail[:300],
        "breach": score.breach,
        "status": a.status,
        "abstain_reason": a.abstain_reason,
        "error_type": a.error_type,
        "plan": a.plan,
        "row_count": len(a.rows),
        "latency_ms": round(a.latency_ms, 1),
        "prompt_tokens": a.prompt_tokens,
        "completion_tokens": a.completion_tokens,
        "repaired": a.repaired,
    }
