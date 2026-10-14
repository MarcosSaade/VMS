"""The structured-query eval set on a real database (task 3.1 done criteria): the snapshot
regenerates to the pinned bytes (or matches them after `make data-pull`), loads into a migrated
database, and every gold SQL and gold plan executes as the read-only login and reproduces the
expected result stored in the dev and test files."""

from __future__ import annotations

import asyncio

import psycopg
import pytest

from tests.integration.conftest import REPO_ROOT, StackSettings
from vms.common.settings import load_settings
from vms.eval.cli import ensure_snapshot_files, query_settings, run_golds
from vms.eval.compare import Tolerance
from vms.eval.dataset import load_items
from vms.eval.harness import db_fingerprint, gold_llm, plan_engine, run_items
from vms.eval.scorer import ItemScore
from vms.eval.settings import EvalSettings
from vms.eval.snapshot import load_tables
from vms.storage.catalog import load_catalog, sync_catalog

pytestmark = pytest.mark.integration


def test_golds_reproduce_the_stored_results(
    stack: StackSettings, fresh_db: str, reader_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(REPO_ROOT)
    settings = load_settings(EvalSettings, "eval", config_dir=REPO_ROOT / "configs")
    manifest = ensure_snapshot_files(settings)  # generates if missing; checks the sha256s

    with psycopg.connect(f"{stack.postgres_dsn} dbname={fresh_db}", autocommit=True) as conn:
        sync_catalog(conn, load_catalog(REPO_ROOT / manifest.catalog))
        assert load_tables(conn, settings.snapshot.dir) == manifest.rows

    s = settings.structured
    stored = {i.id: i for i in [*load_items(s.dev), *load_items(s.test)]}
    items = load_items(s.items)
    with psycopg.connect(reader_dsn, autocommit=True) as conn:
        built, problems = run_golds(settings, items, conn, manifest.now)
    assert problems == []
    # Stored results are canonical (sorted unless ordered, floats rounded), so a rebuild here
    # reproduces them exactly, and with them the test split's hash.
    for item in built:
        assert item.expected == stored[item.id].expected, item.id


def test_harness_scores_the_gold_planner_perfectly(
    stack: StackSettings, fresh_db: str, reader_dsn: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The harness self-test (task 3.2): the real engine, planner prompt, compiler and read-only
    login, with each item's gold plan as the LLM's reply, on every dev and test item. Anything
    below 100% is a bug in the harness, the scorer or the dataset."""
    monkeypatch.chdir(REPO_ROOT)
    settings = load_settings(EvalSettings, "eval", config_dir=REPO_ROOT / "configs")
    manifest = ensure_snapshot_files(settings)
    owner = f"{stack.postgres_dsn} dbname={fresh_db}"
    with psycopg.connect(owner, autocommit=True) as conn:
        sync_catalog(conn, load_catalog(REPO_ROOT / manifest.catalog))
        load_tables(conn, settings.snapshot.dir)
    s = settings.structured
    items = [*load_items(s.dev), *load_items(s.test)]
    qs = query_settings(REPO_ROOT)

    async def go() -> list[ItemScore]:
        llm = gold_llm(items)
        async with plan_engine(reader_dsn, qs, manifest.now, llm, REPO_ROOT) as engine:
            return await run_items(
                engine,
                items,
                concurrency=4,
                tolerance=Tolerance(),
                fingerprint=lambda: db_fingerprint(owner),
            )

    scores = asyncio.run(go())
    wrong = [(sc.id, sc.error_class, sc.detail) for sc in scores if not sc.correct]
    assert wrong == []
    assert not any(sc.breach for sc in scores)
