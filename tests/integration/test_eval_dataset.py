"""The structured-query eval set on a real database (task 3.1 done criteria): the snapshot
regenerates to the pinned bytes (or matches them after `make data-pull`), loads into a migrated
database, and every gold SQL and gold plan executes as the read-only login and reproduces the
expected result stored in the dev and test files."""

from __future__ import annotations

import psycopg
import pytest

from tests.integration.conftest import REPO_ROOT, StackSettings
from vms.common.settings import load_settings
from vms.eval.cli import ensure_snapshot_files, run_golds
from vms.eval.dataset import load_items
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
