"""Bootstrap against a migrated database: catalog sync and the query service's read-only login."""

from __future__ import annotations

import json
from pathlib import Path

import psycopg
import pytest
from psycopg import sql
from pydantic import SecretStr, ValidationError

from tests.integration.conftest import REPO_ROOT, StackSettings
from vms.common.ids import new_ulid
from vms.common.settings import load_settings
from vms.query.settings import QuerySettings
from vms.storage.catalog import CameraCatalog, load_catalog, sync_catalog
from vms.storage.cli import main
from vms.storage.roles import ensure_query_login

pytestmark = pytest.mark.integration

CATALOG = REPO_ROOT / "configs/cameras.yaml"


def owner(stack: StackSettings, db: str) -> psycopg.Connection[tuple[object, ...]]:
    return psycopg.connect(f"{stack.postgres_dsn} dbname={db}", autocommit=True)


def test_sync_catalog_is_idempotent_and_keeps_stale_rows(
    stack: StackSettings, fresh_db: str
) -> None:
    catalog = load_catalog(CATALOG)
    with owner(stack, fresh_db) as conn:
        first = sync_catalog(conn, catalog)
        second = sync_catalog(conn, catalog)
        assert first == second
        assert (first.cameras, first.zones) == (4, 3)
        assert first.stale_cameras == first.stale_zones == ()
        rows = conn.execute("SELECT zone_id, camera_id, polygon FROM zones ORDER BY 1").fetchall()
        assert [r[0] for r in rows] == [
            "cam-001:front_door",
            "cam-002:register_1",
            "cam-002:register_2",
        ]
        # The polygon as configured (zones are redrawn with `make zones`, so not a literal).
        assert rows[0][2] == [list(p) for p in catalog.cameras[0].zones[0].polygon]

        # Edit: move a zone, change a timezone, drop a camera from the config.
        data = catalog.model_dump(mode="json")
        data["cameras"][0]["zones"][0]["polygon"] = [[0, 0], [1, 0], [1, 1]]
        data["cameras"][1]["timezone"] = "America/Toronto"
        del data["cameras"][3]
        report = sync_catalog(conn, CameraCatalog.model_validate(data))
        assert report.stale_cameras == ("cam-004",)
        assert conn.execute(
            "SELECT polygon FROM zones WHERE zone_id = 'cam-001:front_door'"
        ).fetchone() == ([[0, 0], [1, 0], [1, 1]],)
        assert conn.execute(
            "SELECT timezone FROM cameras WHERE camera_id = 'cam-002'"
        ).fetchone() == ("America/Toronto",)
        assert conn.execute("SELECT count(*) FROM cameras").fetchone() == (4,)


def test_reader_can_read_the_allowlist_and_nothing_else(
    stack: StackSettings, fresh_db: str, reader_dsn: str
) -> None:
    settings = load_settings(QuerySettings, "query", config_dir=REPO_ROOT / "configs")
    with owner(stack, fresh_db) as conn:
        sync_catalog(conn, load_catalog(CATALOG))
    with psycopg.connect(reader_dsn, autocommit=True) as conn:
        for table, columns in settings.sql.tables.items():
            cols = sql.SQL(", ").join(sql.Identifier(c) for c in columns)
            conn.execute(sql.SQL("SELECT {} FROM {}").format(cols, sql.Identifier(table)))
        assert conn.execute("SELECT count(*) FROM cameras").fetchone() == (4,)

        denied = [
            "SELECT * FROM query_log",
            "SELECT status FROM frames",  # column outside the allowlist
            "SELECT run_id FROM frames",
            "SELECT det_idx FROM detections",
            "SELECT * FROM frames",  # * includes ungranted columns
            "INSERT INTO cameras VALUES ('x', 'x', 's', 'UTC')",
            "UPDATE cameras SET site = 'x'",
            "DELETE FROM events",
            "TRUNCATE detections",
            "CREATE TABLE evil (x int)",
            "CREATE TEMP TABLE evil (x int)",
            "DROP TABLE cameras",
        ]
        for statement in denied:
            with pytest.raises(
                (psycopg.errors.InsufficientPrivilege, psycopg.errors.ReadOnlySqlTransaction)
            ):
                conn.execute(statement)  # type: ignore[arg-type]

        # Read-only even if a write privilege were granted by mistake.
        assert conn.execute("SHOW default_transaction_read_only").fetchone() == ("on",)

        # The session can switch read-only off; the grants alone must still deny every write.
        conn.execute("SET default_transaction_read_only = off")
        for statement in denied[5:]:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute(statement)  # type: ignore[arg-type]


def test_read_only_login_must_not_be_the_owner(stack: StackSettings, fresh_db: str) -> None:
    with owner(stack, fresh_db) as conn, pytest.raises(ValueError, match="must not be"):
        ensure_query_login(conn, stack.postgres_user, SecretStr("x"))


def test_bootstrap_cli(
    stack: StackSettings, fresh_db: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = f"vms_it_reader_{new_ulid().lower()}"
    monkeypatch.setenv("POSTGRES_READONLY_USER", user)
    monkeypatch.setenv("POSTGRES_READONLY_PASSWORD", new_ulid())
    try:
        assert main(["bootstrap", "--db", fresh_db, "--catalog", str(CATALOG)]) == 0
        assert main(["bootstrap", "--db", fresh_db, "--catalog", str(CATALOG)]) == 0  # re-run
        with owner(stack, fresh_db) as conn:
            assert conn.execute("SELECT count(*) FROM zones").fetchone() == (3,)
            member = conn.execute(
                "SELECT pg_has_role(%s, 'vms_query_reader', 'MEMBER')", (user,)
            ).fetchone()
            assert member == (True,)
    finally:
        with owner(stack, "postgres") as conn:
            conn.execute(sql.SQL("DROP ROLE IF EXISTS {}").format(sql.Identifier(user)))

    # An invalid catalog fails at load, before the CLI connects to the database.
    bad = tmp_path / "cameras.yaml"
    bad.write_text(json.dumps({"version": 1, "cameras": [{"camera_id": "x"}]}))
    with pytest.raises(ValidationError, match="Field required"):
        main(["bootstrap", "--db", fresh_db, "--catalog", str(bad)])
