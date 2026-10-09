"""Migrations apply, roll back and re-apply on a fresh database."""

from __future__ import annotations

import psycopg
import pytest
from alembic import command

from tests.integration.conftest import StackSettings, alembic_config

pytestmark = pytest.mark.integration


def tables(stack: StackSettings, db: str) -> set[str]:
    with psycopg.connect(f"{stack.postgres_dsn} dbname={db}") as conn:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        ).fetchall()
    return {r[0] for r in rows}


def test_upgrade_downgrade_upgrade(stack: StackSettings, fresh_db: str) -> None:
    config = alembic_config(stack, fresh_db)
    assert tables(stack, fresh_db) >= {"frames", "alembic_version"}
    command.downgrade(config, "base")
    assert "frames" not in tables(stack, fresh_db)
    command.upgrade(config, "head")
    assert "frames" in tables(stack, fresh_db)


def test_frames_constraints(stack: StackSettings, fresh_db: str) -> None:
    insert = (
        "INSERT INTO frames (frame_id, camera_id, run_id, seq, ts, object_key, size_bytes) "
        "VALUES (%s, 'cam', 'run', %s, now(), 'k', %s)"
    )
    with psycopg.connect(f"{stack.postgres_dsn} dbname={fresh_db}") as conn:
        conn.execute(insert, ("f1", 0, 10))
        status = conn.execute("SELECT status FROM frames WHERE frame_id = 'f1'").fetchone()
        assert status == ("pending",)
        for frame_id, seq, size in [("f2", 0, 10), ("f3", -1, 10), ("f4", 1, 0)]:
            with pytest.raises(psycopg.errors.IntegrityError), conn.transaction():
                conn.execute(insert, (frame_id, seq, size))
