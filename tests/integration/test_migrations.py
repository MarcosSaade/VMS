"""Migrations apply, roll back and re-apply on a fresh database; constraints hold."""

from __future__ import annotations

import psycopg
import pytest
from alembic import command

from tests.integration.conftest import StackSettings, alembic_config

pytestmark = pytest.mark.integration

SCHEMA_V1 = {"frames", "cameras", "zones", "detections", "tracks", "events", "query_log"}


def tables(stack: StackSettings, db: str) -> set[str]:
    with psycopg.connect(f"{stack.postgres_dsn} dbname={db}") as conn:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
        ).fetchall()
    return {r[0] for r in rows}


def reader_grants(stack: StackSettings, db: str) -> int:
    with psycopg.connect(f"{stack.postgres_dsn} dbname={db}") as conn:
        row = conn.execute(
            "SELECT count(*) FROM information_schema.column_privileges "
            "WHERE grantee = 'vms_query_reader'"
        ).fetchone()
    assert row is not None
    return int(row[0])


def test_upgrade_downgrade_upgrade(stack: StackSettings, fresh_db: str) -> None:
    config = alembic_config(stack, fresh_db)
    assert tables(stack, fresh_db) == SCHEMA_V1 | {"alembic_version"}
    granted = reader_grants(stack, fresh_db)
    assert granted > 0

    command.downgrade(config, "0002")
    assert reader_grants(stack, fresh_db) == 0  # revoked, though the role itself is kept
    command.downgrade(config, "0001")
    assert tables(stack, fresh_db) == {"frames", "alembic_version"}
    command.downgrade(config, "base")
    assert tables(stack, fresh_db) == {"alembic_version"}

    command.upgrade(config, "head")
    assert tables(stack, fresh_db) == SCHEMA_V1 | {"alembic_version"}
    assert reader_grants(stack, fresh_db) == granted


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


DETECTION = (
    "INSERT INTO detections (frame_id, det_idx, ts, camera_id, class_name, confidence, "
    "x1, y1, x2, y2, model_version) VALUES (%s, %s, now(), 'cam', 'person', %s, %s, 0, 10, 10, 'm')"
)


@pytest.mark.parametrize(
    ("statement", "params"),
    [
        (DETECTION, ("f", 0, 0.5, 0)),  # same (frame_id, det_idx) as the seeded row
        (DETECTION, ("f", 1, 1.5, 0)),  # confidence above 1
        (DETECTION, ("f", 2, 0.5, 20)),  # x1 > x2
        (DETECTION, ("f", -1, 0.5, 0)),  # negative index
        (
            "INSERT INTO events VALUES ('e', now(), 'cam', 't', 'z', 'zone_teleport')",
            None,
        ),
        (
            "INSERT INTO tracks VALUES ('t', 'cam', 'person', now(), now() - interval '1 s', 1)",
            None,
        ),
        ("INSERT INTO tracks VALUES ('t', 'cam', 'person', now(), now(), 0)", None),
        (
            "INSERT INTO zones VALUES ('z', 'missing-camera', 'door', '[[0,0],[1,0],[1,1]]')",
            None,
        ),
        ("INSERT INTO cameras VALUES ('c2', 'entrance', 's', 'UTC')", None),  # name taken
        ("INSERT INTO zones VALUES ('z2', 'c', 'door', '[[0,0],[1,0]]')", None),  # 2 points
        (
            "INSERT INTO query_log (query_id, mode, engine, question, latency_ms, success, "
            "error_type) VALUES ('q', 'structured', 'plan', 'x', 1, true, 'planner')",
            None,
        ),
    ],
)
def test_schema_v1_constraints(
    stack: StackSettings, fresh_db: str, statement: str, params: tuple[object, ...] | None
) -> None:
    with psycopg.connect(f"{stack.postgres_dsn} dbname={fresh_db}") as conn:
        conn.execute("INSERT INTO cameras VALUES ('c', 'entrance', 's', 'UTC')")
        conn.execute(DETECTION, ("f", 0, 0.5, 0))
        with pytest.raises(psycopg.errors.IntegrityError):
            conn.execute(statement, params)
