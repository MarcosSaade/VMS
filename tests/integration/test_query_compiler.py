"""Compiled plans run on the migrated schema as the read-only login, and give the answers the
compiler's documented semantics promise on a small hand-built dataset."""

from __future__ import annotations

import datetime as dt
from typing import Any

import psycopg
import pytest

from tests.integration.conftest import REPO_ROOT, StackSettings
from tests.unit.query.test_compiler import NOW, all_plan_shapes
from vms.common.settings import load_settings
from vms.query.catalog import CAMERAS_SQL, ZONES_SQL, QueryCatalog, catalog_from_rows
from vms.query.compiler import PostgresCompiler
from vms.query.plan import QueryPlan
from vms.query.settings import QuerySettings
from vms.query.sql_validator import SqlValidator
from vms.storage.catalog import load_catalog, sync_catalog

pytestmark = pytest.mark.integration

UTC = dt.UTC


def at(hour: int, minute: int = 0) -> dt.datetime:
    """A time on 2026-10-12 (UTC); London is UTC+1 that day."""
    return dt.datetime(2026, 10, 12, hour, minute, tzinfo=UTC)


TRACKS = [
    # track_id, camera_id, class, first_seen, last_seen, n_frames
    ("cam-001:a:1", "cam-001", "person", at(7), at(7, 1), 10),
    ("cam-001:a:2", "cam-001", "person", at(10), at(10, 5), 30),
    ("cam-001:a:3", "cam-001", "car", at(10), at(10, 2), 5),
    ("cam-002:a:1", "cam-002", "person", at(12), at(12, 10), 50),
    # Seen yesterday evening only (London): not "today".
    (
        "cam-002:a:2",
        "cam-002",
        "person",
        at(0) - dt.timedelta(hours=3),
        at(0) - dt.timedelta(hours=2),
        4,
    ),
]
EVENTS = [
    # event_id, ts, camera_id, track_id, zone_id, type
    ("e1", at(7), "cam-001", "cam-001:a:1", "cam-001:front_door", "zone_enter"),
    ("e2", at(7, 1), "cam-001", "cam-001:a:1", "cam-001:front_door", "zone_exit"),
    ("e3", at(10), "cam-001", "cam-001:a:2", "cam-001:front_door", "zone_enter"),
    ("e4", at(10, 3), "cam-001", "cam-001:a:2", "cam-001:front_door", "zone_exit"),
    ("e5", at(10, 4), "cam-001", "cam-001:a:2", "cam-001:front_door", "zone_enter"),
    ("e6", at(12), "cam-002", "cam-002:a:1", "cam-002:register_1", "zone_enter"),
]


@pytest.fixture
def seeded(stack: StackSettings, fresh_db: str) -> QueryCatalog:
    with psycopg.connect(f"{stack.postgres_dsn} dbname={fresh_db}", autocommit=True) as conn:
        sync_catalog(conn, load_catalog(REPO_ROOT / "configs/cameras.yaml"))
        with conn.cursor() as cur:
            cur.executemany("INSERT INTO tracks VALUES (%s, %s, %s, %s, %s, %s)", TRACKS)
            cur.executemany("INSERT INTO events VALUES (%s, %s, %s, %s, %s, %s)", EVENTS)
        cameras = conn.execute(CAMERAS_SQL).fetchall()
        zones = conn.execute(ZONES_SQL).fetchall()
    return catalog_from_rows(cameras, zones, ["person", "car"])


def run(reader_dsn: str, catalog: QueryCatalog, plan: dict[str, Any]) -> list[tuple[Any, ...]]:
    settings = load_settings(QuerySettings, "query", config_dir=REPO_ROOT / "configs")
    compiled = PostgresCompiler(catalog, default_limit=100, max_limit=1000).compile(
        QueryPlan.model_validate(plan), NOW
    )
    sql = SqlValidator(settings.sql).validate(compiled.sql).sql
    with psycopg.connect(reader_dsn) as conn:
        return conn.execute(sql, compiled.params).fetchall()  # type: ignore[arg-type]


def test_every_plan_shape_runs_as_reader(seeded: QueryCatalog, reader_dsn: str) -> None:
    settings = load_settings(QuerySettings, "query", config_dir=REPO_ROOT / "configs")
    validator = SqlValidator(settings.sql)
    compiler = PostgresCompiler(seeded, default_limit=100, max_limit=1000)
    with psycopg.connect(reader_dsn, autocommit=True) as conn:
        for plan in all_plan_shapes():
            compiled = compiler.compile(plan, NOW)
            sql = validator.validate(compiled.sql).sql
            try:
                conn.execute(sql, compiled.params)  # type: ignore[arg-type]
            except psycopg.Error as exc:
                pytest.fail(f"{plan.model_dump_json(exclude_defaults=True)}: {exc}\n{sql}")


def test_people_today_at_entrance(seeded: QueryCatalog, reader_dsn: str) -> None:
    rows = run(
        reader_dsn,
        seeded,
        {
            "intent": "count",
            "target": "tracks",
            "group_by": ["camera"],
            "filters": {"classes": ["person"], "time_range": {"kind": "named", "name": "today"}},
        },
    )
    # checkout's second person was seen yesterday evening, London time.
    assert rows == [("entrance", 2), ("checkout", 1)]


def test_this_morning_is_london_time(seeded: QueryCatalog, reader_dsn: str) -> None:
    # 06:00-12:00 London = 05:00-11:00 UTC: entries at 07:00, 10:00 and 10:04 UTC.
    rows = run(
        reader_dsn,
        seeded,
        {
            "intent": "count",
            "target": "events",
            "filters": {
                "zones": ["front_door"],
                "event_types": ["zone_enter"],
                "time_range": {"kind": "named", "name": "this_morning"},
            },
        },
    )
    assert rows == [(3,)]


def test_tracks_in_zone_counted_once(seeded: QueryCatalog, reader_dsn: str) -> None:
    rows = run(reader_dsn, seeded, {"intent": "count", "target": "tracks", "group_by": ["zone"]})
    # a:2 entered front_door twice but is one track.
    assert rows == [("front_door", 2), ("register_1", 1)]


def test_dwell_in_zone(seeded: QueryCatalog, reader_dsn: str) -> None:
    rows = run(
        reader_dsn,
        seeded,
        {
            "intent": "dwell",
            "target": "tracks",
            "measure": {"op": "avg", "field": "dwell_seconds"},
            "filters": {"zones": ["front_door"]},
        },
    )
    # a:1 60 s; a:2 first enter 10:00 to last exit 10:03 = 180 s; register_1 has no exit.
    assert float(rows[0][0]) == pytest.approx(120.0)


def test_trend_buckets_are_local_hours(seeded: QueryCatalog, reader_dsn: str) -> None:
    rows = run(
        reader_dsn,
        seeded,
        {
            "intent": "trend",
            "target": "tracks",
            "bucket": "hour",
            "filters": {"cameras": ["entrance"], "classes": ["person"]},
        },
    )
    assert rows == [(dt.datetime(2026, 10, 12, 8), 1), (dt.datetime(2026, 10, 12, 11), 1)]
