"""Plan -> Postgres SQL: golden SQL per intent, parameters never inlined, and every valid plan
shape compiles to SQL the validator accepts."""

from __future__ import annotations

import datetime as dt
import itertools
import re
from typing import Any

import pytest
from pydantic import ValidationError

from vms.query.catalog import QueryCatalog
from vms.query.compiler import CompiledQuery, CompileError, PostgresCompiler
from vms.query.plan import GroupKey, Intent, QueryPlan, ResultKind, Target
from vms.query.sql_validator import SqlValidator

UTC = dt.UTC
NOW = dt.datetime(2026, 10, 12, 13, 30, tzinfo=UTC)  # 14:30 London, 22:30 Tokyo
TODAY_LONDON = (
    dt.datetime(2026, 10, 11, 23, tzinfo=UTC),
    dt.datetime(2026, 10, 12, 23, tzinfo=UTC),
)
TODAY_TOKYO = (dt.datetime(2026, 10, 11, 15, tzinfo=UTC), dt.datetime(2026, 10, 12, 15, tzinfo=UTC))


@pytest.fixture
def compiler(catalog: QueryCatalog) -> PostgresCompiler:
    return PostgresCompiler(catalog, default_limit=100, max_limit=1000)


def compile_plan(compiler: PostgresCompiler, plan: dict[str, Any]) -> CompiledQuery:
    return compiler.compile(QueryPlan.model_validate(plan), NOW)


def bound(sql: str, params: dict[str, Any]) -> str:
    """The SQL with each placeholder replaced by repr(value): readable golden strings."""
    return re.sub(r"%\((p\d+)\)s", lambda m: repr(params[m.group(1)]), sql)


def ts(value: dt.datetime) -> str:
    return repr(value)


# --- golden SQL, one per intent ---------------------------------------------------------------


def test_count_tracks_today(compiler: PostgresCompiler) -> None:
    q = compile_plan(
        compiler,
        {
            "intent": "count",
            "target": "tracks",
            "filters": {
                "cameras": ["entrance"],
                "classes": ["person"],
                "time_range": {"kind": "named", "name": "today"},
            },
        },
    )
    start, end = TODAY_LONDON
    assert bound(q.sql, q.params) == (
        "SELECT count(*) AS n FROM tracks t "
        f"WHERE (t.camera_id IN ('cam-001') AND t.last_seen >= {ts(start)} "
        f"AND t.first_seen < {ts(end)}) AND t.class_name IN ('person') LIMIT 1"
    )
    assert q.result_kind is ResultKind.SCALAR


def test_count_events_by_zone_entries(compiler: PostgresCompiler) -> None:
    q = compile_plan(
        compiler,
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
    start = dt.datetime(2026, 10, 12, 5, tzinfo=UTC)
    end = dt.datetime(2026, 10, 12, 11, tzinfo=UTC)
    assert bound(q.sql, q.params) == (
        "SELECT count(*) AS n FROM events e "
        f"WHERE (e.camera_id IN ('cam-001') AND e.ts >= {ts(start)} AND e.ts < {ts(end)}) "
        "AND e.zone_id IN ('cam-001:front_door') AND e.event_type IN ('zone_enter') LIMIT 1"
    )


def test_count_spans_timezones_with_one_branch_each(compiler: PostgresCompiler) -> None:
    q = compile_plan(
        compiler,
        {
            "intent": "count",
            "target": "tracks",
            "group_by": ["camera"],
            "filters": {"time_range": {"kind": "named", "name": "today"}},
        },
    )
    (t0, t1), (l0, l1) = TODAY_TOKYO, TODAY_LONDON
    assert bound(q.sql, q.params) == (
        "SELECT c.name AS camera, count(*) AS n FROM tracks t "
        "JOIN cameras c ON c.camera_id = t.camera_id "
        f"WHERE ((t.camera_id IN ('cam-004') AND t.last_seen >= {ts(t0)} "
        f"AND t.first_seen < {ts(t1)}) "
        f"OR (t.camera_id IN ('cam-002', 'cam-001', 'cam-003') AND t.last_seen >= {ts(l0)} "
        f"AND t.first_seen < {ts(l1)})) "
        "GROUP BY c.name ORDER BY n DESC, camera LIMIT 1000"
    )
    assert q.result_kind is ResultKind.TABLE


def test_aggregate_confidence(compiler: PostgresCompiler) -> None:
    q = compile_plan(
        compiler,
        {
            "intent": "aggregate",
            "target": "detections",
            "measure": {"op": "avg", "field": "confidence"},
            "filters": {"classes": ["bicycle"], "min_confidence": 0.5, "cameras": ["entrance"]},
        },
    )
    assert bound(q.sql, q.params) == (
        "SELECT avg(d.confidence) AS value FROM detections d "
        "WHERE d.camera_id IN ('cam-001') AND d.class_name IN ('bicycle') "
        "AND d.confidence >= 0.5 LIMIT 1"
    )


def test_list_events(compiler: PostgresCompiler) -> None:
    q = compile_plan(
        compiler,
        {
            "intent": "list",
            "target": "events",
            "filters": {"cameras": ["checkout"], "event_types": ["loiter"]},
            "order": {"by": "time", "direction": "desc"},
            "limit": 10,
        },
    )
    assert bound(q.sql, q.params) == (
        "SELECT e.event_id, e.ts, c.name AS camera, z.name AS zone, e.event_type, e.track_id "
        "FROM events e JOIN cameras c ON c.camera_id = e.camera_id "
        "JOIN zones z ON z.zone_id = e.zone_id "
        "WHERE e.camera_id IN ('cam-002') AND e.event_type IN ('loiter') "
        "ORDER BY e.ts DESC, e.event_id DESC LIMIT 10"
    )
    assert q.limit == 10


def test_list_tracks_in_zone_uses_zone_entries(compiler: PostgresCompiler) -> None:
    q = compile_plan(
        compiler,
        {
            "intent": "list",
            "target": "tracks",
            "filters": {
                "zones": ["register_2"],
                "classes": ["person"],
                "time_range": {"kind": "last", "amount": 2, "unit": "hours"},
            },
            "order": {"by": "time", "direction": "asc"},
        },
    )
    start = NOW - dt.timedelta(hours=2)
    assert bound(q.sql, q.params) == (
        "SELECT t.track_id, c.name AS camera, t.class_name, t.first_seen, t.last_seen, "
        "t.n_frames FROM tracks t JOIN cameras c ON c.camera_id = t.camera_id "
        "WHERE t.camera_id IN ('cam-002') AND t.track_id IN (SELECT e.track_id FROM events e "
        "WHERE e.event_type = 'zone_enter' AND (e.camera_id IN ('cam-002') "
        f"AND e.ts >= {ts(start)} AND e.ts < {ts(NOW)}) "
        "AND e.zone_id IN ('cam-002:register_2')) AND t.class_name IN ('person') "
        "ORDER BY t.first_seen ASC, t.track_id ASC LIMIT 100"
    )


def test_trend_buckets_in_camera_local_time(compiler: PostgresCompiler) -> None:
    q = compile_plan(
        compiler,
        {
            "intent": "trend",
            "target": "tracks",
            "bucket": "hour",
            "filters": {"cameras": ["entrance"], "classes": ["person"]},
        },
    )
    bucket = "date_trunc('hour', t.first_seen AT TIME ZONE c.timezone)"
    assert bound(q.sql, q.params) == (
        f"SELECT {bucket} AS bucket, count(*) AS n FROM tracks t "
        "JOIN cameras c ON c.camera_id = t.camera_id "
        "WHERE t.camera_id IN ('cam-001') AND t.class_name IN ('person') "
        f"GROUP BY {bucket} ORDER BY bucket ASC LIMIT 1000"
    )
    assert q.result_kind is ResultKind.TIME_SERIES


def test_top_k(compiler: PostgresCompiler) -> None:
    q = compile_plan(
        compiler,
        {"intent": "top_k", "target": "tracks", "group_by": ["class_name"], "limit": 5},
    )
    assert bound(q.sql, q.params) == (
        "SELECT t.class_name, count(*) AS n FROM tracks t "
        "WHERE t.camera_id IN ('cam-002', 'cam-001', 'cam-004', 'cam-003') "
        "GROUP BY t.class_name ORDER BY n DESC, class_name LIMIT 5"
    )


def test_dwell_on_camera(compiler: PostgresCompiler) -> None:
    q = compile_plan(
        compiler,
        {
            "intent": "dwell",
            "target": "tracks",
            "measure": {"op": "max", "field": "dwell_seconds"},
            "filters": {"cameras": ["parking_lot"], "classes": ["truck"]},
        },
    )
    assert bound(q.sql, q.params) == (
        "SELECT max(EXTRACT(EPOCH FROM t.last_seen - t.first_seen)) AS value FROM tracks t "
        "WHERE t.camera_id IN ('cam-003') AND t.class_name IN ('truck') LIMIT 1"
    )


def test_dwell_in_zone_uses_visits(compiler: PostgresCompiler) -> None:
    q = compile_plan(
        compiler,
        {
            "intent": "dwell",
            "target": "tracks",
            "measure": {"op": "avg", "field": "dwell_seconds"},
            "group_by": ["zone"],
            "filters": {"cameras": ["checkout"], "classes": ["person"]},
        },
    )
    assert bound(q.sql, q.params) == (
        "SELECT z.name AS zone, avg(EXTRACT(EPOCH FROM v.exited - v.entered)) AS value "
        "FROM (SELECT e.track_id, e.camera_id, e.zone_id, "
        "min(e.ts) FILTER (WHERE e.event_type = 'zone_enter') AS entered, "
        "max(e.ts) FILTER (WHERE e.event_type = 'zone_exit') AS exited "
        "FROM events e WHERE e.event_type IN ('zone_enter', 'zone_exit') "
        "AND e.camera_id IN ('cam-002') GROUP BY e.track_id, e.camera_id, e.zone_id) v "
        "JOIN tracks t ON t.track_id = v.track_id JOIN zones z ON z.zone_id = v.zone_id "
        "WHERE v.exited > v.entered AND t.class_name IN ('person') "
        "GROUP BY z.name ORDER BY value DESC, zone LIMIT 1000"
    )


def test_frames_with_class(compiler: PostgresCompiler) -> None:
    q = compile_plan(
        compiler,
        {
            "intent": "frames",
            "target": "frames",
            "filters": {"cameras": ["parking_lot"], "classes": ["dog"], "min_confidence": 0.6},
            "limit": 20,
        },
    )
    assert bound(q.sql, q.params) == (
        "SELECT f.frame_id, f.ts, c.name AS camera, f.object_key, f.width, f.height "
        "FROM frames f JOIN cameras c ON c.camera_id = f.camera_id "
        "WHERE f.camera_id IN ('cam-003') AND f.frame_id IN (SELECT d.frame_id FROM detections d "
        "WHERE d.camera_id IN ('cam-003') AND d.class_name IN ('dog') AND d.confidence >= 0.6) "
        "ORDER BY f.ts DESC, f.frame_id DESC LIMIT 20"
    )
    assert q.result_kind is ResultKind.FRAMES


def test_tracks_grouped_by_zone_count_each_zone_once(compiler: PostgresCompiler) -> None:
    q = compile_plan(compiler, {"intent": "count", "target": "tracks", "group_by": ["zone"]})
    assert "JOIN (SELECT DISTINCT e.track_id, e.zone_id FROM events e" in q.sql
    assert "JOIN zones z ON z.zone_id = tz.zone_id" in q.sql


def test_events_by_class_join_tracks(compiler: PostgresCompiler) -> None:
    q = compile_plan(compiler, {"intent": "count", "target": "events", "group_by": ["class_name"]})
    assert "JOIN tracks t ON t.track_id = e.track_id" in q.sql
    assert q.sql.startswith("SELECT t.class_name, count(*) AS n")


# --- safety and errors ------------------------------------------------------------------------


HOSTILE = "car') OR 1=1; DROP TABLE detections; --"


def test_values_are_bound_never_inlined(catalog: QueryCatalog) -> None:
    # A hostile class name reaches the compiler only if semantic validation is bypassed; even
    # then it must stay a parameter value.
    compiler = PostgresCompiler(catalog, default_limit=100, max_limit=1000)
    for target in ("tracks", "detections", "events"):
        q = compile_plan(
            compiler,
            {"intent": "count", "target": target, "filters": {"classes": [HOSTILE, "it's"]}},
        )
        assert HOSTILE not in q.sql
        assert "DROP" not in q.sql
        assert "it's" not in q.sql
        assert HOSTILE in q.params.values()


def test_only_placeholders_and_constants_reach_sql(compiler: PostgresCompiler) -> None:
    q = compile_plan(
        compiler,
        {
            "intent": "count",
            "target": "detections",
            "filters": {"min_confidence": 0.75, "cameras": ["entrance"]},
        },
    )
    assert "0.75" not in q.sql
    assert "cam-001" not in q.sql
    # Every placeholder has a value and every value has a placeholder.
    assert set(re.findall(r"%\((p\d+)\)s", q.sql)) == set(q.params)


def test_limit_is_capped(catalog: QueryCatalog) -> None:
    compiler = PostgresCompiler(catalog, default_limit=10, max_limit=50)
    q = compile_plan(compiler, {"intent": "list", "target": "events", "limit": 1000})
    assert q.limit == 50
    assert q.sql.endswith("LIMIT 50")
    q = compile_plan(compiler, {"intent": "list", "target": "events"})
    assert q.limit == 10


def test_abstain_and_visual_query_do_not_compile(compiler: PostgresCompiler) -> None:
    with pytest.raises(CompileError, match="abstain"):
        compile_plan(compiler, {"intent": "abstain", "abstain_reason": "ambiguous"})
    with pytest.raises(CompileError, match="CLIP"):
        compile_plan(
            compiler, {"intent": "frames", "target": "frames", "visual_query": "a red car"}
        )


@pytest.mark.parametrize(
    "filters", [{"cameras": ["garage"]}, {"zones": ["vault"]}], ids=["camera", "zone"]
)
def test_unknown_names_fail(compiler: PostgresCompiler, filters: dict[str, Any]) -> None:
    with pytest.raises(CompileError, match="unknown"):
        compile_plan(compiler, {"intent": "count", "target": "events", "filters": filters})


def test_deterministic(compiler: PostgresCompiler) -> None:
    plan = {
        "intent": "trend",
        "target": "events",
        "bucket": "day",
        "group_by": ["zone", "event_type"],
        "filters": {"time_range": {"kind": "named", "name": "this_week"}},
    }
    assert compile_plan(compiler, plan) == compile_plan(compiler, plan)


def test_empty_catalog_matches_nothing() -> None:
    compiler = PostgresCompiler(
        QueryCatalog(cameras=(), zones=(), classes=()), default_limit=100, max_limit=1000
    )
    q = compile_plan(compiler, {"intent": "count", "target": "tracks"})
    assert "WHERE FALSE" in q.sql


# --- every valid plan shape passes the validator ----------------------------------------------


def all_plan_shapes() -> list[QueryPlan]:
    """Every intent x target x group_by combination with the filters each target accepts."""
    time_ranges = [
        None,
        {"kind": "named", "name": "last_night"},
        {"kind": "last", "amount": 3, "unit": "days"},
        {"kind": "absolute", "start": "2026-10-03T09:00:00", "end": "2026-10-03T17:00:00"},
    ]
    measures = {
        Intent.AGGREGATE: [
            {"op": "sum", "field": "n_frames"},
            {"op": "min", "field": "confidence"},
        ],
        Intent.DWELL: [{"op": "avg", "field": "dwell_seconds"}],
    }
    keys = list(GroupKey)
    group_bys = [[], *[[k] for k in keys], *[list(p) for p in itertools.combinations(keys, 2)]]
    plans = []
    for intent, target, group_by, tr, with_zone in itertools.product(
        [i for i in Intent if i is not Intent.ABSTAIN],
        Target,
        group_bys,
        time_ranges,
        [False, True],
    ):
        filters: dict[str, Any] = {"time_range": tr, "classes": ["person", "car"]}
        if target in (Target.EVENTS, Target.TRACKS) and with_zone:
            filters["zones"] = ["register_1", "register_2"]
        if target is Target.EVENTS:
            filters["event_types"] = ["zone_enter"]
        if target in (Target.DETECTIONS, Target.FRAMES):
            filters["min_confidence"] = 0.3
        for measure in measures.get(intent, [None]):
            raw = {
                "intent": intent,
                "target": target,
                "group_by": group_by,
                "filters": filters,
                "measure": measure,
                "bucket": "hour" if intent is Intent.TREND else None,
                "limit": 3 if intent is Intent.TOP_K else None,
            }
            try:
                plans.append(QueryPlan.model_validate(raw))
            except ValidationError:
                continue
    return plans


def test_every_plan_shape_compiles_to_valid_sql(
    compiler: PostgresCompiler, validator: SqlValidator
) -> None:
    plans = all_plan_shapes()
    assert len(plans) > 300
    assert {p.intent for p in plans} == set(Intent) - {Intent.ABSTAIN}
    for plan in plans:
        compiled = compiler.compile(plan, NOW)
        checked = validator.validate(compiled.sql)
        assert checked.limit == compiled.limit, plan
        assert set(re.findall(r"%\((p\d+)\)s", checked.sql)) == set(compiled.params), plan
