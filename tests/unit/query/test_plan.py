import json
from typing import Any

import pytest
from pydantic import ValidationError

from vms.query.plan import (
    PLAN_LIMIT_MAX,
    SCHEMA_VERSION,
    AbsoluteTimeRange,
    Intent,
    LastTimeRange,
    NamedTimeRange,
    QueryPlan,
    ResultKind,
    plan_json_schema,
    result_kind,
)


def plan(**fields: Any) -> QueryPlan:
    return QueryPlan.model_validate(fields)


def errors_of(**fields: Any) -> str:
    with pytest.raises(ValidationError) as excinfo:
        QueryPlan.model_validate(fields)
    return str(excinfo.value)


# --- Valid plans, one per intent -------------------------------------------------------------

VALID: dict[str, dict[str, Any]] = {
    "count": {
        "intent": "count",
        "target": "tracks",
        "filters": {
            "cameras": ["entrance"],
            "classes": ["person"],
            "time_range": {"kind": "named", "name": "today"},
        },
    },
    "count_grouped": {
        "intent": "count",
        "target": "events",
        "group_by": ["zone", "event_type"],
        "filters": {"event_types": ["zone_enter", "zone_exit"]},
    },
    "aggregate": {
        "intent": "aggregate",
        "target": "detections",
        "measure": {"op": "avg", "field": "confidence"},
        "group_by": ["class_name"],
        "order": {"by": "value", "direction": "desc"},
    },
    "list": {
        "intent": "list",
        "target": "events",
        "filters": {"zones": ["front_door"]},
        "order": {"by": "time"},
        "limit": 20,
    },
    "trend": {
        "intent": "trend",
        "target": "tracks",
        "bucket": "hour",
        "filters": {"time_range": {"kind": "last", "amount": 24, "unit": "hours"}},
    },
    "top_k": {
        "intent": "top_k",
        "target": "tracks",
        "group_by": ["camera"],
        "limit": 3,
        "order": {"by": "count"},
    },
    "dwell": {
        "intent": "dwell",
        "target": "tracks",
        "measure": {"op": "avg", "field": "dwell_seconds"},
        "filters": {"zones": ["register_1"], "classes": ["person"]},
    },
    "frames_hybrid": {
        "intent": "frames",
        "target": "frames",
        "visual_query": "a red truck at night",
        "filters": {"cameras": ["loading_dock"], "min_confidence": 0.5},
        "limit": 12,
    },
    "abstain": {"intent": "abstain", "abstain_reason": "unsupported"},
}


@pytest.mark.parametrize("name", sorted(VALID))
def test_valid_plans_parse_and_round_trip(name: str) -> None:
    parsed = plan(**VALID[name])
    assert parsed.schema_version == SCHEMA_VERSION
    again = QueryPlan.model_validate_json(parsed.model_dump_json())
    assert again == parsed


def test_every_intent_has_a_valid_example() -> None:
    assert {plan(**p).intent for p in VALID.values()} == set(Intent)


def test_plans_are_immutable() -> None:
    parsed = plan(**VALID["count"])
    with pytest.raises(ValidationError):
        parsed.limit = 5  # type: ignore[misc]


# --- Schema-level rejections -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("fields", "fragment"),
    [
        ({"intent": "explode", "target": "tracks"}, "intent"),
        ({"intent": "count", "target": "tracks", "sql": "DROP TABLE x"}, "Extra inputs"),
        ({"intent": "count", "target": "tracks", "schema_version": "9.9"}, "schema_version"),
        ({"intent": "list", "target": "tracks", "limit": 0}, "limit"),
        ({"intent": "list", "target": "tracks", "limit": PLAN_LIMIT_MAX + 1}, "limit"),
        (
            {"intent": "count", "target": "detections", "filters": {"min_confidence": 1.5}},
            "min_confidence",
        ),
        ({"intent": "count", "target": "tracks", "filters": {"cameras": ["c"] * 21}}, "cameras"),
        ({"intent": "frames", "target": "frames", "visual_query": "x" * 201}, "visual_query"),
        ({"intent": "frames", "target": "frames", "visual_query": ""}, "visual_query"),
        (
            {
                "intent": "count",
                "target": "tracks",
                "filters": {"time_range": {"kind": "sometime"}},
            },
            "time_range",
        ),
        (
            {"intent": "count", "target": "events", "filters": {"event_types": ["teleport"]}},
            "event_types",
        ),
    ],
)
def test_schema_rejections(fields: dict[str, Any], fragment: str) -> None:
    assert fragment in errors_of(**fields)


def test_absolute_time_range_is_local_and_ordered() -> None:
    ok = AbsoluteTimeRange(start="2026-10-03T09:00:00", end="2026-10-03T10:00:00")  # type: ignore[arg-type]
    assert ok.start.tzinfo is None
    with pytest.raises(ValidationError, match="camera-local"):
        AbsoluteTimeRange(start="2026-10-03T09:00:00Z", end="2026-10-03T10:00:00Z")  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="before"):
        AbsoluteTimeRange(start="2026-10-03T10:00:00", end="2026-10-03T09:00:00")  # type: ignore[arg-type]


def test_time_range_discriminator_selects_the_model() -> None:
    for tr, cls in [
        (
            {"kind": "absolute", "start": "2026-10-03T09:00", "end": "2026-10-03T10:00"},
            AbsoluteTimeRange,
        ),
        ({"kind": "last", "amount": 2, "unit": "hours"}, LastTimeRange),
        ({"kind": "named", "name": "this_morning"}, NamedTimeRange),
    ]:
        parsed = plan(intent="count", target="tracks", filters={"time_range": tr})
        assert isinstance(parsed.filters.time_range, cls)


# --- Combination rules -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fields", "fragment"),
    [
        # abstain
        ({"intent": "abstain"}, "requires abstain_reason"),
        (
            {"intent": "abstain", "abstain_reason": "ambiguous", "target": "tracks"},
            "takes no other fields",
        ),
        (
            {
                "intent": "abstain",
                "abstain_reason": "ambiguous",
                "filters": {"cameras": ["entrance"]},
            },
            "filters",
        ),
        (
            {"intent": "count", "target": "tracks", "abstain_reason": "ambiguous"},
            "only allowed with intent 'abstain'",
        ),
        # target
        ({"intent": "count"}, "requires target"),
        ({"intent": "frames", "target": "tracks"}, "takes target frames"),
        (
            {
                "intent": "dwell",
                "target": "events",
                "measure": {"op": "avg", "field": "dwell_seconds"},
            },
            "takes target tracks",
        ),
        (
            {
                "intent": "aggregate",
                "target": "events",
                "measure": {"op": "avg", "field": "confidence"},
            },
            "takes target",
        ),
        # measure
        ({"intent": "aggregate", "target": "detections"}, "requires measure"),
        (
            {
                "intent": "count",
                "target": "detections",
                "measure": {"op": "avg", "field": "confidence"},
            },
            "measure is only allowed",
        ),
        (
            {
                "intent": "aggregate",
                "target": "tracks",
                "measure": {"op": "avg", "field": "confidence"},
            },
            "requires target 'detections'",
        ),
        (
            {
                "intent": "aggregate",
                "target": "tracks",
                "measure": {"op": "avg", "field": "dwell_seconds"},
            },
            "use intent 'dwell'",
        ),
        (
            {"intent": "dwell", "target": "tracks", "measure": {"op": "avg", "field": "n_frames"}},
            "dwell_seconds",
        ),
        # bucket
        ({"intent": "trend", "target": "tracks"}, "requires bucket"),
        (
            {"intent": "count", "target": "tracks", "bucket": "day"},
            "only allowed with intent 'trend'",
        ),
        # top_k and group_by
        ({"intent": "top_k", "target": "tracks", "limit": 3}, "exactly one group_by"),
        ({"intent": "top_k", "target": "tracks", "group_by": ["camera"]}, "requires limit"),
        ({"intent": "list", "target": "tracks", "group_by": ["camera"]}, "group_by is not allowed"),
        ({"intent": "count", "target": "tracks", "group_by": ["camera", "camera"]}, "unique"),
        # visual query and order
        (
            {"intent": "list", "target": "tracks", "visual_query": "red car"},
            "only allowed with intent 'frames'",
        ),
        ({"intent": "list", "target": "tracks", "order": {"by": "value"}}, "not valid for intent"),
        # filters and group keys versus target
        (
            {"intent": "count", "target": "tracks", "filters": {"event_types": ["loiter"]}},
            "requires target 'events'",
        ),
        (
            {"intent": "count", "target": "detections", "filters": {"zones": ["front_door"]}},
            "filters.zones",
        ),
        (
            {"intent": "count", "target": "tracks", "filters": {"min_confidence": 0.5}},
            "filters.min_confidence",
        ),
        (
            {"intent": "count", "target": "tracks", "group_by": ["event_type"]},
            "group_by 'event_type'",
        ),
        ({"intent": "count", "target": "detections", "group_by": ["zone"]}, "group_by 'zone'"),
        (
            {"intent": "count", "target": "frames", "group_by": ["class_name"]},
            "group_by 'class_name'",
        ),
    ],
)
def test_combination_rules(fields: dict[str, Any], fragment: str) -> None:
    assert fragment in errors_of(**fields)


def test_all_combination_errors_are_reported_together() -> None:
    message = errors_of(
        intent="list", target="tracks", bucket="day", visual_query="x", group_by=["camera"]
    )
    assert "bucket is only allowed" in message
    assert "visual_query is only allowed" in message
    assert "group_by is not allowed" in message


def test_trend_may_group_into_several_series() -> None:
    parsed = plan(intent="trend", target="tracks", bucket="hour", group_by=["camera"])
    assert result_kind(parsed) is ResultKind.TIME_SERIES


# --- Result kinds and JSON schema ------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "kind"),
    [
        ("count", ResultKind.SCALAR),
        ("count_grouped", ResultKind.TABLE),
        ("aggregate", ResultKind.TABLE),
        ("list", ResultKind.TABLE),
        ("trend", ResultKind.TIME_SERIES),
        ("top_k", ResultKind.TABLE),
        ("dwell", ResultKind.SCALAR),
        ("frames_hybrid", ResultKind.FRAMES),
        ("abstain", ResultKind.ABSTENTION),
    ],
)
def test_result_kind(name: str, kind: ResultKind) -> None:
    assert result_kind(plan(**VALID[name])) is kind


def test_json_schema_is_serializable_and_closed() -> None:
    schema = plan_json_schema()
    json.dumps(schema)
    assert schema["additionalProperties"] is False
    assert set(schema["properties"]["intent"]["$ref"].split("/")[-1:]) == {"Intent"}
    assert set(schema["$defs"]["Intent"]["enum"]) == {i.value for i in Intent}
    time_range = schema["$defs"]["Filters"]["properties"]["time_range"]
    assert "discriminator" in json.dumps(time_range)
