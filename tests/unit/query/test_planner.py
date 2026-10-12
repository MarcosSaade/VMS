"""Planner with a scripted fake LLM: prompt contents, parsing, semantic validation, one repair,
abstention."""

from __future__ import annotations

import asyncio
import datetime as dt
from typing import Any

import pytest

from tests.unit.query.conftest import REPO_ROOT
from tests.unit.query.fakes import ScriptedLlm
from vms.common.clock import FakeClock
from vms.query.catalog import QueryCatalog
from vms.query.fewshots import load_fewshots
from vms.query.llm import LlmError
from vms.query.plan import AbstainReason, Intent, QueryPlan
from vms.query.planner import (
    Planner,
    PlannerUnavailableError,
    PlanningFailedError,
    PlanResult,
    abstention,
    parse_plan,
)
from vms.query.prompts import PLANNER_FIELDS, PromptTemplate
from vms.query.semantic import validate_plan
from vms.query.settings import PlannerSettings, QuerySettings

NOW = dt.datetime(2026, 10, 12, 13, 30, tzinfo=dt.UTC)
GOOD = {"intent": "count", "target": "tracks", "filters": {"cameras": ["entrance"]}}


def make_planner(llm: ScriptedLlm, settings: QuerySettings, repair_attempts: int = 1) -> Planner:
    return Planner(
        llm,
        PromptTemplate.load(REPO_ROOT / settings.assets.planner_prompt, PLANNER_FIELDS),
        load_fewshots(REPO_ROOT / settings.assets.fewshots),
        planner=PlannerSettings(repair_attempts=repair_attempts),
        sql=settings.sql,
        clock=FakeClock(NOW),
    )


def plan(planner: Planner, catalog: QueryCatalog, question: str = "q?") -> PlanResult:
    return asyncio.run(planner.plan(question, catalog))


def test_valid_plan_first_try(query_settings: QuerySettings, catalog: QueryCatalog) -> None:
    llm = ScriptedLlm(GOOD, tokens=7)
    result = plan(make_planner(llm, query_settings), catalog, "How many at the entrance?")
    assert result.plan == QueryPlan.model_validate(GOOD)
    assert not result.repaired
    assert result.tokens == 14
    assert len(llm.calls) == 1
    system, user = llm.calls[0]
    assert (system.role, user.role, user.content) == ("system", "user", "How many at the entrance?")
    assert llm.schemas[0]["title"] == "QueryPlan"


def test_prompt_lists_catalog_time_schema_and_examples(
    query_settings: QuerySettings, catalog: QueryCatalog
) -> None:
    llm = ScriptedLlm(GOOD)
    plan(make_planner(llm, query_settings), catalog)
    prompt = llm.calls[0][0].content
    assert "- checkout (Europe/London); zones: register_1, register_2" in prompt
    assert "- loading_dock (Asia/Tokyo); zones: none" in prompt
    assert "person, bicycle, car, truck, dog" in prompt
    # The current time in each camera timezone, from the injected clock.
    assert "2026-10-12 22:30 Monday (Asia/Tokyo)" in prompt
    assert "2026-10-12 14:30 Monday (Europe/London)" in prompt
    assert "- tracks: track_id, camera_id, class_name" in prompt
    assert '"QueryPlan"' in prompt  # the JSON schema
    assert "Q: How many people were seen at the entrance today?" in prompt
    assert "$" not in prompt.replace("$defs", "").replace("$ref", "")


def test_code_fences_are_stripped() -> None:
    parsed, errors = parse_plan('```json\n{"intent": "count", "target": "events"}\n```')
    assert errors == []
    assert parsed is not None
    assert parsed.intent is Intent.COUNT


@pytest.mark.parametrize(
    ("raw", "fragment"),
    [
        ("not json", "not valid JSON"),
        ("[1, 2]", "one JSON object"),
        ('{"intent": "trend", "target": "tracks"}', "requires bucket"),
        ('{"intent": "count", "target": "tracks", "colour": "red"}', "colour: Extra inputs"),
        ('{"intent": "count", "target": "planets"}', "target:"),
    ],
)
def test_parse_errors_are_readable(raw: str, fragment: str) -> None:
    parsed, errors = parse_plan(raw)
    assert parsed is None
    assert any(fragment in e for e in errors), errors


def test_invalid_json_is_repaired(query_settings: QuerySettings, catalog: QueryCatalog) -> None:
    llm = ScriptedLlm("{not json", GOOD)
    result = plan(make_planner(llm, query_settings), catalog)
    assert result.repaired
    assert result.plan == QueryPlan.model_validate(GOOD)
    # The repair turn carries the bad output and the error.
    repair = llm.calls[1]
    assert [m.role for m in repair] == ["system", "user", "assistant", "user"]
    assert repair[2].content == "{not json"
    assert "not valid JSON" in repair[3].content


def test_schema_errors_are_all_sent_back(
    query_settings: QuerySettings, catalog: QueryCatalog
) -> None:
    bad = {"intent": "trend", "target": "tracks", "measure": {"op": "avg", "field": "confidence"}}
    llm = ScriptedLlm(bad, GOOD)
    plan(make_planner(llm, query_settings), catalog)
    feedback = llm.calls[1][3].content
    assert "requires bucket" in feedback
    assert "measure is only allowed" in feedback


def test_semantic_errors_list_known_names(
    query_settings: QuerySettings, catalog: QueryCatalog
) -> None:
    bad = {"intent": "count", "target": "tracks", "filters": {"cameras": ["lobby"]}}
    llm = ScriptedLlm(bad, GOOD)
    result = plan(make_planner(llm, query_settings), catalog)
    assert result.attempts[0].errors == [
        "unknown camera(s) 'lobby'; known: checkout, entrance, loading_dock, parking_lot"
    ]
    assert "known: checkout, entrance" in llm.calls[1][3].content


def test_only_one_repair(query_settings: QuerySettings, catalog: QueryCatalog) -> None:
    bad = {"intent": "count", "target": "tracks", "filters": {"classes": ["unicorn"]}}
    llm = ScriptedLlm(bad, bad, GOOD)
    with pytest.raises(PlanningFailedError) as excinfo:
        plan(make_planner(llm, query_settings), catalog)
    assert len(llm.calls) == 2
    assert "unknown class(es) 'unicorn'" in excinfo.value.errors[0]
    assert len(excinfo.value.attempts) == 2


def test_repair_can_be_disabled(query_settings: QuerySettings, catalog: QueryCatalog) -> None:
    llm = ScriptedLlm("nope", GOOD)
    with pytest.raises(PlanningFailedError):
        plan(make_planner(llm, query_settings, repair_attempts=0), catalog)
    assert len(llm.calls) == 1


def test_llm_failure_is_unavailable(query_settings: QuerySettings, catalog: QueryCatalog) -> None:
    llm = ScriptedLlm(LlmError("connection refused"))
    with pytest.raises(PlannerUnavailableError, match="connection refused"):
        plan(make_planner(llm, query_settings), catalog)


@pytest.mark.parametrize("reason", list(AbstainReason))
def test_abstention_is_a_valid_outcome(
    query_settings: QuerySettings, catalog: QueryCatalog, reason: AbstainReason
) -> None:
    llm = ScriptedLlm({"intent": "abstain", "abstain_reason": reason.value})
    result = plan(make_planner(llm, query_settings), catalog)
    found = abstention(result.plan)
    assert found is not None
    assert found.reason is reason
    assert found.message


def test_visual_query_abstains_as_unsupported() -> None:
    p = QueryPlan.model_validate(
        {"intent": "frames", "target": "frames", "visual_query": "a red jacket"}
    )
    found = abstention(p)
    assert found is not None
    assert found.reason is AbstainReason.UNSUPPORTED
    assert "CLIP" in found.message
    assert abstention(QueryPlan.model_validate(GOOD)) is None


# --- semantic validation ----------------------------------------------------------------------


def semantic(catalog: QueryCatalog, raw: dict[str, Any], max_limit: int = 1000) -> list[str]:
    return validate_plan(
        QueryPlan.model_validate(raw), catalog, NOW, PlannerSettings(), max_limit=max_limit
    )


def test_semantic_accepts_a_good_plan(catalog: QueryCatalog) -> None:
    raw = {
        "intent": "count",
        "target": "events",
        "filters": {"cameras": ["checkout"], "zones": ["register_1"], "classes": ["person"]},
    }
    assert semantic(catalog, raw) == []


def test_semantic_reports_every_problem(catalog: QueryCatalog) -> None:
    raw = {
        "intent": "list",
        "target": "events",
        "limit": 900,
        "filters": {
            "cameras": ["garage"],
            "zones": ["vault"],
            "classes": ["dragon"],
            "time_range": {"kind": "last", "amount": 500, "unit": "days"},
        },
    }
    errors = semantic(catalog, raw, max_limit=100)
    assert len(errors) == 5
    assert errors[0].startswith("unknown camera(s) 'garage'")
    assert errors[1].startswith("unknown zone(s) 'vault'; known: front_door, register_1")
    assert errors[2].startswith("unknown class(es) 'dragon'")
    assert errors[3] == "time range longer than 400 days"
    assert errors[4] == "limit 900 exceeds the maximum of 100 rows"


def test_zone_must_be_on_the_plans_cameras(catalog: QueryCatalog) -> None:
    raw = {
        "intent": "count",
        "target": "events",
        "filters": {"cameras": ["entrance"], "zones": ["register_1"]},
    }
    assert semantic(catalog, raw) == [
        "zone 'register_1' is on camera 'checkout', not on the cameras in filters.cameras"
    ]


def test_absolute_range_in_the_future_is_rejected(catalog: QueryCatalog) -> None:
    future = {"kind": "absolute", "start": "2027-03-01T09:00:00", "end": "2027-03-01T10:00:00"}
    errors = semantic(
        catalog, {"intent": "count", "target": "events", "filters": {"time_range": future}}
    )
    assert len(errors) == 1
    assert "starts in the future" in errors[0]
    later_today = {"kind": "absolute", "start": "2026-10-12T20:00:00", "end": "2026-10-12T21:00:00"}
    raw = {"intent": "count", "target": "events", "filters": {"time_range": later_today}}
    assert semantic(catalog, raw) == []
