"""The plan engine end to end with fakes: each outcome's status, error type, what runs, and what
is written to query_log. Plus the HTTP layer through FastAPI's in-process client."""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry import trace

from tests.unit.query.conftest import REPO_ROOT
from tests.unit.query.fakes import FakeFrames, FakeQueryLog, FakeReader, ScriptedLlm
from vms.common.clock import FakeClock
from vms.common.ids import UlidGenerator
from vms.common.metrics import Metrics
from vms.query.app import create_app
from vms.query.engine import CatalogCache, ErrorType, PlanEngine, QueryResponse, QueryStatus
from vms.query.fewshots import load_fewshots
from vms.query.llm import LlmError
from vms.query.planner import Planner
from vms.query.prompts import PLANNER_FIELDS, PromptTemplate
from vms.query.settings import PlannerSettings, QuerySettings, SqlValidatorSettings
from vms.query.sql_validator import SqlValidator
from vms.query.store import ExecutionError, ExecutionTimeoutError, QueryRows

NOW = dt.datetime(2026, 10, 12, 13, 30, tzinfo=dt.UTC)
CAMERAS = [("cam-001", "entrance", "Europe/London"), ("cam-002", "checkout", "Europe/London")]
ZONES = [("cam-001:front_door", "cam-001", "front_door")]
COUNT_PEOPLE = {
    "intent": "count",
    "target": "tracks",
    "filters": {
        "cameras": ["entrance"],
        "classes": ["person"],
        "time_range": {"kind": "named", "name": "today"},
    },
}
FRAME_ID = "01JABCDEFGHJKMNPQRSTVWXYZ0"


class Harness:
    def __init__(
        self,
        settings: QuerySettings,
        llm: ScriptedLlm,
        results: list[QueryRows | Exception] | None = None,
        *,
        log_fails: bool = False,
        sql: SqlValidatorSettings | None = None,
    ) -> None:
        self.clock = FakeClock(NOW)
        self.llm = llm
        self.reader = FakeReader(CAMERAS, ZONES, results)
        self.log = FakeQueryLog(fail=log_fails)
        self.metrics = Metrics("query")
        planner = Planner(
            llm,
            PromptTemplate.load(REPO_ROOT / settings.assets.planner_prompt, PLANNER_FIELDS),
            load_fewshots(REPO_ROOT / settings.assets.fewshots),
            planner=PlannerSettings(),
            sql=settings.sql,
            clock=self.clock,
        )
        self.catalog = CatalogCache(self.reader, ["person", "car"], self.clock, ttl_s=30)
        self.engine = PlanEngine(
            planner,
            self.catalog,
            self.reader,
            self.log,
            SqlValidator(sql or settings.sql),
            default_limit=100,
            max_limit=1000,
            clock=self.clock,
            ids=UlidGenerator(self.clock),
            metrics=self.metrics,
            tracer=trace.NoOpTracer(),
            logger=logging.getLogger("test"),
        )

    def ask(self, question: str = "How many people at the entrance today?") -> QueryResponse:
        return asyncio.run(self.engine.answer(question))

    def metric(self, name: str, **labels: str) -> float:
        value = self.metrics.registry.get_sample_value(name, labels)
        return 0.0 if value is None else value


def test_answer_with_rows_plan_sql_and_log(query_settings: QuerySettings) -> None:
    h = Harness(
        query_settings, ScriptedLlm(COUNT_PEOPLE, tokens=5), [QueryRows(["n"], [(7,)], False)]
    )
    r = h.ask()
    assert r.status is QueryStatus.OK
    assert r.http_status == 200
    assert r.answer == "7 people."
    assert (r.columns, r.rows, r.row_count, r.truncated) == (["n"], [[7]], 1, False)
    assert r.result_kind == "scalar"
    assert r.plan == {**COUNT_PEOPLE}
    assert r.llm == "scripted"
    assert r.tokens == 10
    # The executed SQL is the validator's regenerated text; values only in params.
    sql, params, max_rows = h.reader.executed[0]
    assert sql == r.compiled_sql
    assert sql.startswith("SELECT COUNT(*) AS n FROM tracks AS t")
    for value in ("entrance", "cam-001", "person"):
        assert value not in sql
    assert set(params.values()) >= {"cam-001", "person"}
    assert r.params["p0"] == "2026-10-11T23:00:00+00:00"  # JSON-friendly copy
    assert max_rows == 1
    assert set(r.timings_ms) == {"catalog", "plan", "compile", "validate", "execute", "total"}
    # Logged once, successful.
    (entry,) = h.log.entries
    assert entry.query_id == r.query_id
    assert entry.success is True
    assert entry.error_type is None
    assert entry.compiled_sql == r.compiled_sql
    assert entry.query_plan == r.plan
    assert entry.row_count == 1
    assert entry.tokens == 10
    assert entry.llm == "scripted"
    assert entry.timings == r.timings_ms
    assert h.metric("vms_query_requests_total", status="ok", error_type="") == 1


def test_abstention_runs_no_sql(query_settings: QuerySettings) -> None:
    h = Harness(
        query_settings, ScriptedLlm({"intent": "abstain", "abstain_reason": "unanswerable"})
    )
    r = h.ask("What was the license plate?")
    assert r.status is QueryStatus.ABSTAINED
    assert r.http_status == 200
    assert r.abstain_reason == "unanswerable"
    assert r.result_kind == "abstention"
    assert "plates" in r.answer
    assert h.reader.executed == []
    assert r.compiled_sql is None
    assert h.log.entries[0].success is True
    assert h.log.entries[0].row_count is None


def test_visual_query_is_unsupported(query_settings: QuerySettings) -> None:
    plan = {"intent": "frames", "target": "frames", "visual_query": "a red jacket"}
    h = Harness(query_settings, ScriptedLlm(plan))
    r = h.ask("Find someone in a red jacket")
    assert r.status is QueryStatus.ABSTAINED
    assert r.abstain_reason == "unsupported"
    assert "CLIP" in r.answer
    assert h.reader.executed == []


def test_invalid_plan_after_repair(query_settings: QuerySettings) -> None:
    bad = {"intent": "count", "target": "tracks", "filters": {"cameras": ["garage"]}}
    h = Harness(query_settings, ScriptedLlm(bad, bad))
    r = h.ask()
    assert r.status is QueryStatus.ERROR
    assert r.http_status == 422
    assert r.error is not None
    assert r.error.type is ErrorType.INVALID_PLAN
    assert "unknown camera(s) 'garage'" in r.error.errors[0]
    assert r.repaired is True
    assert h.reader.executed == []
    assert h.log.entries[0].error_type == "invalid_plan"
    assert h.log.entries[0].success is False
    assert h.metric("vms_query_plan_repairs_total") == 1


def test_repaired_plan_succeeds(query_settings: QuerySettings) -> None:
    h = Harness(
        query_settings, ScriptedLlm("oops", COUNT_PEOPLE), [QueryRows(["n"], [(2,)], False)]
    )
    r = h.ask()
    assert r.status is QueryStatus.OK
    assert r.repaired is True
    assert r.tokens == 40


def test_planner_unavailable(query_settings: QuerySettings) -> None:
    h = Harness(query_settings, ScriptedLlm(LlmError("connection refused")))
    r = h.ask()
    assert r.http_status == 503
    assert r.error is not None
    assert r.error.type is ErrorType.PLANNER_UNAVAILABLE
    assert h.log.entries[0].query_plan is None


@pytest.mark.parametrize(
    ("exc", "kind", "status"),
    [
        (ExecutionTimeoutError("cancelled after 5000 ms"), ErrorType.TIMEOUT, 504),
        (ExecutionError("relation does not exist"), ErrorType.EXECUTION_ERROR, 500),
    ],
)
def test_execution_failures(
    query_settings: QuerySettings, exc: Exception, kind: ErrorType, status: int
) -> None:
    h = Harness(query_settings, ScriptedLlm(COUNT_PEOPLE), [exc])
    r = h.ask()
    assert r.http_status == status
    assert r.error is not None
    assert r.error.type is kind
    assert r.compiled_sql is not None  # shown so the failure can be debugged
    assert h.log.entries[0].error_type == kind.value


def test_validator_rejection_blocks_execution(query_settings: QuerySettings) -> None:
    # A validator whose allowlist lacks `tracks` stands in for a compiler bug.
    narrow = query_settings.sql.model_copy(
        update={"tables": {k: v for k, v in query_settings.sql.tables.items() if k != "tracks"}}
    )
    h = Harness(query_settings, ScriptedLlm(COUNT_PEOPLE), sql=narrow)
    r = h.ask()
    assert r.error is not None
    assert r.error.type is ErrorType.SQL_REJECTED
    assert "forbidden_table" in r.error.detail
    assert h.reader.executed == []


def test_query_log_failure_does_not_fail_the_answer(query_settings: QuerySettings) -> None:
    h = Harness(query_settings, ScriptedLlm(COUNT_PEOPLE), log_fails=True)
    r = h.ask()
    assert r.status is QueryStatus.OK
    assert h.metric("vms_query_query_log_failures_total") == 1


def test_frames_get_image_urls_and_truncation(query_settings: QuerySettings) -> None:
    rows = [(f"{FRAME_ID[:-1]}{i}", NOW, "entrance", f"k{i}", 1280, 720) for i in range(3)]
    columns = ["frame_id", "ts", "camera", "object_key", "width", "height"]
    plan = {"intent": "frames", "target": "frames", "filters": {"classes": ["car"]}}
    settings = query_settings.model_copy()
    h = Harness(settings, ScriptedLlm(plan), [QueryRows(columns, rows, False)])
    h.engine._default_limit = 3  # the result fills the default limit: there may be more
    r = h.ask("Show frames with cars")
    assert r.columns[-1] == "image_url"
    assert r.rows[0][-1] == f"/v1/frames/{FRAME_ID[:-1]}0/image"
    assert r.rows[0][1] == NOW.isoformat()
    assert r.truncated is True
    assert r.answer == "3 frames (showing the first 3)."


def test_catalog_is_cached_for_its_ttl(query_settings: QuerySettings) -> None:
    h = Harness(query_settings, ScriptedLlm(COUNT_PEOPLE, COUNT_PEOPLE, COUNT_PEOPLE))
    h.ask()
    h.ask()
    assert h.reader.catalog_loads == 1
    h.clock.advance(31)
    h.ask()
    assert h.reader.catalog_loads == 2


# --- HTTP ------------------------------------------------------------------------------------


@pytest.fixture
def harness(query_settings: QuerySettings) -> Harness:
    return Harness(query_settings, ScriptedLlm(COUNT_PEOPLE), [QueryRows(["n"], [(3,)], False)])


@pytest.fixture
def client(harness: Harness) -> Iterator[TestClient]:
    frames = FakeFrames({"frames/cam-001/2026-10-12/x.jpg": b"\xff\xd8jpeg"})
    harness.reader.fixed["FROM frames WHERE frame_id"] = QueryRows(
        ["object_key"], [("frames/cam-001/2026-10-12/x.jpg",)], False
    )
    app = create_app(
        harness.engine,
        harness.catalog,
        harness.reader,
        frames,
        harness.metrics,
        ready_checks={"postgres": harness.reader.ready, "object_store": frames.ready},
    )
    with TestClient(app) as c:
        yield c


def test_post_query(client: TestClient) -> None:
    response = client.post("/v1/query", json={"question": "  How many   people?  "})
    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    assert body["question"] == "How many people?"
    assert body["answer"] == "3 people."
    assert body["status"] == "ok"


@pytest.mark.parametrize(
    "payload", [{}, {"question": ""}, {"question": "   "}, {"question": "x" * 1001}, {"q": "a"}]
)
def test_bad_requests(client: TestClient, payload: dict[str, Any]) -> None:
    assert client.post("/v1/query", json=payload).status_code == 422


def test_error_status_and_body_shape(query_settings: QuerySettings) -> None:
    h = Harness(query_settings, ScriptedLlm(LlmError("down")))
    app = create_app(h.engine, h.catalog, h.reader, FakeFrames(), h.metrics, ready_checks={})
    with TestClient(app) as c:
        response = c.post("/v1/query", json={"question": "anything"})
    assert response.status_code == 503
    assert response.json()["error"]["type"] == "planner_unavailable"


def test_catalog_endpoint(client: TestClient) -> None:
    body = client.get("/v1/catalog").json()
    assert body["cameras"][0] == {
        "camera_id": "cam-001",
        "name": "entrance",
        "timezone": "Europe/London",
        "zones": ["front_door"],
    }
    assert body["classes"] == ["person", "car"]


def test_frame_image(client: TestClient, harness: Harness) -> None:
    response = client.get(f"/v1/frames/{FRAME_ID}/image")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.content == b"\xff\xd8jpeg"
    assert harness.reader.fetched[-1][1] == {"frame_id": FRAME_ID}


def test_frame_image_rejects_bad_ids_and_unknown_frames(
    client: TestClient, harness: Harness
) -> None:
    assert client.get("/v1/frames/../etc/image").status_code == 404
    assert client.get("/v1/frames/not-a-ulid/image").status_code == 422
    harness.reader.fixed.clear()
    assert client.get(f"/v1/frames/{FRAME_ID}/image").status_code == 404


def test_probes_and_metrics(client: TestClient, harness: Harness) -> None:
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/readyz").status_code == 200
    harness.reader.is_ready = False
    assert client.get("/readyz").status_code == 503
    client.post("/v1/query", json={"question": "q"})
    text = client.get("/metrics").text
    assert "vms_query_requests_total" in text
    assert "vms_query_stage_duration_seconds" in text
    assert 'route="/v1/query"' in text


# --- camera wall, examples, UI -----------------------------------------------------------------


def test_live_view(harness: Harness, client: TestClient) -> None:
    ts = dt.datetime(2026, 10, 12, 13, 29, tzinfo=dt.UTC)
    harness.reader.fixed["model_version IS NOT NULL"] = QueryRows(
        ["frame_id", "ts", "width", "height"], [(FRAME_ID, ts, 1280, 720)], False
    )
    harness.reader.fixed["FROM detections"] = QueryRows(
        [], [(FRAME_ID, "person", 0.91, "cam-001:a:7", 10.0, 20.0, 110.0, 220.0)], False
    )
    harness.reader.fixed["FROM zones"] = QueryRows(
        [], [("cam-001", "front_door", [[0.1, 0.5], [0.9, 0.5], [0.5, 1.0]])], False
    )
    cameras = client.get("/v1/live").json()["cameras"]
    assert [c["name"] for c in cameras] == ["entrance", "checkout"]
    entrance = cameras[0]
    assert entrance["zones"] == [
        {"name": "front_door", "polygon": [[0.1, 0.5], [0.9, 0.5], [0.5, 1.0]]}
    ]
    frame = entrance["frame"]
    assert frame["image_url"] == f"/v1/frames/{FRAME_ID}/image"
    assert (frame["width"], frame["height"]) == (1280, 720)
    assert frame["detections"] == [
        {
            "class_name": "person",
            "confidence": 0.91,
            "track_id": "cam-001:a:7",
            "box": [10.0, 20.0, 110.0, 220.0],
        }
    ]
    # Camera ids are bound parameters of the fixed query, one probe per camera.
    probes = [p for sql, p in harness.reader.fetched if "model_version IS NOT NULL" in sql]
    assert probes == [{"camera_id": "cam-001"}, {"camera_id": "cam-002"}]


def test_examples_endpoint(harness: Harness) -> None:
    app = create_app(
        harness.engine,
        harness.catalog,
        harness.reader,
        FakeFrames(),
        harness.metrics,
        ready_checks={},
        examples=["How many cars?", "Show the last 10 zone events."],
    )
    with TestClient(app) as c:
        assert c.get("/v1/examples").json() == {
            "questions": ["How many cars?", "Show the last 10 zone events."]
        }


def test_ui_is_served_and_api_routes_win(harness: Harness, tmp_path: Path) -> None:
    (tmp_path / "index.html").write_text("<!doctype html><title>VMS</title>")
    app = create_app(
        harness.engine,
        harness.catalog,
        harness.reader,
        FakeFrames(),
        harness.metrics,
        ready_checks={},
        ui_dir=tmp_path,
    )
    with TestClient(app) as c:
        assert "<title>VMS</title>" in c.get("/").text
        assert c.get("/healthz").json() == {"status": "ok"}
        assert c.get("/v1/examples").status_code == 200


def test_no_ui_dir_means_no_root_page(harness: Harness, tmp_path: Path) -> None:
    app = create_app(
        harness.engine,
        harness.catalog,
        harness.reader,
        FakeFrames(),
        harness.metrics,
        ready_checks={},
        ui_dir=tmp_path / "missing",
    )
    with TestClient(app) as c:
        assert c.get("/").status_code == 404
