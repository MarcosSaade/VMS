"""The HTTP layer through FastAPI's in-process test client (no sockets)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from tests.unit.ingest.conftest import JPEG, NOW, FakeFrameRepository, FakeObjectStore, count
from vms.common.clock import FakeClock
from vms.common.ids import UlidGenerator
from vms.common.ingest_api import FRAMES_PATH
from vms.common.metrics import Metrics
from vms.common.settings import TelemetrySettings
from vms.common.telemetry import init_telemetry
from vms.ingest.app import create_app
from vms.ingest.service import FrameIngestor
from vms.ingest.settings import FrameLimits

NOW_MS = int(NOW.timestamp() * 1000)


@pytest.fixture
def client(
    ingestor: FrameIngestor, metrics: Metrics, store: FakeObjectStore, repo: FakeFrameRepository
) -> Iterator[TestClient]:
    app = create_app(
        ingestor,
        metrics,
        max_jpeg_bytes=1000,
        ready_checks={"postgres": repo.ready, "object_store": store.ready},
    )
    with TestClient(app) as c:
        yield c


def fields(**overrides: Any) -> dict[str, str]:
    values = {"camera_id": "cam-001", "run_id": "run-1", "seq": "0", "ts_unix_ms": str(NOW_MS)}
    values.update({k: str(v) for k, v in overrides.items()})
    return values


def post(client: TestClient, data: dict[str, str], jpeg: bytes | None = JPEG) -> Any:
    files = {"jpeg": ("0.jpg", jpeg, "image/jpeg")} if jpeg is not None else None
    return client.post(FRAMES_PATH, data=data, files=files)


def test_created_then_duplicate(client: TestClient, repo: FakeFrameRepository) -> None:
    first = post(client, fields())
    assert first.status_code == 201
    assert first.json()["duplicate"] is False
    second = post(client, fields())
    assert second.status_code == 200
    assert second.json() == {"frame_id": first.json()["frame_id"], "duplicate": True}
    assert len(repo.rows) == 1


@pytest.mark.parametrize(
    "bad",
    [
        {"camera_id": "../etc"},  # path traversal into the object key
        {"camera_id": "cam/001"},
        {"camera_id": ".hidden"},
        {"camera_id": ""},
        {"camera_id": "c" * 65},
        {"run_id": "run 1"},
        {"seq": "-1"},
        {"seq": "abc"},
        {"ts_unix_ms": "soon"},
    ],
)
def test_invalid_fields_are_422_and_nothing_is_stored(
    client: TestClient, store: FakeObjectStore, metrics: Metrics, bad: dict[str, str]
) -> None:
    response = post(client, fields(**bad))
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_fields"
    assert store.objects == {}
    assert count(metrics, "invalid") == 1


def test_missing_fields_and_file(client: TestClient) -> None:
    data = fields()
    del data["run_id"]
    assert post(client, data).status_code == 422
    assert post(client, fields(), jpeg=None).json()["error"] == "missing_jpeg"


def test_unexpected_extra_fields_are_rejected_by_the_parser(client: TestClient) -> None:
    response = post(client, {**fields(), "extra": "x"})
    assert response.status_code == 400
    assert response.json()["error"] == "bad_multipart"


def test_not_a_jpeg_is_415(client: TestClient) -> None:
    assert post(client, fields(), jpeg=b"GIF89a....").status_code == 415


def test_body_over_the_limit_is_refused_before_parsing(
    client: TestClient, store: FakeObjectStore
) -> None:
    response = post(client, fields(), jpeg=b"\xff\xd8\xff" + b"\x00" * 50_000)
    assert response.status_code == 413
    assert response.json()["error"] == "body_too_large"
    assert store.objects == {}


def test_missing_content_length_is_411(client: TestClient) -> None:
    def chunks() -> Iterator[bytes]:
        yield b"--x\r\n"

    response = client.post(
        FRAMES_PATH, content=chunks(), headers={"content-type": "multipart/form-data; boundary=x"}
    )
    assert response.status_code == 411


def test_storage_failure_is_500(client: TestClient, repo: FakeFrameRepository) -> None:
    repo.fail_insert = True
    response = post(client, fields())
    assert response.status_code == 500
    assert response.json()["error"] == "database_unavailable"


def test_health_readiness_and_metrics(
    client: TestClient, store: FakeObjectStore, repo: FakeFrameRepository
) -> None:
    assert client.get("/healthz").status_code == 200
    assert client.get("/readyz").json() == {"postgres": "ok", "object_store": "ok"}
    store.is_ready = False
    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["object_store"] == "unavailable"
    post(client, fields())
    text = client.get("/metrics").text
    assert 'vms_ingest_frames_total{outcome="created"} 1.0' in text
    assert 'vms_ingest_request_duration_seconds_count{route="/v1/frames",status="201"} 1.0' in text


@pytest.fixture
def traced() -> tuple[InMemorySpanExporter, Any]:
    spans = InMemorySpanExporter()
    tracer = init_telemetry("ingest", TelemetrySettings(), exporter=spans, set_global=False)
    return spans, tracer.get_tracer("test")


def traced_client(
    store: FakeObjectStore, repo: FakeFrameRepository, tracer: Any, metrics: Metrics
) -> TestClient:
    clock = FakeClock(NOW)
    counter = iter(range(1, 1_000_000))
    ids = UlidGenerator(clock, randbytes=lambda n: next(counter).to_bytes(n, "big"))
    ingestor = FrameIngestor(
        store, repo, FrameLimits(max_jpeg_bytes=1000), metrics, clock=clock, ids=ids, tracer=tracer
    )
    return TestClient(create_app(ingestor, metrics, max_jpeg_bytes=1000, ready_checks={}))


SENDER_TRACE = "4bf92f3577b34da6a3ce929d0e0e4736"
SENDER_SPAN = "00f067aa0ba902b7"


def test_frame_trace_continues_the_sender_and_is_stored_for_the_worker(
    store: FakeObjectStore, repo: FakeFrameRepository, traced: tuple[InMemorySpanExporter, Any]
) -> None:
    """Task 2.1 (D25): loadgen's traceparent header -> ingest.frame span -> frames.traceparent."""
    spans, tracer = traced
    metrics = Metrics("ingest")
    with traced_client(store, repo, tracer, metrics) as client:
        response = client.post(
            FRAMES_PATH,
            data=fields(),
            files={"jpeg": ("0.jpg", JPEG, "image/jpeg")},
            headers={"traceparent": f"00-{SENDER_TRACE}-{SENDER_SPAN}-01"},
        )
    assert response.status_code == 201

    by_name = {s.name: s for s in spans.get_finished_spans()}
    frame = by_name["ingest.frame"]
    assert frame.context is not None
    assert frame.parent is not None
    assert format(frame.context.trace_id, "032x") == SENDER_TRACE
    assert format(frame.parent.span_id, "016x") == SENDER_SPAN
    assert frame.attributes is not None
    assert frame.attributes["outcome"] == "created"
    for stage in ("object_put", "db_insert"):
        child = by_name[f"ingest.{stage}"]
        assert child.parent is not None
        assert child.parent.span_id == frame.context.span_id
    # The row carries the ingest.frame span (not the sender's): the worker hangs off it.
    row = repo.rows[("cam-001", "run-1", 0)]
    assert row.traceparent == f"00-{SENDER_TRACE}-{frame.context.span_id:016x}-01"
    # Multipart parsing is timed and traced too (a measured cost, E-010).
    parse = by_name["ingest.parse"]
    assert parse.parent is not None
    assert format(parse.parent.span_id, "016x") == SENDER_SPAN
    for stage in ("parse", "object_put", "db_insert"):
        value = metrics.registry.get_sample_value(
            "vms_ingest_stage_duration_seconds_count", {"stage": stage}
        )
        assert value == 1, stage


def test_unsampled_sender_decision_is_stored_and_nothing_is_recorded(
    store: FakeObjectStore, repo: FakeFrameRepository, traced: tuple[InMemorySpanExporter, Any]
) -> None:
    spans, tracer = traced
    with traced_client(store, repo, tracer, Metrics("ingest")) as client:
        client.post(
            FRAMES_PATH,
            data=fields(),
            files={"jpeg": ("0.jpg", JPEG, "image/jpeg")},
            headers={"traceparent": f"00-{SENDER_TRACE}-{SENDER_SPAN}-00"},
        )
    assert spans.get_finished_spans() == ()
    # The "not sampled" flag travels on, so the worker does not sample the frame on its own.
    stored = repo.rows[("cam-001", "run-1", 0)].traceparent
    assert stored is not None
    assert stored.startswith(f"00-{SENDER_TRACE}-")
    assert stored.endswith("-00")


def test_frame_without_trace_context_stores_none(
    repo: FakeFrameRepository, client: TestClient
) -> None:
    # Telemetry off (the global no-op tracer) and no header: nothing to continue or store.
    post(client, fields())
    assert repo.rows[("cam-001", "run-1", 0)].traceparent is None
