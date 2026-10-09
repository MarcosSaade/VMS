"""The HTTP layer through FastAPI's in-process test client (no sockets)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.unit.ingest.conftest import JPEG, NOW, FakeFrameRepository, FakeObjectStore, count
from vms.common.ingest_api import FRAMES_PATH
from vms.common.metrics import Metrics
from vms.ingest.app import create_app
from vms.ingest.service import FrameIngestor

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
