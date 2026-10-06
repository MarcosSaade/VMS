"""Smoke tests for the compose CPU stack: each component does its job, not just answers health.

Run with `make up && make test-integration`.
"""

from __future__ import annotations

import base64
import urllib.error
import urllib.request
import uuid
from typing import Any

import boto3
import psycopg
import pytest
from botocore.config import Config
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

from tests.integration.conftest import HTTP_TIMEOUT_S, StackSettings, get_json, poll

pytestmark = pytest.mark.integration


def test_postgres_has_pgvector(stack: StackSettings) -> None:
    with psycopg.connect(f"{stack.postgres_dsn} dbname={stack.postgres_db}") as conn:
        available = conn.execute(
            "SELECT 1 FROM pg_available_extensions WHERE name = 'vector'"
        ).fetchone()
    assert available is not None, "pgvector extension is not installable"


@pytest.fixture
def s3(stack: StackSettings) -> Any:
    return boto3.client(
        "s3",
        endpoint_url=stack.s3_endpoint_url,
        aws_access_key_id=stack.s3_access_key_id,
        aws_secret_access_key=stack.s3_secret_access_key,
        region_name="us-east-1",  # signing Region for the local endpoint only
        config=Config(connect_timeout=HTTP_TIMEOUT_S, read_timeout=HTTP_TIMEOUT_S),
    )


def test_object_store_buckets_and_round_trip(stack: StackSettings, s3: Any) -> None:
    buckets = {b["Name"] for b in s3.list_buckets()["Buckets"]}
    assert stack.s3_bucket in buckets

    key = f"it/{uuid.uuid4().hex}.bin"
    s3.put_object(Bucket=stack.s3_bucket, Key=key, Body=b"\xff\xd8frame")
    try:
        body = s3.get_object(Bucket=stack.s3_bucket, Key=key)["Body"].read()
        assert body == b"\xff\xd8frame"
    finally:
        s3.delete_object(Bucket=stack.s3_bucket, Key=key)


def test_object_store_rejects_anonymous_access(stack: StackSettings) -> None:
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(f"{stack.s3_endpoint_url}/{stack.s3_bucket}", timeout=5)
    assert excinfo.value.code == 403


def test_prometheus_scrapes_every_target(stack: StackSettings) -> None:
    def all_targets_up() -> list[dict[str, object]] | None:
        targets = get_json(stack.url(stack.prometheus_port, "/api/v1/targets?state=active"))
        active = targets["data"]["activeTargets"]
        return active if active and all(t["health"] == "up" for t in active) else None

    # A target added since the last scrape reports "unknown" for up to one interval.
    active = poll(all_targets_up, timeout_s=45)
    jobs = {t["labels"]["job"] for t in active}
    assert {"prometheus", "otel-collector", "otel-collector-otlp", "tempo"} <= jobs


def test_grafana_datasources_are_provisioned_and_healthy(stack: StackSettings) -> None:
    token = base64.b64encode(
        f"{stack.grafana_admin_user}:{stack.grafana_admin_password}".encode()
    ).decode()
    auth = {"Authorization": f"Basic {token}"}

    datasources = get_json(stack.url(stack.grafana_port, "/api/datasources"), auth)
    by_uid = {d["uid"]: d["type"] for d in datasources}
    assert by_uid.get("prometheus") == "prometheus"
    assert by_uid.get("tempo") == "tempo"

    for uid in ("prometheus", "tempo"):
        health = get_json(stack.url(stack.grafana_port, f"/api/datasources/uid/{uid}/health"), auth)
        assert health["status"] == "OK", (uid, health)


def test_span_reaches_tempo_through_collector(stack: StackSettings) -> None:
    provider = TracerProvider(resource=Resource.create({"service.name": "vms-integration-test"}))
    exporter = OTLPSpanExporter(
        endpoint=f"localhost:{stack.otel_grpc_port}", insecure=True, timeout=5
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    with provider.get_tracer(__name__).start_as_current_span("compose-smoke") as span:
        trace_id = format(span.get_span_context().trace_id, "032x")
    provider.shutdown()

    def trace_with_span() -> object:
        trace = get_json(stack.url(stack.tempo_port, f"/api/v2/traces/{trace_id}"))
        # Tempo answers 200 with an empty trace until the span is queryable.
        return trace if "compose-smoke" in str(trace) else None

    # The collector batches spans, then Tempo makes them queryable from its live store.
    poll(trace_with_span, timeout_s=30)
