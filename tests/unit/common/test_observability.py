import io
import json
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from vms.common import ServiceSettings, init_observability, load_settings

pytestmark = pytest.mark.usefixtures("restore_root_logger")


def test_three_line_setup_logs_traces_and_exposes_metrics(tmp_path: Path) -> None:
    (tmp_path / "ingest.yaml").write_text("telemetry:\n  enabled: false\n", encoding="utf-8")
    logs, spans = io.StringIO(), InMemorySpanExporter()

    # The three lines a service needs (test-only kwargs: config dir, exporter, log capture).
    settings = load_settings(ServiceSettings, "ingest", config_dir=tmp_path)
    obs = init_observability(
        settings, span_exporter=spans, set_global_tracer=False, log_stream=logs
    )
    metrics_app = obs.metrics_app()

    frames = obs.metrics.counter("received_frames", "total", "Frames received")
    with obs.tracer.start_as_current_span("receive_frame") as span:
        obs.logger.info("frame received", extra={"camera_id": "cam-01"})
        frames.inc()
        trace_id = format(span.get_span_context().trace_id, "032x")
    obs.shutdown()

    [entry] = [json.loads(line) for line in logs.getvalue().splitlines()]
    assert entry["service"] == "ingest"
    assert entry["trace_id"] == trace_id
    assert entry["camera_id"] == "cam-01"
    [finished] = spans.get_finished_spans()
    assert finished.name == "receive_frame"
    assert finished.resource.attributes["service.name"] == "ingest"
    assert "vms_ingest_received_frames_total 1.0" in obs.metrics.exposition().decode()
    assert callable(metrics_app)


def test_service_name_with_dash_maps_to_metric_component(tmp_path: Path) -> None:
    (tmp_path / "query-service.yaml").write_text("", encoding="utf-8")
    settings = load_settings(ServiceSettings, "query-service", config_dir=tmp_path)
    obs = init_observability(settings, set_global_tracer=False, log_stream=io.StringIO())
    obs.metrics.counter("requests", "total", "x")
    assert b"vms_query_service_requests_total" in obs.metrics.exposition()
