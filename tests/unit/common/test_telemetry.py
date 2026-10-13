from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from vms.common.settings import TelemetrySettings
from vms.common.telemetry import (
    context_from_traceparent,
    current_traceparent,
    http_telemetry,
    init_telemetry,
    remote_parent,
    shutdown_telemetry,
)


def test_disabled_returns_global_provider_and_records_nothing() -> None:
    provider = init_telemetry("ingest", TelemetrySettings(enabled=False))
    assert provider is trace.get_tracer_provider()
    with provider.get_tracer("t").start_as_current_span("noop") as span:
        assert not span.is_recording()
    shutdown_telemetry(provider)  # must be safe on the API provider


def test_exporter_receives_nested_spans_with_service_name() -> None:
    exporter = InMemorySpanExporter()
    provider = init_telemetry("ingest", TelemetrySettings(), exporter=exporter, set_global=False)
    assert isinstance(provider, TracerProvider)
    tracer = provider.get_tracer("t")
    with tracer.start_as_current_span("request"), tracer.start_as_current_span("store_frame"):
        pass
    spans = {s.name: s for s in exporter.get_finished_spans()}
    assert set(spans) == {"request", "store_frame"}
    child, parent = spans["store_frame"], spans["request"]
    assert child.parent is not None
    assert child.parent.span_id == parent.context.span_id
    assert parent.resource.attributes["service.name"] == "ingest"
    shutdown_telemetry(provider)


def test_zero_sample_ratio_drops_spans() -> None:
    exporter = InMemorySpanExporter()
    provider = init_telemetry(
        "ingest", TelemetrySettings(sample_ratio=0.0), exporter=exporter, set_global=False
    )
    with provider.get_tracer("t").start_as_current_span("dropped"):
        pass
    assert exporter.get_finished_spans() == ()


def test_enabled_builds_otlp_pipeline_without_connecting() -> None:
    # The OTLP exporter connects lazily, so construction must not need a collector.
    provider = init_telemetry(
        "ingest",
        TelemetrySettings(enabled=True, otlp_endpoint="localhost:1", export_timeout_s=0.1),
        set_global=False,
    )
    assert isinstance(provider, TracerProvider)
    provider.shutdown()


def test_traceparent_round_trips_through_text() -> None:
    exporter = InMemorySpanExporter()
    provider = init_telemetry("ingest", TelemetrySettings(), exporter=exporter, set_global=False)
    tracer = provider.get_tracer("t")
    with tracer.start_as_current_span("ingest.frame") as span:
        stored = current_traceparent()
    ctx = span.get_span_context()
    assert stored is not None
    assert stored.startswith(f"00-{ctx.trace_id:032x}-{ctx.span_id:016x}-")
    assert int(stored[-2:], 16) & 0x01  # sampled (newer SDKs also set 0x02, random trace id)

    with tracer.start_as_current_span("worker.frame", context=context_from_traceparent(stored)):
        pass
    child = exporter.get_finished_spans()[-1]
    assert child.context is not None
    assert child.parent is not None
    assert child.context.trace_id == ctx.trace_id
    assert child.parent.span_id == ctx.span_id


def test_no_span_or_bad_text_gives_none() -> None:
    assert current_traceparent() is None  # the API's no-op provider: no valid span
    for bad in (None, "", "garbage", "00-" + "0" * 32 + "-" + "0" * 16 + "-01"):
        assert context_from_traceparent(bad) is None


def test_remote_parent_attaches_the_header_only_without_an_active_span() -> None:
    header = {"traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"}
    with remote_parent(header):
        ctx = trace.get_current_span().get_span_context()
        assert format(ctx.span_id, "016x") == "00f067aa0ba902b7"
    assert not trace.get_current_span().get_span_context().is_valid  # detached again

    # FastAPI's own server span is active: it stays the parent.
    provider = init_telemetry(
        "ingest", TelemetrySettings(), exporter=InMemorySpanExporter(), set_global=False
    )
    tracer = provider.get_tracer("t")
    with tracer.start_as_current_span("POST /v1/frames") as server, remote_parent(header):
        assert trace.get_current_span() is server


def test_probes_and_scrapes_are_not_traced() -> None:
    exclude = http_telemetry()["exclude"]
    assert exclude is not None
    for path in ("/healthz", "/readyz", "/metrics"):
        assert exclude({"type": "http", "path": path})
    assert not exclude({"type": "http", "path": "/v1/frames"})
