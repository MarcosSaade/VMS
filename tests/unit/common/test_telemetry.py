from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from vms.common.settings import TelemetrySettings
from vms.common.telemetry import init_telemetry, shutdown_telemetry


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
