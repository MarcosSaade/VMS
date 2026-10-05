"""OpenTelemetry tracing setup.

A trace is the tree of timed spans for one unit of work, for example one frame going
loadgen -> ingest -> queue -> worker -> Triton -> DB write. It shows where latency goes, which
aggregate metrics cannot. Spans are exported over OTLP to the collector and stored in Tempo.

When telemetry is disabled (the default, and in every unit test) the global tracer provider stays
the API's no-op provider: instrumented code still runs, but spans are non-recording and cost
almost nothing. Tests that need to inspect spans pass their own exporter (e.g. an in-memory one).
"""

from __future__ import annotations

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

from vms.common.settings import TelemetrySettings


def init_telemetry(
    service: str,
    settings: TelemetrySettings,
    *,
    exporter: SpanExporter | None = None,
    set_global: bool = True,
) -> trace.TracerProvider:
    """Configure tracing for a service and return the tracer provider.

    `exporter` replaces the OTLP exporter (tests); it is exported synchronously so spans are
    visible immediately. `set_global=False` keeps the process-wide provider untouched, since
    OpenTelemetry allows setting it only once per process.
    """
    if not settings.enabled and exporter is None:
        return trace.get_tracer_provider()

    provider = TracerProvider(
        resource=Resource.create({SERVICE_NAME: service}),
        # Respect the caller's sampling decision so a trace is kept or dropped as a whole.
        sampler=ParentBased(TraceIdRatioBased(settings.sample_ratio)),
    )
    if exporter is not None:
        provider.add_span_processor(SimpleSpanProcessor(exporter))
    else:
        otlp = OTLPSpanExporter(
            endpoint=settings.otlp_endpoint,
            insecure=settings.otlp_insecure,
            timeout=settings.export_timeout_s,
        )
        provider.add_span_processor(BatchSpanProcessor(otlp))
    if set_global:
        trace.set_tracer_provider(provider)
    return provider


def shutdown_telemetry(provider: trace.TracerProvider) -> None:
    """Flush pending spans; call on graceful shutdown. No-op for the API's default provider."""
    if isinstance(provider, TracerProvider):
        provider.shutdown()
