"""OpenTelemetry tracing setup.

A trace is the tree of timed spans for one unit of work, for example one frame going
loadgen -> ingest -> queue -> worker -> Triton -> DB write. It shows where latency goes, which
aggregate metrics cannot. Spans are exported over OTLP to the collector and stored in Tempo.

When telemetry is disabled (the default, and in every unit test) the global tracer provider stays
the API's no-op provider: instrumented code still runs, but spans are non-recording and cost
almost nothing. Tests that need to inspect spans pass their own exporter (e.g. an in-memory one).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, MutableMapping
from contextlib import contextmanager
from typing import Any, Final

from fastapi.telemetry import TelemetryConfig
from opentelemetry import context as otel_context
from opentelemetry import propagate, trace
from opentelemetry.context import Context
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

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


# W3C trace context, independent of the globally configured propagators: the value is stored in
# the database (frames.traceparent), so its format must not change with process configuration.
_W3C = TraceContextTextMapPropagator()


def current_traceparent() -> str | None:
    """The active span as a W3C `traceparent`, or None when there is no valid span.

    Kept for unsampled spans too (flags `00`): whoever continues the trace then makes the same
    "do not record" decision instead of sampling the frame again on its own.
    """
    carrier: dict[str, str] = {}
    _W3C.inject(carrier)
    return carrier.get("traceparent")


def context_from_traceparent(traceparent: str | None) -> Context | None:
    """A context whose parent is the span in `traceparent`; None if absent or malformed (the
    caller then starts a new trace)."""
    if not traceparent:
        return None
    ctx = _W3C.extract({"traceparent": traceparent})
    if not trace.get_current_span(ctx).get_span_context().is_valid:
        return None
    return ctx


@contextmanager
def remote_parent(headers: Mapping[str, str]) -> Iterator[None]:
    """Continue the caller's trace (its `traceparent` header) inside an HTTP handler.

    FastAPI's built-in telemetry already does this, and starts a server span, whenever a tracer
    provider is configured; that span must stay the parent, so nothing is attached then. This
    covers the remaining case: telemetry off globally but a tracer injected (tests).
    """
    if trace.get_current_span().get_span_context().is_valid:
        yield
        return
    token = otel_context.attach(propagate.extract(dict(headers)))
    try:
        yield
    finally:
        otel_context.detach(token)


# Probes and scrapes: Docker's health check and Prometheus call these every few seconds, which
# would bury real requests in Tempo under one-span traces.
UNTRACED_PATHS: Final = frozenset({"/healthz", "/readyz", "/metrics"})


def http_telemetry() -> TelemetryConfig:
    """`FastAPI(telemetry=...)` for every service: FastAPI's built-in server spans (and trace
    context extraction), except for probes and metric scrapes."""

    def exclude(scope: MutableMapping[str, Any]) -> bool:
        return scope.get("path") in UNTRACED_PATHS

    return TelemetryConfig(exclude=exclude)
