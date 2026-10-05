"""One-call observability setup for a service: JSON logs, traces and metrics.

    settings = load_settings(ServiceSettings, "ingest")
    obs = init_observability(settings)
    app.mount("/metrics", obs.metrics_app())

Then `obs.logger.info(...)`, `with obs.tracer.start_as_current_span(...)` and
`obs.metrics.counter(...)`. Call `obs.shutdown()` on graceful shutdown to flush spans.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TextIO

from opentelemetry import trace
from opentelemetry.sdk.trace.export import SpanExporter
from prometheus_client import make_asgi_app

from vms.common.logs import configure_logging
from vms.common.metrics import Metrics
from vms.common.settings import ServiceSettings
from vms.common.telemetry import init_telemetry, shutdown_telemetry


@dataclass
class Observability:
    service: str
    logger: logging.Logger
    tracer: trace.Tracer
    metrics: Metrics
    tracer_provider: trace.TracerProvider

    def metrics_app(self) -> Callable[..., Any]:
        """ASGI app serving this service's registry in Prometheus format (mount at /metrics)."""
        app: Callable[..., Any] = make_asgi_app(registry=self.metrics.registry)
        return app

    def shutdown(self) -> None:
        shutdown_telemetry(self.tracer_provider)


def init_observability(
    settings: ServiceSettings,
    *,
    span_exporter: SpanExporter | None = None,
    set_global_tracer: bool = True,
    log_stream: TextIO | None = None,
    process_metrics: bool = True,
) -> Observability:
    """Configure logging, tracing and metrics for `settings.service`.

    The keyword arguments exist for tests: an in-memory span exporter, leaving the global tracer
    provider alone, and capturing logs.
    """
    service = settings.service
    configure_logging(service, settings.log.level, stream=log_stream)
    provider = init_telemetry(
        service, settings.telemetry, exporter=span_exporter, set_global=set_global_tracer
    )
    component = service.replace("-", "_")
    return Observability(
        service=service,
        logger=logging.getLogger(f"vms.{component}"),
        tracer=provider.get_tracer(f"vms.{component}"),
        metrics=Metrics(component, process_metrics=process_metrics),
        tracer_provider=provider,
    )
