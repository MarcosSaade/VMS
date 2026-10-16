"""Prometheus metrics with enforced naming: `vms_<component>_<metric>_<unit>`.

Metrics are the cheap, aggregated signal (rates, latency histograms, queue lag) that dashboards,
alerts, autoscaling (KEDA on queue lag) and load-test reports read. Consistent names make
dashboards and alert rules predictable, so the name is assembled and validated here instead of
being typed by hand:

    metrics = Metrics("ingest")
    frames = metrics.counter("received_frames", "total", "Frames accepted", labels=["camera_id"])
    latency = metrics.histogram("request_duration", "seconds", "Ingest request latency")

Rules (Prometheus conventions):
- component and metric are lowercase snake_case;
- counters use unit `total` (they only go up; rate() is taken over them);
- gauges and histograms use a base unit: `seconds`, `bytes`, `ratio` or a counted thing.

Each `Metrics` owns a registry (default: a fresh one), so tests never collide on the global one.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest
from prometheus_client.metrics import MetricWrapperBase
from prometheus_client.process_collector import ProcessCollector
from prometheus_client.registry import Collector

PREFIX: Final = "vms"
COUNTER_UNIT: Final = "total"
BASE_UNITS: Final = frozenset(
    {
        "seconds",
        "bytes",
        "ratio",
        "info",
        # Counted things, for gauges and histograms such as batch sizes or queue depth.
        "frames",
        "rows",
        "requests",
        "messages",
        "detections",
        "tokens",
        "cameras",
        "items",
        "usd",
        # A statistic in its own units (a mean luma level, a standardized shift, a PSI), for
        # gauges whose unit depends on a label (vms_drift_*{signal=...}).
        "value",
    }
)
_SNAKE = re.compile(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$")


def metric_name(component: str, metric: str, unit: str) -> str:
    """Build and validate `vms_<component>_<metric>_<unit>`."""
    for part, value in (("component", component), ("metric", metric)):
        if not _SNAKE.match(value):
            raise ValueError(f"{part} must be lowercase snake_case, got {value!r}")
    if unit != COUNTER_UNIT and unit not in BASE_UNITS:
        allowed = ", ".join(sorted(BASE_UNITS | {COUNTER_UNIT}))
        raise ValueError(f"unknown unit {unit!r}; allowed: {allowed}")
    if metric.endswith(f"_{unit}") or metric == unit:
        raise ValueError(f"metric {metric!r} already ends with its unit {unit!r}")
    return f"{PREFIX}_{component}_{metric}_{unit}"


class Metrics:
    """Factory for one component's metrics. Repeated calls with the same name return the same
    metric, so modules can declare what they need without coordinating registration."""

    def __init__(
        self,
        component: str,
        registry: CollectorRegistry | None = None,
        *,
        process_metrics: bool = False,
    ) -> None:
        metric_name(component, "check", "total")  # validate the component once, up front
        self.component = component
        self.registry = registry if registry is not None else CollectorRegistry()
        self._metrics: dict[str, MetricWrapperBase] = {}
        if process_metrics:
            # CPU, memory and file descriptors of this process (standard `process_*` names).
            collector: Collector = ProcessCollector(registry=None)
            self.registry.register(collector)

    def counter(
        self, metric: str, unit: str, documentation: str, labels: Sequence[str] = ()
    ) -> Counter:
        if unit != COUNTER_UNIT:
            raise ValueError(f"counters must use unit {COUNTER_UNIT!r}, got {unit!r}")
        name = metric_name(self.component, metric, unit)
        return self._get_or_create(name, Counter, documentation, labels)

    def gauge(
        self, metric: str, unit: str, documentation: str, labels: Sequence[str] = ()
    ) -> Gauge:
        self._require_base_unit("gauge", unit)
        name = metric_name(self.component, metric, unit)
        return self._get_or_create(name, Gauge, documentation, labels)

    def histogram(
        self,
        metric: str,
        unit: str,
        documentation: str,
        labels: Sequence[str] = (),
        buckets: Sequence[float] | None = None,
    ) -> Histogram:
        self._require_base_unit("histogram", unit)
        name = metric_name(self.component, metric, unit)
        existing = self._metrics.get(name)
        if existing is not None:
            return self._check_type(name, existing, Histogram)
        created = Histogram(
            name,
            documentation,
            labels,
            registry=self.registry,
            buckets=tuple(buckets) if buckets is not None else Histogram.DEFAULT_BUCKETS,
        )
        self._metrics[name] = created
        return created

    def exposition(self) -> bytes:
        """The registry in Prometheus text format (what `/metrics` serves)."""
        return generate_latest(self.registry)

    def _get_or_create[M: MetricWrapperBase](
        self, name: str, kind: type[M], documentation: str, labels: Sequence[str]
    ) -> M:
        existing = self._metrics.get(name)
        if existing is not None:
            return self._check_type(name, existing, kind)
        created = kind(name, documentation, labels, registry=self.registry)
        self._metrics[name] = created
        return created

    @staticmethod
    def _check_type[M: MetricWrapperBase](name: str, metric: MetricWrapperBase, kind: type[M]) -> M:
        if not isinstance(metric, kind):
            raise ValueError(f"{name} already registered as {type(metric).__name__}")
        return metric

    @staticmethod
    def _require_base_unit(kind: str, unit: str) -> None:
        if unit == COUNTER_UNIT:
            raise ValueError(f"unit {COUNTER_UNIT!r} is reserved for counters, not a {kind}")
