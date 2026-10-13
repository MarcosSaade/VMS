"""Counts and timings of a loadgen run: live Prometheus metrics plus a steady-state summary.

Prometheus metrics (scraped while a run is live) count everything. The summary written to the
results JSON keeps only samples whose due time is after the warmup, and reports percentiles
from the raw samples rather than from histogram buckets.

Two timings answer different questions:
- send latency: request start to response; the server's (and network's) speed.
- schedule lag: how late a frame's send started relative to its due time. It is the
  generator's own health signal: if lag grows, the generator (or its host) cannot keep up and
  the offered load is lower than configured, so the run's numbers are not trustworthy.
"""

from __future__ import annotations

from collections import Counter as CountMap
from collections.abc import Sequence
from typing import Any

import numpy as np

from vms.bench.stats import percentiles_ms
from vms.common.metrics import Metrics
from vms.loadgen.transport import Outcome, SendResult

LATENCY_BUCKETS_S = (
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
)
LAG_BUCKETS_S = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)


def percentiles(samples: Sequence[float]) -> dict[str, float | None]:
    """p50/p95/p99, mean and max in milliseconds; None when there are no samples."""
    return percentiles_ms(samples)


class Recorder:
    """`due` arguments are seconds since the run start (the schedule's time base)."""

    def __init__(self, metrics: Metrics, warmup_s: float) -> None:
        self._warmup_s = warmup_s
        self._sent = metrics.counter("sent_frames", "total", "Frames whose send started")
        self._accepted = metrics.counter("accepted_frames", "total", "Frames accepted (2xx)")
        self._rejected = metrics.counter(
            "rejected_frames", "total", "Frames shed by the server (429/503)"
        )
        self._failed = metrics.counter(
            "failed_frames",
            "total",
            "Frames that failed (timeout, connection, error status)",
            labels=["reason"],
        )
        self._dropped = metrics.counter(
            "dropped_frames", "total", "Frames dropped before sending: camera in-flight limit hit"
        )
        self._latency = metrics.histogram(
            "send_duration",
            "seconds",
            "Request start to response, incl. retries",
            buckets=LATENCY_BUCKETS_S,
        )
        self._lag = metrics.histogram(
            "schedule_lag", "seconds", "Send start minus due time", buckets=LAG_BUCKETS_S
        )
        self._in_flight = metrics.gauge("in_flight", "requests", "Sends awaiting a response")
        # Steady-state samples (after warmup).
        self.latencies_s: list[float] = []
        self.lags_s: list[float] = []
        self.counts: CountMap[str] = CountMap()
        self.failure_reasons: CountMap[str] = CountMap()
        self.sent_per_second: CountMap[int] = CountMap()
        self.retries = 0
        self.total_sent = 0  # including warmup, for CPU per frame

    def _steady(self, due: float) -> bool:
        return due >= self._warmup_s

    def dropped(self, due: float) -> None:
        self._dropped.inc()
        if self._steady(due):
            self.counts["dropped"] += 1

    def sent(self, due: float, lag_s: float) -> None:
        self._sent.inc()
        self.total_sent += 1
        self._lag.observe(lag_s)
        self._in_flight.inc()
        if self._steady(due):
            self.counts["sent"] += 1
            self.lags_s.append(lag_s)
            self.sent_per_second[int(due - self._warmup_s)] += 1

    def completed(self, due: float, result: SendResult, latency_s: float) -> None:
        self._in_flight.dec()
        self._latency.observe(latency_s)
        if result.outcome is Outcome.ACCEPTED:
            self._accepted.inc()
        elif result.outcome is Outcome.REJECTED:
            self._rejected.inc()
        else:
            self._failed.labels(reason=result.reason).inc()
        if self._steady(due):
            self.counts[result.outcome.value] += 1
            self.retries += result.attempts - 1
            if result.outcome is Outcome.FAILED:
                self.failure_reasons[result.reason] += 1
            if result.outcome is Outcome.ACCEPTED:
                self.latencies_s.append(latency_s)

    def summary(self, steady_s: float, offered_fps: float | None) -> dict[str, Any]:
        """Steady-state numbers for the results JSON. `steady_s` excludes the warmup."""
        sent = self.counts["sent"]
        done = sum(self.counts[o.value] for o in Outcome)
        # Full seconds only; the last partial second would bias the spread.
        per_second = [self.sent_per_second.get(s, 0) for s in range(int(steady_s))]
        return {
            "steady_s": round(steady_s, 3),
            "offered_fps": offered_fps,
            "sent_fps": round(sent / steady_s, 2) if steady_s > 0 else None,
            "accepted_fps": round(self.counts["accepted"] / steady_s, 2) if steady_s else None,
            "sent_fps_per_second_std": (
                round(float(np.std(per_second)), 2) if len(per_second) > 1 else None
            ),
            "frames": {
                "sent": sent,
                "accepted": self.counts["accepted"],
                "rejected": self.counts["rejected"],
                "failed": self.counts["failed"],
                "dropped": self.counts["dropped"],
                "incomplete": sent - done,
                "retries": self.retries,
            },
            "failure_reasons": dict(sorted(self.failure_reasons.items())),
            "error_rate": round(self.counts["failed"] / sent, 6) if sent else None,
            "send_latency_ms": percentiles(self.latencies_s),
            "schedule_lag_ms": percentiles(self.lags_s),
        }
