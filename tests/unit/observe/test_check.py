"""The telemetry check against fake Prometheus and Tempo HTTP APIs."""

from __future__ import annotations

import urllib.parse
from datetime import UTC, datetime
from typing import Any

from vms.common.clock import FakeClock
from vms.observe.check import (
    FRAME_SPANS,
    QUERY_SPANS,
    REQUIRED_METRICS,
    SCRAPE_JOBS,
    TelemetryCheck,
    format_trace,
    parse_trace,
    run_checks,
)

NOW = datetime(2026, 10, 13, 10, 0, tzinfo=UTC)


def otlp_trace(spans: list[tuple[str, str, int, int]]) -> dict[str, Any]:
    """Tempo's /api/v2/traces body: one resource per service."""
    by_service: dict[str, list[dict[str, Any]]] = {}
    for service, name, start, end in spans:
        by_service.setdefault(service, []).append(
            {"name": name, "startTimeUnixNano": str(start), "endTimeUnixNano": str(end)}
        )
    return {
        "trace": {
            "resourceSpans": [
                {
                    "resource": {
                        "attributes": [{"key": "service.name", "value": {"stringValue": svc}}]
                    },
                    "scopeSpans": [{"spans": items}],
                }
                for svc, items in by_service.items()
            ]
        }
    }


def frame_trace(missing: str = "") -> dict[str, Any]:
    names = sorted(FRAME_SPANS - {missing})
    return otlp_trace(
        [(n.split(".")[0], n, 1_000_000 * i, 1_000_000 * (i + 2)) for i, n in enumerate(names)]
    )


def query_trace() -> dict[str, Any]:
    return otlp_trace([("query", n, 1_000 * i, 1_000 * (i + 1)) for i, n in enumerate(QUERY_SPANS)])


class FakeBackends:
    def __init__(self) -> None:
        self.jobs_up = set(SCRAPE_JOBS)
        self.metrics = set(REQUIRED_METRICS)
        self.traces: dict[str, dict[str, Any]] = {"f-old": frame_trace(), "q1": query_trace()}
        self.search: dict[str, list[str]] = {"worker.frame": ["f-old"], "query": ["q1"]}
        self.urls: list[str] = []

    def __call__(self, url: str) -> Any:
        self.urls.append(url)
        parsed = urllib.parse.urlsplit(url)
        params = dict(urllib.parse.parse_qsl(parsed.query))
        if parsed.path == "/api/v1/targets":
            return {
                "data": {
                    "activeTargets": [
                        {"labels": {"job": j}, "health": "up" if j in self.jobs_up else "down"}
                        for j in SCRAPE_JOBS
                    ]
                }
            }
        if parsed.path == "/api/v1/query":
            name = params["query"].removeprefix("count(").removesuffix(")")
            return {"data": {"result": [{"value": [0, "1"]}] if name in self.metrics else []}}
        if parsed.path == "/api/search":
            key = "worker.frame" if "worker.frame" in params["q"] else "query"
            ids = self.search[key]
            return {
                "traces": [{"traceID": t, "startTimeUnixNano": str(i)} for i, t in enumerate(ids)]
            }
        if parsed.path.startswith("/api/v2/traces/"):
            return self.traces[parsed.path.rsplit("/", 1)[1]]
        raise AssertionError(url)


def check(backends: FakeBackends) -> tuple[list[str], list[str]]:
    clock = FakeClock(NOW)
    checker = TelemetryCheck(
        backends, prometheus_url="http://prom", tempo_url="http://tempo", clock=clock
    )
    printed: list[str] = []
    failures = run_checks(
        checker,
        timeout_s=6.0,
        interval_s=3.0,
        clock=clock,
        sleep=clock.advance,
        out=printed.append,
    )
    return failures, printed


def test_all_present_passes_and_prints_both_breakdowns() -> None:
    failures, printed = check(FakeBackends())
    assert failures == []
    assert printed[0].startswith("One frame, from loadgen to stored detections:")
    assert "worker.queue_wait" in printed[0]
    assert printed[1].startswith("One question")


def test_missing_target_metric_and_spans_are_each_reported() -> None:
    backends = FakeBackends()
    backends.jobs_up.discard("worker")
    backends.metrics.discard("vms_worker_capture_to_done_seconds_count")
    # The only frame trace lost the queue hop: the worker started its own trace.
    backends.traces["f-old"] = frame_trace(missing="worker.queue_wait")
    failures, _ = check(backends)
    assert len(failures) == 3
    assert failures[0] == "Prometheus has no healthy target for: worker"
    assert failures[1] == "Prometheus has no series for: vms_worker_capture_to_done_seconds_count"
    assert "no complete frame trace" in failures[2]


def test_a_complete_trace_is_found_among_incomplete_ones() -> None:
    backends = FakeBackends()
    backends.traces["f-new"] = frame_trace(missing="worker.write")  # e.g. still in flight
    backends.search["worker.frame"] = ["f-old", "f-new"]
    failures, _ = check(backends)
    assert failures == []


def test_search_window_starts_at_since() -> None:
    backends = FakeBackends()
    clock = FakeClock(NOW)
    since = int(NOW.timestamp()) - 120
    checker = TelemetryCheck(
        backends, prometheus_url="http://p", tempo_url="http://t", clock=clock, since_unix_s=since
    )
    checker.find_trace("{}", frozenset())
    params = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(backends.urls[-2]).query))
    assert int(params["start"]) == since


def test_search_window_ends_now() -> None:
    backends = FakeBackends()
    check(backends)
    search = next(u for u in backends.urls if "/api/search" in u)
    params = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(search).query))
    assert int(params["end"]) == int(NOW.timestamp()) + 1
    assert int(params["start"]) == int(NOW.timestamp()) - 3600


def test_parse_and_format_trace_in_start_order() -> None:
    spans = parse_trace(
        otlp_trace(
            [("worker", "worker.frame", 5_000_000, 9_000_000), ("ingest", "a", 0, 1_000_000)]
        )
    )
    assert [(s.service, s.name) for s in spans] == [("ingest", "a"), ("worker", "worker.frame")]
    text = format_trace("t", spans)
    assert text.splitlines()[-1].split() == ["5.0", "4.0", "worker", "worker.frame"]


def test_late_traces_are_waited_for_within_one_overall_deadline() -> None:
    backends = FakeBackends()
    backends.search["worker.frame"] = []  # not searchable yet
    clock = FakeClock(NOW)
    checker = TelemetryCheck(backends, prometheus_url="http://p", tempo_url="http://t", clock=clock)

    def sleep(seconds: float) -> None:
        clock.advance(seconds)
        if clock.monotonic() >= 3.0:  # Tempo indexes the trace a few seconds later
            backends.search["worker.frame"] = ["f-old"]

    failures = run_checks(
        checker, timeout_s=6.0, interval_s=3.0, clock=clock, sleep=sleep, out=lambda _: None
    )
    assert failures == []

    # Never arrives: every check gives up by the same deadline, not one timeout each.
    backends.search["worker.frame"], backends.search["query"] = [], []
    clock = FakeClock(NOW)
    checker = TelemetryCheck(backends, prometheus_url="http://p", tempo_url="http://t", clock=clock)
    failures = run_checks(
        checker, timeout_s=6.0, interval_s=3.0, clock=clock, sleep=clock.advance, out=print
    )
    assert len(failures) == 2
    assert clock.monotonic() == 6.0
