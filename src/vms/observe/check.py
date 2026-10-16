"""Check that a run's telemetry arrived (task 2.1): metrics in Prometheus, whole traces in Tempo.

`make demo` runs this last, so CI fails if a service stops exporting, a scrape target goes
missing, or the trace context is lost somewhere between loadgen and the worker's DB write.

1. Prometheus: the ingest, worker, drift and query jobs each have a target that is up, every
   metric in REQUIRED_METRICS has at least one series, and every dashboard panel's query
   returns data (vms.observe.dashboards; GPU and placeholder panels exempt).
2. Tempo, frame trace: a trace in which a worker.frame span descends from a loadgen span
   (TraceQL `>>`), containing every span in FRAME_SPANS. Its breakdown is printed: where one
   frame spent its time from send to stored detections.
3. Tempo, query trace: a trace of one question with every span in QUERY_SPANS.

Telemetry arrives asynchronously (scrape interval, collector batching, Tempo ingestion), so each
check is retried until `timeout_s`. Tempo is the slow one: a trace becomes searchable about a
minute or two after its last span arrives (observed with the compose Tempo, E-024).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from vms.common.clock import Clock, SystemClock
from vms.common.logs import configure_logging, get_logger
from vms.observe.dashboards import RATE, Dashboard, all_dashboards, panel_queries

log = get_logger(__name__)

SCRAPE_JOBS = ("ingest", "worker", "drift", "query")
REQUIRED_METRICS = (
    "vms_ingest_frames_total",
    "vms_ingest_stage_duration_seconds_count",
    "vms_ingest_capture_to_receive_seconds_count",
    "vms_worker_frames_total",
    "vms_worker_stage_duration_seconds_count",
    "vms_worker_capture_to_done_seconds_count",  # freshness
    "vms_worker_queue_wait_seconds_count",
    "vms_worker_backlog_frames",
    "vms_worker_backlog_age_seconds",
    # Set for every camera with frames, from the monitor's first update (the statistics
    # themselves need a finished reference period, longer than a CI demo).
    "vms_drift_window_frames",
    "vms_drift_updates_total",
    "vms_query_requests_total",
    "vms_query_stage_duration_seconds_count",
    "vms_query_llm_tokens_total",
)
FRAME_TRACE_QUERY = (
    '{resource.service.name="loadgen"} >> {resource.service.name="worker" && name="worker.frame"}'
)
FRAME_SPANS = frozenset(
    {
        "loadgen.send_frame",
        "ingest.frame",
        "ingest.object_put",
        "ingest.db_insert",
        "worker.queue_wait",
        "worker.frame",
        "worker.fetch",
        "worker.decode",
        "worker.infer",
        "worker.write",
    }
)
QUERY_TRACE_QUERY = '{resource.service.name="query" && name="query.execute"}'
QUERY_SPANS = frozenset(
    {
        "query",
        "query.catalog",
        "query.plan",
        "query.llm",
        "query.compile",
        "query.validate",
        "query.execute",
    }
)

FetchJson = Callable[[str], Any]


class CheckSettings(BaseSettings):
    """Host ports of the stack, from the same `.env` compose was started with."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    prometheus_port: int = Field(default=9090, ge=1, le=65535)
    tempo_port: int = Field(default=3200, ge=1, le=65535)
    http_timeout_s: float = Field(default=5.0, gt=0)


@dataclass(frozen=True)
class Span:
    service: str
    name: str
    start_ns: int
    end_ns: int

    @property
    def duration_ms(self) -> float:
        return (self.end_ns - self.start_ns) / 1e6


class TelemetryCheck:
    def __init__(
        self,
        fetch_json: FetchJson,
        *,
        prometheus_url: str,
        tempo_url: str,
        clock: Clock,
        since_unix_s: int | None = None,
        lookback_s: float = 3600.0,
    ) -> None:
        """Traces are searched from `since_unix_s` (else `lookback_s` ago), so a check after a
        run is not satisfied by traces an earlier run left in Tempo."""
        self._get = fetch_json
        self._prom = prometheus_url.rstrip("/")
        self._tempo = tempo_url.rstrip("/")
        self._clock = clock
        self._since = since_unix_s
        self._lookback_s = lookback_s

    # --- Prometheus ---------------------------------------------------------------------

    def missing_targets(self) -> list[str]:
        """Scrape jobs without a healthy target."""
        data = self._get(f"{self._prom}/api/v1/targets?state=active")
        up = {t["labels"]["job"] for t in data["data"]["activeTargets"] if t.get("health") == "up"}
        return [job for job in SCRAPE_JOBS if job not in up]

    def missing_metrics(self) -> list[str]:
        missing = []
        for name in REQUIRED_METRICS:
            query = urllib.parse.urlencode({"query": f"count({name})"})
            data = self._get(f"{self._prom}/api/v1/query?{query}")
            if not data["data"]["result"]:
                missing.append(name)
        return missing

    def empty_panels(self, dashboards: Sequence[Dashboard]) -> list[str]:
        """Dashboard panels whose query returns no series (GPU and placeholder panels are
        exempt). `$__rate_interval` becomes 5m and `$__range` 15m, which cover a demo run."""
        empty = []
        for d in dashboards:
            for title, expr in panel_queries(d):
                query = urllib.parse.urlencode(
                    {"query": expr.replace(RATE, "5m").replace("$__range", "15m")}
                )
                data = self._get(f"{self._prom}/api/v1/query?{query}")
                if not data["data"]["result"]:
                    empty.append(f"{d.title}: {title}")
        return sorted(set(empty))

    # --- Tempo --------------------------------------------------------------------------

    def find_trace(self, traceql: str, required: frozenset[str]) -> list[Span] | None:
        """The first matching trace (most recent first) that has every span in `required`."""
        end = int(self._clock.now().timestamp())
        start = self._since if self._since is not None else end - int(self._lookback_s)
        params = urllib.parse.urlencode({"q": traceql, "start": start, "end": end + 1, "limit": 20})
        found = self._get(f"{self._tempo}/api/search?{params}").get("traces") or []
        for hit in sorted(found, key=lambda t: int(t["startTimeUnixNano"]), reverse=True):
            spans = parse_trace(self._get(f"{self._tempo}/api/v2/traces/{hit['traceID']}"))
            if required <= {s.name for s in spans}:
                return spans
        return None


def parse_trace(body: dict[str, Any]) -> list[Span]:
    """Tempo's /api/v2/traces/<id> (OTLP JSON) as spans sorted by start time."""
    spans = []
    for resource in body.get("trace", {}).get("resourceSpans", []):
        attrs = resource.get("resource", {}).get("attributes", [])
        service = next(
            (a["value"].get("stringValue", "") for a in attrs if a["key"] == "service.name"), ""
        )
        for scope in resource.get("scopeSpans", []):
            for s in scope.get("spans", []):
                spans.append(
                    Span(service, s["name"], int(s["startTimeUnixNano"]), int(s["endTimeUnixNano"]))
                )
    return sorted(spans, key=lambda s: s.start_ns)


def format_trace(title: str, spans: Sequence[Span]) -> str:
    t0 = spans[0].start_ns
    lines = [title, f"  {'start ms':>9}  {'duration ms':>11}  service   span"]
    lines += [
        f"  {(s.start_ns - t0) / 1e6:9.1f}  {s.duration_ms:11.1f}  {s.service:<8}  {s.name}"
        for s in spans
    ]
    return "\n".join(lines) + "\n"


def retry[T](
    check: Callable[[], T],
    done: Callable[[T], bool],
    *,
    deadline: float,
    interval_s: float,
    clock: Clock,
    sleep: Callable[[float], None],
) -> T:
    """Call `check` until `done(result)` or the (monotonic) deadline; returns the last result.
    Network errors count as not done (a backend may still be starting)."""
    while True:
        try:
            result = check()
            if done(result) or clock.monotonic() >= deadline:
                return result
        except OSError as exc:
            if clock.monotonic() >= deadline:
                raise
            log.info("telemetry backend not reachable yet", extra={"error": str(exc)})
        sleep(interval_s)


def run_checks(
    checker: TelemetryCheck,
    *,
    timeout_s: float,
    interval_s: float,
    clock: Clock,
    sleep: Callable[[float], None],
    out: Callable[[str], Any],
    dashboards: Sequence[Dashboard] = (),
) -> list[str]:
    """Run every check, all within one `timeout_s`; returns the failures (empty if none)."""
    wait: dict[str, Any] = {
        "deadline": clock.monotonic() + timeout_s,
        "interval_s": interval_s,
        "clock": clock,
        "sleep": sleep,
    }
    failures = []
    targets = retry(checker.missing_targets, lambda m: not m, **wait)
    if targets:
        failures.append(f"Prometheus has no healthy target for: {', '.join(targets)}")
    metrics = retry(checker.missing_metrics, lambda m: not m, **wait)
    if metrics:
        failures.append(f"Prometheus has no series for: {', '.join(metrics)}")
    panels = retry(lambda: checker.empty_panels(dashboards), lambda m: not m, **wait)
    if panels:
        failures.append(f"Dashboard panels with no data: {'; '.join(panels)}")

    frame = retry(
        lambda: checker.find_trace(FRAME_TRACE_QUERY, FRAME_SPANS), lambda t: t is not None, **wait
    )
    if frame is None:
        failures.append(
            "Tempo has no complete frame trace (loadgen -> ingest -> worker) with spans: "
            + ", ".join(sorted(FRAME_SPANS))
        )
    else:
        out(format_trace("One frame, from loadgen to stored detections:", frame))
    query = retry(
        lambda: checker.find_trace(QUERY_TRACE_QUERY, QUERY_SPANS), lambda t: t is not None, **wait
    )
    if query is None:
        failures.append(
            "Tempo has no complete query trace with spans: " + ", ".join(sorted(QUERY_SPANS))
        )
    else:
        out(format_trace("One question, through the query pipeline:", query))
    return failures


def http_json(timeout_s: float) -> FetchJson:
    def fetch(url: str) -> Any:
        with urllib.request.urlopen(url, timeout=timeout_s) as response:
            return json.load(response)

    return fetch


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vms.observe", description=__doc__)
    parser.add_argument(
        "--since", type=int, help="only traces from this Unix time on (default: the last hour)"
    )
    # Tempo makes a trace searchable a minute or more after its last span arrives.
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument("--interval-s", type=float, default=3.0)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure_logging("observe", args.log_level)
    settings = CheckSettings()
    checker = TelemetryCheck(
        http_json(settings.http_timeout_s),
        prometheus_url=f"http://localhost:{settings.prometheus_port}",
        tempo_url=f"http://localhost:{settings.tempo_port}",
        clock=SystemClock(),
        since_unix_s=args.since,
    )
    failures = run_checks(
        checker,
        timeout_s=args.timeout_s,
        interval_s=args.interval_s,
        clock=SystemClock(),
        sleep=time.sleep,
        out=sys.stdout.write,
        dashboards=all_dashboards(),
    )
    for failure in failures:
        log.error("telemetry check failed", extra={"failure": failure})
    if not failures:
        sys.stdout.write("Telemetry check passed: metrics in Prometheus, traces in Tempo.\n")
    return 1 if failures else 0
