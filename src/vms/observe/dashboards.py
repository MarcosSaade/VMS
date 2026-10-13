"""Grafana dashboards as code (task 2.2): Pipeline, Inference, Query, Resources, Drift.

The dashboards are built here and written as JSON to `deploy/grafana/dashboards/`, which Grafana
provisions on `make up`. The JSON is generated, never edited by hand:

    python -m vms.observe.dashboards           # rewrite the JSON files
    python -m vms.observe.dashboards --check   # fail if they differ from this code (CI)

Every panel's description says what it shows and which decision it informs, so each panel
earns its place. `python -m vms.observe` (the demo's last step) runs every panel query against
Prometheus and fails on an empty one, except panels marked `optional` (GPU metrics exist only on
the GPU instance; drift arrives in task 9.4).

PromQL conventions: rates over `$__rate_interval` (the datasource's 5 s scrape interval makes it
20 s); queries for counters that may have no series yet (errors, retries) end in
`or vector(0)` so the panel shows zero instead of "No data"; backlog gauges are global and
reported by every worker, so they are aggregated with `max`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

OUT_DIR = Path(__file__).resolve().parents[3] / "deploy" / "grafana" / "dashboards"
PROM = {"type": "prometheus", "uid": "prometheus"}
FRESHNESS_SLO_S = 2.0  # ARCHITECTURE 8
QUERY_SLO_S = 1.5  # ARCHITECTURE 8
GRID_WIDTH = 24
RATE = "$__rate_interval"

PanelKind = Literal["timeseries", "stat", "heatmap", "text"]

# Fixed colors per series, so the same thing has the same color on every dashboard.
GREEN, BLUE, ORANGE, RED, PURPLE, YELLOW = "green", "blue", "orange", "red", "purple", "yellow"


@dataclass(frozen=True)
class Target:
    expr: str
    legend: str = ""
    color: str | None = None
    right_unit: str | None = None  # plot on a right-hand axis with this unit


@dataclass(frozen=True)
class Panel:
    title: str
    description: str
    targets: tuple[Target, ...] = ()
    kind: PanelKind = "timeseries"
    unit: str = "short"
    width: int = 12
    height: int = 8
    min: float | None = 0
    soft_max: float | None = None  # keeps a flat zero line from filling the axis with 0..100
    series_colors: tuple[tuple[str, str], ...] = ()  # (legend name, color) for templated legends
    log_scale: bool = False  # for series minutes apart in size (queue wait vs work)
    thresholds: tuple[tuple[float, str], ...] = ()  # (from value, color) after base green
    threshold_line: bool = False  # draw thresholds on a time series as a dashed line
    stack: bool = False
    decimals: int | None = None
    optional: bool = False  # may be empty (GPU only, placeholders): skipped by the check
    content: str = ""  # text panels

    @property
    def exprs(self) -> list[str]:
        return [t.expr for t in self.targets]


@dataclass(frozen=True)
class Row:
    title: str
    panels: tuple[Panel, ...]
    collapsed: bool = False


@dataclass(frozen=True)
class Dashboard:
    uid: str
    title: str
    description: str
    rows: tuple[Row, ...]
    tags: tuple[str, ...] = ("vms",)

    def panels(self) -> Iterator[Panel]:
        for row in self.rows:
            yield from row.panels


# --- PromQL helpers ---------------------------------------------------------------------


def rate(metric: str, selector: str = "", by: str = "", window: str = RATE) -> str:
    inner = f"rate({metric}{{{selector}}}[{window}])" if selector else f"rate({metric}[{window}])"
    return f"sum by ({by}) ({inner})" if by else f"sum({inner})"


def quantile(q: float, histogram: str, selector: str = "", by: str = "") -> str:
    labels = f"le, {by}" if by else "le"
    sel = f"{{{selector}}}" if selector else ""
    return f"histogram_quantile({q}, sum by ({labels}) (rate({histogram}_bucket{sel}[{RATE}])))"


def zero_if_absent(expr: str, label: str = "", value: str = "none") -> str:
    """`expr`, or a zero series while it has none. With `label`, the zero series carries
    `label=value` so the legend names it rather than showing "Value"."""
    zero = f'label_replace(vector(0), "{label}", "{value}", "", "")' if label else "vector(0)"
    # `or on()` matches on no labels, so the zero series is added only when `expr` is empty
    # (a plain `or` would add it next to any series whose labels differ).
    return f"({expr}) or on() {zero}"


# --- Dashboards -------------------------------------------------------------------------

FRESHNESS = "vms_worker_capture_to_done_seconds"
WORKER_STAGES = "fetch|decode|stats|infer|track|write"
ALL_STAGES = f'stage=~"claim|{WORKER_STAGES}"'
DONE = rate("vms_worker_frames_total", 'outcome="done"')


def pipeline() -> Dashboard:
    return Dashboard(
        uid="vms-pipeline",
        title="VMS / Pipeline",
        description="Frames from ingest to stored detections: throughput, freshness, backlog.",
        rows=(
            Row(
                "Now",
                (
                    Panel(
                        "Ingested",
                        "New frames stored by ingest per second (duplicates excluded). Compare "
                        "with Processed: if ingest is higher for long, the backlog grows.",
                        (
                            Target(
                                rate("vms_ingest_frames_total", 'outcome="created"', window="1m")
                            ),
                        ),
                        kind="stat",
                        unit="suffix: frames/s",
                        width=4,
                        height=4,
                        decimals=1,
                    ),
                    Panel(
                        "Processed",
                        "Frames the workers finished per second. Below Ingested: add workers, "
                        "a faster model (Phase 5) or fewer cameras per worker.",
                        (Target(rate("vms_worker_frames_total", 'outcome="done"', window="1m")),),
                        kind="stat",
                        unit="suffix: frames/s",
                        width=4,
                        height=4,
                        decimals=1,
                    ),
                    Panel(
                        "Freshness p95",
                        "95th percentile of capture time to detections queryable over the last "
                        f"minute. Red above the {FRESHNESS_SLO_S:g} s SLO: the system is past "
                        "its rated capacity.",
                        (
                            Target(
                                "histogram_quantile(0.95, sum by (le) "
                                f"(rate({FRESHNESS}_bucket[1m])))"
                            ),
                        ),
                        kind="stat",
                        unit="s",
                        width=4,
                        height=4,
                        thresholds=((FRESHNESS_SLO_S, RED),),
                        decimals=2,
                    ),
                    Panel(
                        "Within SLO",
                        f"Share of frames stored within {FRESHNESS_SLO_S:g} s of capture over "
                        "the last 5 minutes. The rated-capacity test (2.5) wants this at 95% "
                        "or more.",
                        (
                            Target(
                                f'sum(rate({FRESHNESS}_bucket{{le="{FRESHNESS_SLO_S}"}}[5m])) '
                                f"/ sum(rate({FRESHNESS}_count[5m]))"
                            ),
                        ),
                        kind="stat",
                        unit="percentunit",
                        width=4,
                        height=4,
                        min=0,
                        thresholds=((0, RED), (0.95, GREEN)),
                        decimals=1,
                    ),
                    Panel(
                        "Backlog",
                        "Frames waiting for a worker. Steady near zero is healthy; a rising "
                        "line means the workers are the bottleneck.",
                        (Target("max(vms_worker_backlog_frames)"),),
                        kind="stat",
                        unit="suffix: frames",
                        width=4,
                        height=4,
                        thresholds=((100, ORANGE), (1000, RED)),
                    ),
                    Panel(
                        "Oldest waiting",
                        "How long the oldest pending frame has waited. It bounds freshness "
                        "from below: above the SLO, every new frame will be late too.",
                        (Target("max(vms_worker_backlog_age_seconds)"),),
                        kind="stat",
                        unit="s",
                        width=4,
                        height=4,
                        thresholds=((FRESHNESS_SLO_S, RED),),
                        decimals=1,
                    ),
                ),
            ),
            Row(
                "Throughput and freshness",
                (
                    Panel(
                        "Frames in vs out",
                        "Ingest and worker throughput. The gap between the lines is the rate "
                        "at which the backlog grows: the first signal of overload.",
                        (
                            Target(
                                rate("vms_ingest_frames_total", 'outcome="created"'),
                                "ingested",
                                BLUE,
                            ),
                            Target(
                                rate("vms_worker_frames_total", 'outcome="done"'),
                                "processed",
                                GREEN,
                            ),
                        ),
                        unit="suffix: frames/s",
                    ),
                    Panel(
                        "Freshness (capture to queryable)",
                        "Percentiles of capture time to detections committed, with the "
                        f"{FRESHNESS_SLO_S:g} s SLO as a line. If p95 rises, open a slow "
                        'frame trace in Tempo ({name="worker.frame" && span.freshness_s > 2.0}) '
                        "and see whether queue wait or work grew.",
                        (
                            Target(quantile(0.5, FRESHNESS), "p50", GREEN),
                            Target(quantile(0.95, FRESHNESS), "p95", ORANGE),
                            Target(quantile(0.99, FRESHNESS), "p99", RED),
                        ),
                        unit="s",
                        thresholds=((FRESHNESS_SLO_S, RED),),
                        threshold_line=True,
                    ),
                    Panel(
                        "Backlog",
                        "Pending frames (left) and the oldest one's wait. A sawtooth means "
                        "bursts the workers absorb; a ramp means sustained overload.",
                        (
                            Target("max(vms_worker_backlog_frames)", "pending frames", BLUE),
                            Target(
                                "max(vms_worker_backlog_age_seconds)",
                                "oldest wait",
                                ORANGE,
                                right_unit="s",
                            ),
                        ),
                        unit="suffix: frames",
                    ),
                    Panel(
                        "Where freshness goes: waiting vs working",
                        "Mean time a frame waits in the queue against the mean time a worker "
                        "spends on it (all stages), on a log scale. Waiting dominating means too "
                        "little worker "
                        "capacity; working dominating means the per-frame path itself is slow "
                        "(see Inference).",
                        (
                            Target(
                                f"{rate('vms_worker_queue_wait_seconds_sum')} / "
                                f"{rate('vms_worker_queue_wait_seconds_count')}",
                                "queue wait",
                                ORANGE,
                            ),
                            Target(
                                f"{rate('vms_worker_stage_duration_seconds_sum', ALL_STAGES)} / "
                                f"{DONE}",
                                "worker time per frame",
                                GREEN,
                            ),
                            Target(
                                f"{rate('vms_ingest_capture_to_receive_seconds_sum')} / "
                                f"{rate('vms_ingest_capture_to_receive_seconds_count')}",
                                "capture to ingest",
                                BLUE,
                            ),
                        ),
                        log_scale=True,
                        unit="s",
                    ),
                ),
            ),
            Row(
                "Losses and ingest",
                (
                    Panel(
                        "Frames lost or retried",
                        "Frames ingest refused (invalid, error) and frames the workers failed, "
                        "retried or lost to a lease takeover. Any non-zero line breaks the "
                        "zero-frame-loss condition of rated capacity.",
                        (
                            Target(
                                zero_if_absent(
                                    rate(
                                        "vms_ingest_frames_total",
                                        'outcome=~"invalid|error"',
                                        "outcome",
                                    )
                                ),
                                "ingest {{outcome}}",
                            ),
                            Target(
                                zero_if_absent(
                                    rate(
                                        "vms_worker_frames_total",
                                        'outcome=~"failed|retry|lost"',
                                        "outcome",
                                    ),
                                    "outcome",
                                    "worker: none failed",
                                ),
                                "{{outcome}}",
                            ),
                        ),
                        soft_max=1,
                        unit="suffix: frames/s",
                    ),
                    Panel(
                        "Ingest stage latency p95",
                        "Per-frame ingest work: multipart parsing, the object-store write and "
                        "the row insert. Decides where to optimize ingest (raw body instead of "
                        "multipart, D4; store tuning; batching inserts).",
                        (
                            Target(
                                quantile(
                                    0.95,
                                    "vms_ingest_stage_duration_seconds",
                                    'stage=~"parse|object_put|db_insert"',
                                    "stage",
                                ),
                                "{{stage}}",
                            ),
                        ),
                        unit="s",
                    ),
                    Panel(
                        "Duplicates",
                        "Frames sent again that ingest recognized by their idempotency key "
                        "(D17). Non-zero during a run means client retries; normal after a "
                        "loadgen restart with the same run id.",
                        (
                            Target(
                                zero_if_absent(
                                    rate("vms_ingest_frames_total", 'outcome="duplicate"')
                                ),
                                "duplicates",
                            ),
                        ),
                        soft_max=1,
                        unit="suffix: frames/s",
                    ),
                    Panel(
                        "Ingest request latency",
                        "HTTP latency of POST /v1/frames as the server sees it. Rising with "
                        "flat stage latencies points at the event loop or the connection pool.",
                        (
                            Target(
                                quantile(
                                    0.5,
                                    "vms_ingest_request_duration_seconds",
                                    'route="/v1/frames"',
                                ),
                                "p50",
                                GREEN,
                            ),
                            Target(
                                quantile(
                                    0.95,
                                    "vms_ingest_request_duration_seconds",
                                    'route="/v1/frames"',
                                ),
                                "p95",
                                ORANGE,
                            ),
                        ),
                        unit="s",
                    ),
                ),
            ),
        ),
    )


def inference() -> Dashboard:
    stage_time = (
        f"sum by (stage) (rate(vms_worker_stage_duration_seconds_sum"
        f'{{stage=~"{WORKER_STAGES}"}}[{RATE}]))'
    )
    return Dashboard(
        uid="vms-inference",
        title="VMS / Inference",
        description="Worker internals: per-stage latency, where worker time goes, tracking.",
        rows=(
            Row(
                "Latency",
                (
                    Panel(
                        "Inference latency",
                        "Model call per frame (batch size 1 in v0). p50 against p95: a wide "
                        "gap means contention for CPU (other processes, thread oversubscription).",
                        (
                            Target(
                                quantile(0.5, "vms_worker_stage_duration_seconds", 'stage="infer"'),
                                "p50",
                                GREEN,
                            ),
                            Target(
                                quantile(
                                    0.95, "vms_worker_stage_duration_seconds", 'stage="infer"'
                                ),
                                "p95",
                                ORANGE,
                            ),
                            Target(
                                quantile(
                                    0.99, "vms_worker_stage_duration_seconds", 'stage="infer"'
                                ),
                                "p99",
                                RED,
                            ),
                        ),
                        unit="s",
                    ),
                    Panel(
                        "Inference latency distribution",
                        "Every inference observation per time bucket. Two bands mean two "
                        "regimes (for example warm and cold, or a second process competing).",
                        (
                            Target(
                                "sum by (le) (increase(vms_worker_stage_duration_seconds_bucket"
                                f'{{stage="infer"}}[{RATE}]))',
                                "{{le}}",
                            ),
                        ),
                        kind="heatmap",
                        unit="s",
                    ),
                    Panel(
                        "Stage latency p95",
                        "p95 of each step a worker takes per frame. The tallest line is the "
                        "first optimization target (Phase 5 for infer; decode and stats on "
                        "the CPU path, E-013).",
                        (
                            Target(
                                quantile(
                                    0.95,
                                    "vms_worker_stage_duration_seconds",
                                    f'stage=~"{WORKER_STAGES}"',
                                    "stage",
                                ),
                                "{{stage}}",
                            ),
                        ),
                        unit="s",
                    ),
                    Panel(
                        "Where worker time goes",
                        "Seconds spent in each stage per second of wall time, per worker, "
                        "stacked: the share of a worker's time by stage (the rest is idle or "
                        "lease bookkeeping). It bounds the end-to-end gain of a faster model: "
                        "if infer is 70%, a 2x faster model gives at most 1.5x (Amdahl).",
                        (
                            Target(
                                f"{stage_time} / scalar(count(vms_worker_leased_cameras))",
                                "{{stage}}",
                            ),
                        ),
                        unit="percentunit",
                        stack=True,
                    ),
                ),
            ),
            Row(
                "Throughput and tracking",
                (
                    Panel(
                        "Frames processed per worker",
                        "Throughput of each worker. Uneven lines mean uneven camera leases "
                        "(D20); flat at the model's ceiling means the worker is saturated.",
                        (
                            Target(
                                rate("vms_worker_frames_total", 'outcome="done"', "instance"),
                                "{{instance}}",
                            ),
                        ),
                        unit="suffix: frames/s",
                    ),
                    Panel(
                        "Detections and tracks per frame",
                        "Detections stored and tracked objects per processed frame. A sudden "
                        "change on unchanged cameras is a model or input problem (drift, 9.x).",
                        (
                            Target(
                                f"{rate('vms_worker_detections_total')} / {DONE}",
                                "detections / frame",
                            ),
                            Target(
                                f"{rate('vms_worker_track_sightings_total')} / {DONE}",
                                "tracked objects / frame",
                            ),
                        ),
                        unit="short",
                    ),
                    Panel(
                        "Tracking health",
                        "Late frames (stored without tracking because an earlier-captured "
                        "frame came after a later one), tracker resets after pauses, and lease "
                        "or zone-load errors. Late frames mean ingest reordering; resets mean "
                        "cameras pausing longer than the tracker buffer.",
                        (
                            Target(rate("vms_worker_late_frames_total"), "late frames"),
                            Target(rate("vms_worker_tracker_resets_total"), "tracker resets"),
                            Target(
                                zero_if_absent(rate("vms_worker_loop_errors_total", by="op"), "op"),
                                "queue errors {{op}}",
                            ),
                        ),
                        soft_max=1,
                        unit="suffix: /s",
                    ),
                    Panel(
                        "Zone events",
                        "Enter, exit and loiter events per minute. The structured questions "
                        "count these; zero while people are in view means zones or tracking "
                        "are broken.",
                        (
                            Target(
                                zero_if_absent(
                                    "sum by (event_type) "
                                    f"(rate(vms_worker_zone_events_total[{RATE}])) * 60",
                                    "event_type",
                                ),
                                "{{event_type}}",
                            ),
                        ),
                        unit="suffix: /min",
                    ),
                ),
            ),
            Row(
                "GPU (DCGM exporter, GPU sessions only)",
                (
                    Panel(
                        "GPU utilization",
                        "Share of time the GPU executes kernels. Low while frames queue means "
                        "the GPU waits on the CPU path (decode, pre- and post-processing).",
                        (Target("avg by (gpu) (DCGM_FI_DEV_GPU_UTIL) / 100", "GPU {{gpu}}"),),
                        unit="percentunit",
                        optional=True,
                    ),
                    Panel(
                        "GPU memory used",
                        "Frame-buffer memory in use. Close to the card's total limits the "
                        "batch size (5.3).",
                        (
                            Target(
                                "sum by (gpu) (DCGM_FI_DEV_FB_USED) * 1024 * 1024", "GPU {{gpu}}"
                            ),
                        ),
                        unit="bytes",
                        optional=True,
                    ),
                ),
                collapsed=True,
            ),
        ),
    )


def query() -> Dashboard:
    route = 'route="/v1/query"'
    return Dashboard(
        uid="vms-query",
        title="VMS / Query",
        description="Structured questions: latency by stage, outcomes, LLM tokens.",
        rows=(
            Row(
                "Latency",
                (
                    Panel(
                        "Question latency",
                        "End-to-end latency of POST /v1/query, with the "
                        f"{QUERY_SLO_S:g} s SLO as a line. Above it, check the stage panel: "
                        "with a hosted LLM the planner is usually the whole story (E-017).",
                        (
                            Target(
                                quantile(0.5, "vms_query_request_duration_seconds", route),
                                "p50",
                                GREEN,
                            ),
                            Target(
                                quantile(0.95, "vms_query_request_duration_seconds", route),
                                "p95",
                                ORANGE,
                            ),
                        ),
                        unit="s",
                        thresholds=((QUERY_SLO_S, RED),),
                        threshold_line=True,
                    ),
                    Panel(
                        "Time per stage (mean)",
                        "Mean seconds per question in each stage. Plan (the LLM) against "
                        "execute (Postgres) decides whether to work on the model (smaller, "
                        "local, cached prompt prefix) or on the schema and indexes.",
                        (
                            Target(
                                f"{rate('vms_query_stage_duration_seconds_sum', by='stage')} / "
                                f"{rate('vms_query_stage_duration_seconds_count', by='stage')}",
                                "{{stage}}",
                            ),
                        ),
                        unit="s",
                    ),
                ),
            ),
            Row(
                "Outcomes",
                (
                    Panel(
                        "Questions by outcome",
                        "Answered, abstained (the planner declined: unsupported, ambiguous, "
                        "not recorded) and errors, per minute. A rising abstention share means "
                        "users ask for things the plan language cannot express.",
                        (
                            Target(
                                f"sum by (status) (rate(vms_query_requests_total[{RATE}])) * 60",
                                "{{status}}",
                            ),
                        ),
                        unit="suffix: /min",
                        stack=True,
                        width=8,
                        series_colors=(("ok", GREEN), ("abstained", ORANGE), ("error", RED)),
                    ),
                    Panel(
                        "Errors by type",
                        "Failed questions per minute by stage: planner unavailable (LLM down), "
                        "invalid plan after repair, validator rejection, timeout, execution. "
                        "Any sql_rejected is a compiler bug.",
                        (
                            Target(
                                zero_if_absent(
                                    "sum by (error_type) (rate(vms_query_requests_total"
                                    f'{{status="error"}}[{RATE}])) * 60',
                                    "error_type",
                                ),
                                "{{error_type}}",
                            ),
                        ),
                        soft_max=1,
                        width=8,
                        unit="suffix: /min",
                    ),
                    Panel(
                        "Plan repairs and log failures",
                        "Plans that needed the repair round (each costs a second LLM call) and "
                        "query_log writes that failed (the eval and audit trail loses rows).",
                        (
                            Target(
                                f"sum(rate(vms_query_plan_repairs_total[{RATE}])) * 60", "repairs"
                            ),
                            Target(
                                f"sum(rate(vms_query_query_log_failures_total[{RATE}])) * 60",
                                "query_log failures",
                            ),
                        ),
                        soft_max=1,
                        width=8,
                        unit="suffix: /min",
                    ),
                ),
            ),
            Row(
                "LLM",
                (
                    Panel(
                        "Tokens per minute",
                        "Prompt and completion tokens sent to the planner LLM. Prompt tokens "
                        "dominate (few-shot examples); this is the number that prefix caching "
                        "or a shorter prompt would cut, and what a hosted model bills.",
                        (
                            Target(
                                f"sum by (kind) (rate(vms_query_llm_tokens_total[{RATE}])) * 60",
                                "{{kind}}",
                            ),
                        ),
                        unit="short",
                        stack=True,
                        series_colors=(("prompt", BLUE), ("completion", PURPLE)),
                    ),
                    Panel(
                        "Tokens per question",
                        "Mean tokens per question. Jumps when the prompt or few-shot bank "
                        "changes, or when repairs become common.",
                        (
                            Target(
                                f"sum(rate(vms_query_llm_tokens_total[{RATE}])) / "
                                f"sum(rate(vms_query_requests_total[{RATE}]))",
                                "tokens / question",
                            ),
                        ),
                        unit="short",
                    ),
                ),
            ),
        ),
    )


def resources() -> Dashboard:
    jobs = 'job=~"ingest|worker|query|prometheus|tempo"'
    return Dashboard(
        uid="vms-resources",
        title="VMS / Resources",
        description="CPU and memory per process: ingest, worker, query, Prometheus, Tempo.",
        rows=(
            Row(
                "Processes",
                (
                    Panel(
                        "CPU per service",
                        "CPU cores used by each process. A worker near the machine's core "
                        "count is CPU-bound (add a GPU or a lighter model); ingest near one "
                        "core is limited by its single event loop (E-010). Postgres and the "
                        "object store export no process metrics here.",
                        (
                            Target(
                                f"sum by (job) (rate(process_cpu_seconds_total{{{jobs}}}[{RATE}]))",
                                "{{job}}",
                            ),
                        ),
                        unit="suffix: cores",
                    ),
                    Panel(
                        "Memory per service",
                        "Resident memory of each process. Steady growth under constant load "
                        "is a leak; the worker's level is mostly the model and PyTorch.",
                        (
                            Target(
                                f"sum by (job) (process_resident_memory_bytes{{{jobs}}})",
                                "{{job}}",
                            ),
                        ),
                        unit="bytes",
                    ),
                    Panel(
                        "Open file descriptors",
                        "Sockets and files per process. Growth with load that never comes back "
                        "down means connections are leaking (pools, object-store clients).",
                        (Target(f"sum by (job) (process_open_fds{{{jobs}}})", "{{job}}"),),
                        unit="short",
                    ),
                    Panel(
                        "Worker cameras leased",
                        "Cameras each worker holds (D20). Scaling is by cameras: a worker at "
                        "max_cameras while others hold none means leases are not spreading.",
                        (Target("sum by (instance) (vms_worker_leased_cameras)", "{{instance}}"),),
                        unit="suffix: cameras",
                    ),
                ),
            ),
        ),
    )


def drift() -> Dashboard:
    return Dashboard(
        uid="vms-drift",
        title="VMS / Drift",
        description="Per-camera input and output statistics and change alarms (task 9.4).",
        rows=(
            Row(
                "Drift (placeholder until task 9.4)",
                (
                    Panel(
                        "What goes here",
                        "Placeholder: what the drift panels will show and how alarms will be "
                        "handled, until task 9.4 replaces it with live per-camera statistics.",
                        kind="text",
                        width=24,
                        height=7,
                        optional=True,
                        content=(
                            "### Drift detection arrives in task 9.4\n\n"
                            "Per camera, against a reference window (tasks 9.1 to 9.3):\n\n"
                            "- **Input:** brightness, blur (Laplacian variance), noise. The "
                            "worker already stores brightness, contrast and blur per frame in "
                            "the `frames` table.\n"
                            "- **Output:** detections per frame, confidence distribution, "
                            "class mix.\n"
                            "- **Alarms:** CUSUM statistic against its threshold, alarms per "
                            "camera, detection delay on the injected scenarios (9.2).\n\n"
                            "The procedure for an alarm is in `docs/RUNBOOK.md` (9.4)."
                        ),
                    ),
                ),
            ),
        ),
    )


def all_dashboards() -> list[Dashboard]:
    return [pipeline(), inference(), query(), resources(), drift()]


# --- Grafana JSON -----------------------------------------------------------------------


def _color(name: str | None) -> dict[str, Any] | None:
    return {"mode": "fixed", "fixedColor": name} if name else None


def _thresholds(panel: Panel) -> dict[str, Any]:
    steps: list[dict[str, Any]] = [{"color": GREEN, "value": None}]
    steps += [{"color": color, "value": value} for value, color in panel.thresholds]
    return {"mode": "absolute", "steps": steps}


def _targets(panel: Panel) -> list[dict[str, Any]]:
    out = []
    for i, t in enumerate(panel.targets):
        target: dict[str, Any] = {
            "datasource": PROM,
            "refId": chr(ord("A") + i),
            "expr": t.expr,
            "legendFormat": t.legend or "__auto",
            "range": True,
        }
        if panel.kind == "heatmap":
            target["format"] = "heatmap"
        out.append(target)
    return out


def _overrides(panel: Panel) -> list[dict[str, Any]]:
    out = [
        {
            "matcher": {"id": "byName", "options": name},
            "properties": [{"id": "color", "value": _color(color)}],
        }
        for name, color in panel.series_colors
    ]
    for t in panel.targets:
        props: list[dict[str, Any]] = []
        color = _color(t.color)
        if color:
            props.append({"id": "color", "value": color})
        if t.right_unit:
            props += [
                {"id": "unit", "value": t.right_unit},
                {"id": "custom.axisPlacement", "value": "right"},
            ]
        if props and t.legend:
            out.append({"matcher": {"id": "byName", "options": t.legend}, "properties": props})
    return out


def panel_json(panel: Panel, panel_id: int, x: int, y: int) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": panel_id,
        "type": panel.kind,
        "title": panel.title,
        "description": panel.description,
        "gridPos": {"x": x, "y": y, "w": panel.width, "h": panel.height},
    }
    if panel.kind == "text":
        base["options"] = {"mode": "markdown", "content": panel.content}
        return base
    base["datasource"] = PROM
    base["targets"] = _targets(panel)
    defaults: dict[str, Any] = {"unit": panel.unit, "thresholds": _thresholds(panel)}
    if panel.min is not None:
        defaults["min"] = panel.min
    if panel.decimals is not None:
        defaults["decimals"] = panel.decimals
    if panel.kind == "timeseries":
        defaults["color"] = {"mode": "palette-classic"}
        defaults["custom"] = {
            "drawStyle": "line",
            "lineWidth": 2,
            "fillOpacity": 25 if panel.stack else 8,
            "showPoints": "never",
            # Gaps stay gaps: a quantile with no traffic is undefined, not the last value.
            "spanNulls": False,
            "stacking": {"mode": "normal" if panel.stack else "none", "group": "A"},
            "thresholdsStyle": {"mode": "dashed" if panel.threshold_line else "off"},
            "axisSoftMin": 0,
        }
        if panel.log_scale:
            defaults["custom"]["scaleDistribution"] = {"type": "log", "log": 10}
            del defaults["min"]  # zero has no place on a log axis
            defaults["custom"].pop("axisSoftMin")
        if panel.soft_max is not None:
            defaults["custom"]["axisSoftMax"] = panel.soft_max
        base["options"] = {
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "multi", "sort": "desc"},
        }
    elif panel.kind == "stat":
        defaults["color"] = {"mode": "thresholds"}
        base["options"] = {
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "colorMode": "background",
            "graphMode": "area",
            "justifyMode": "center",
            "textMode": "value",
            "orientation": "auto",
        }
    elif panel.kind == "heatmap":
        defaults["unit"] = "short"  # cells are observation counts; the y axis carries the unit
        base["options"] = {
            "calculate": False,
            "cellGap": 1,
            "color": {"mode": "scheme", "scheme": "Oranges", "steps": 64},
            "yAxis": {"unit": panel.unit, "decimals": 0},
            "cellValues": {"unit": "short"},
            "rowsFrame": {"layout": "auto"},
            "legend": {"show": True},
            "tooltip": {"mode": "single", "yHistogram": True},
        }
    base["fieldConfig"] = {"defaults": defaults, "overrides": _overrides(panel)}
    return base


def _layout(rows: Sequence[Row]) -> list[dict[str, Any]]:
    """Grafana grid: 24 columns; panels flow left to right and wrap. Rows are row panels; a
    collapsed row carries its panels inside it."""
    out: list[dict[str, Any]] = []
    next_id, y = 1, 0
    for row in rows:
        row_json: dict[str, Any] = {
            "id": next_id,
            "type": "row",
            "title": row.title,
            "collapsed": row.collapsed,
            "gridPos": {"x": 0, "y": y, "w": GRID_WIDTH, "h": 1},
            "panels": [],
        }
        next_id += 1
        y += 1
        out.append(row_json)
        x, line_height = 0, 0
        for panel in row.panels:
            if x + panel.width > GRID_WIDTH:
                x, y, line_height = 0, y + line_height, 0
            item = panel_json(panel, next_id, x, y)
            next_id += 1
            (row_json["panels"] if row.collapsed else out).append(item)
            x += panel.width
            line_height = max(line_height, panel.height)
        y += line_height
    return out


def dashboard_json(d: Dashboard) -> dict[str, Any]:
    return {
        "uid": d.uid,
        "title": d.title,
        "description": d.description,
        "tags": list(d.tags),
        "editable": False,
        "graphTooltip": 1,  # shared crosshair
        "timezone": "browser",
        "time": {"from": "now-15m", "to": "now"},
        "refresh": "5s",
        "schemaVersion": 41,
        "templating": {"list": []},
        "annotations": {"list": []},
        "links": [
            {
                "type": "dashboards",
                "title": "VMS dashboards",
                "tags": ["vms"],
                "asDropdown": True,
                "includeVars": False,
                "keepTime": True,
            }
        ],
        "panels": _layout(d.rows),
    }


def render(d: Dashboard) -> str:
    return json.dumps(dashboard_json(d), indent=2, sort_keys=False) + "\n"


def panel_queries(d: Dashboard, *, include_optional: bool = False) -> Iterator[tuple[str, str]]:
    """(panel title, PromQL) for every query panel, for the telemetry check."""
    for panel in d.panels():
        if panel.optional and not include_optional:
            continue
        for expr in panel.exprs:
            yield panel.title, expr


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vms.observe.dashboards", description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--check", action="store_true", help="fail if the files are stale")
    args = parser.parse_args(argv)
    stale = []
    for d in all_dashboards():
        path = args.out / f"{d.uid.removeprefix('vms-')}.json"
        text = render(d)
        if args.check:
            if not path.exists() or path.read_text() != text:
                stale.append(str(path))
        else:
            args.out.mkdir(parents=True, exist_ok=True)
            path.write_text(text)
            sys.stdout.write(f"wrote {path}\n")
    if stale:
        sys.stderr.write(
            "dashboards out of date; run `make dashboards`:\n  " + "\n  ".join(stale) + "\n"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
