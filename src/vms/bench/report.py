"""Render results files as Markdown tables for `docs/BENCHMARKS.md`.

`bench.e2e` results get one row per step (mean and std across repetitions, SLO counts) and the
rated capacity. Other kinds (loadgen, worker model) get their `metrics.aggregate` as a table.
The output is a starting point to paste and trim, not a generated section: BENCHMARKS.md stays
curated, with the methodology and caveats written by hand.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from vms.bench.results import RunResult
from vms.bench.runner import E2EMetrics, StepAggregate

Fmt = Callable[[float], str]


def _num(digits: int) -> Fmt:
    return lambda v: f"{v:,.{digits}f}"


def _cell(agg: StepAggregate, key: str, fmt: Fmt) -> str:
    """Mean (std) across repetitions."""
    s = agg.stats.get(key)
    if not s or s["mean"] is None:
        return "-"
    return fmt(s["mean"]) + (f" ({fmt(s['std'])})" if s["std"] is not None else "")


def _pair(agg: StepAggregate, a: str, b: str, fmt: Fmt) -> str:
    """Two means side by side (e.g. p50 / p95); their stds are in the results file."""

    def mean(key: str) -> str:
        s = agg.stats.get(key)
        return "-" if not s or s["mean"] is None else fmt(s["mean"])

    return f"{mean(a)} / {mean(b)}"


def header(result: RunResult) -> str:
    sha = (result.git.sha or "unknown")[:7] + ("+dirty" if result.git.dirty else "")
    gpus = ", ".join(result.hardware.gpus) or "no GPU"
    return (
        f"`{result.name}` ({result.kind}), run `{result.run_id}`, "
        f"{result.started_at:%Y-%m-%d}, git `{sha}`, {result.hardware.cpu}, {gpus}"
    )


def e2e_table(result: RunResult) -> str:
    metrics = E2EMetrics.model_validate(result.metrics)
    scenario = result.scenario or {}
    lines = [
        header(result),
        "",
        f"Warmup {scenario.get('warmup_s')} s and hold {scenario.get('hold_s')} s per step; "
        "mean (std) across repetitions. Freshness from Prometheus histogram buckets.",
        "",
        "| Cameras | Query clients | Offered fps | Accepted fps | Lost frames "
        "| Freshness p50 / p95 s | Backlog slope fps | Worker CPU cores "
        "| Query p50 / p95 ms | Query/s | Query errors | Ingest SLO met | Harness ok |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    one, two, three = _num(1), _num(2), _num(3)
    for agg in metrics.aggregate:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(agg.cameras or "-"),
                    str(agg.concurrency or "-"),
                    _cell(agg, "loadgen.offered_fps", one),
                    _cell(agg, "loadgen.accepted_fps", one),
                    _cell(agg, "lost_frames", _num(0)),
                    _pair(agg, "prometheus.freshness_p50_s", "prometheus.freshness_p95_s", two),
                    _cell(agg, "prometheus.backlog_slope_fps", two),
                    _cell(agg, "prometheus.cpu_cores.worker", two),
                    _pair(agg, "query.latency_ms.p50", "query.latency_ms.p95", _num(0)),
                    _cell(agg, "query.qps", two),
                    _cell(agg, "query.error_rate", three),
                    f"{agg.ingest_ok_reps}/{agg.reps}" if agg.cameras else "-",
                    f"{agg.harness_ok_reps}/{agg.reps}",
                ]
            )
            + " |"
        )
    rc = metrics.rated_capacity
    if rc is not None:
        lines.append("")
        if rc.cameras is None:
            lines.append(f"Rated capacity: below {rc.first_failure} cameras at {rc.fps:g} FPS.")
        else:
            bound = " (lower bound: the largest step passed)" if rc.lower_bound else ""
            lines.append(f"Rated capacity: {rc.cameras} cameras at {rc.fps:g} FPS{bound}.")
    if metrics.interrupted:
        lines.append("")
        lines.append("Interrupted: not every step ran.")
    return "\n".join(lines)


def generic_table(result: RunResult) -> str:
    agg = result.metrics.get("aggregate")
    lines = [header(result), ""]
    if not isinstance(agg, Mapping) or not agg:
        return "\n".join([*lines, "No aggregate in this file."])
    if all(isinstance(v, Mapping) for v in agg.values()):
        lines += ["| Metric | Mean | Std | Min | Max |", "|---|---|---|---|---|"]
        for key, s in agg.items():
            cells = [_fmt_any(s.get(k)) for k in ("mean", "std", "min", "max")]
            lines.append(f"| {key} | " + " | ".join(cells) + " |")
    else:
        lines += ["| Metric | Value |", "|---|---|"]
        lines += [f"| {key} | {_fmt_any(value)} |" for key, value in agg.items()]
    return "\n".join(lines)


def _fmt_any(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:,.3f}".rstrip("0").rstrip(".")
    return str(value)


def render(result: RunResult) -> str:
    return e2e_table(result) if result.kind == "bench.e2e" else generic_table(result)
