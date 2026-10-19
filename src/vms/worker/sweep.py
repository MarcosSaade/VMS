"""Batch-size sweep (task 5.3): throughput against batch latency from ladder results, the
operating point, and the plot.

A sweep is a `kind: ladder` scenario with one step and many batch sizes
(`bench/scenarios/sweep-*.yaml`), so the numbers come from the same harness as the precision
ladder. The operating point is the smallest batch size whose throughput is within `within` of
the best one and whose batch latency p95 leaves the freshness SLO most of its budget (at most
`latency_share` of it): past that knee, a bigger batch only adds latency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vms.bench.results import RunResult


@dataclass(frozen=True)
class SweepPoint:
    batch: int
    images_per_s: float
    images_per_s_std: float
    latency_ms_p50: float
    latency_ms_p95: float
    cpu_cores: float


def sweep_points(result: RunResult, step: str | None = None) -> list[SweepPoint]:
    """One point per batch size of a ladder result's step (its only step by default)."""
    steps: dict[str, Any] = result.metrics["steps"]
    if step is None:
        if len(steps) != 1:
            raise ValueError(f"{result.name} has steps {sorted(steps)}; name one")
        step = next(iter(steps))
    batches = steps[step]["batch"]
    return sorted(
        (
            SweepPoint(
                batch=int(b),
                images_per_s=v["images_per_s_mean"],
                images_per_s_std=v["images_per_s_std"],
                latency_ms_p50=v["batch_latency_ms"]["p50"],
                latency_ms_p95=v["batch_latency_ms"]["p95"],
                cpu_cores=v["cpu_cores_mean"],
            )
            for b, v in batches.items()
        ),
        key=lambda p: p.batch,
    )


def operating_point(
    points: list[SweepPoint], slo_s: float, within: float = 0.05, latency_share: float = 0.25
) -> SweepPoint:
    """The smallest batch size within `within` of the best throughput whose p95 batch latency
    is at most `latency_share` of the SLO."""
    best = max(p.images_per_s for p in points)
    budget_ms = slo_s * 1000 * latency_share
    for p in points:
        if p.images_per_s >= (1 - within) * best and p.latency_ms_p95 <= budget_ms:
            return p
    fits = [p for p in points if p.latency_ms_p95 <= budget_ms]
    if not fits:
        raise ValueError(f"no batch size has p95 within {budget_ms:.0f} ms")
    return max(fits, key=lambda p: p.images_per_s)


def plot_sweep(series: list[tuple[str, list[SweepPoint]]], slo_s: float, path: Path) -> None:
    """Throughput against batch latency p50 (bars: p95), one line per host, batch sizes
    labeled; the chosen operating point of each host is circled."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#00798c", "#d1495b", "#30638e", "#edae49"]
    fig, ax = plt.subplots(figsize=(7, 4.4), dpi=110)
    for (label, points), color in zip(series, colors, strict=False):
        xs = [p.latency_ms_p50 for p in points]
        ys = [p.images_per_s for p in points]
        ax.errorbar(
            xs,
            ys,
            xerr=np.array(
                [[0.0] * len(points), [p.latency_ms_p95 - p.latency_ms_p50 for p in points]]
            ),
            yerr=[p.images_per_s_std for p in points],
            marker="o",
            ms=4,
            lw=1.5,
            capsize=2,
            color=color,
            label=label,
        )
        for p in points:
            ax.annotate(
                f"{p.batch}",
                (p.latency_ms_p50, p.images_per_s),
                textcoords="offset points",
                xytext=(4, -10),
                fontsize=7,
                color=color,
            )
        op = operating_point(points, slo_s)
        ax.scatter(
            [op.latency_ms_p50],
            [op.images_per_s],
            s=140,
            facecolors="none",
            edgecolors=color,
            linewidths=1.5,
        )
    ax.set_xscale("log")
    from matplotlib.ticker import NullFormatter, ScalarFormatter

    ax.set_xticks([20, 50, 100, 200, 500, 1000])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Batch latency p50 (ms, log; bars to p95)")
    ax.set_ylabel("Throughput (images/s)")
    ax.set_title(
        "YOLO11n, ONNX Runtime FP32 on CPU: throughput vs batch latency\n"
        "(labels: batch size; circled: operating point)",
        fontsize=9,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
