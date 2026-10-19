"""Batch-size sweep (task 5.3): points from a ladder result, the operating point, the plot."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from vms.bench.results import GitInfo, Hardware, RunResult
from vms.worker.sweep import SweepPoint, operating_point, plot_sweep, sweep_points


def ladder_result(batches: dict[str, tuple[float, float]], steps: int = 1) -> RunResult:
    """`batches`: batch size -> (images/s, p95 ms); p50 is 80% of p95."""
    step = {
        "batch": {
            b: {
                "images_per_s_mean": ips,
                "images_per_s_std": 1.0,
                "batch_latency_ms": {"p50": 0.8 * p95, "p95": p95},
                "cpu_cores_mean": 4.0,
            }
            for b, (ips, p95) in batches.items()
        }
    }
    now = datetime.now(UTC)
    return RunResult(
        run_id="r",
        name="sweep",
        kind="worker.ladder",
        started_at=now,
        ended_at=now,
        seed=1,
        git=GitInfo(sha=None, dirty=None),
        hardware=Hardware(cpu="cpu", logical_cpus=8, ram_bytes=1, platform="p", gpus=[]),
        packages={},
        config={},
        metrics={"steps": {f"s{i}": step for i in range(steps)}},
    )


SWEEP = {
    "1": (40.0, 30.0),
    "2": (52.0, 45.0),
    "4": (60.0, 80.0),
    "8": (62.0, 150.0),
    "16": (62.5, 300.0),
    "32": (62.0, 640.0),
}


def test_points_are_sorted_by_batch_size() -> None:
    points = sweep_points(ladder_result(SWEEP))
    assert [p.batch for p in points] == [1, 2, 4, 8, 16, 32]
    assert points[2] == SweepPoint(4, 60.0, 1.0, 64.0, 80.0, 4.0)
    with pytest.raises(ValueError, match="name one"):
        sweep_points(ladder_result(SWEEP, steps=2))


def test_operating_point_is_the_knee_within_the_latency_budget() -> None:
    points = sweep_points(ladder_result(SWEEP))
    # 4 is within 5% of the best (62.5) at p95 80 ms; 2 is not; past 4 only latency grows.
    assert operating_point(points, slo_s=2.0).batch == 4
    # A tight budget (25% of 0.4 s = 100 ms) still fits 4; of 0.2 s (50 ms) only 1 and 2.
    assert operating_point(points, slo_s=0.4).batch == 4
    assert operating_point(points, slo_s=0.2).batch == 2
    with pytest.raises(ValueError, match="no batch size"):
        operating_point(points, slo_s=0.1)


def test_plot_is_written(tmp_path: Path) -> None:
    points = sweep_points(ladder_result(SWEEP))
    out = tmp_path / "sweep.png"
    plot_sweep([("laptop", points), ("cloud", points)], 2.0, out)
    assert out.stat().st_size > 1000
