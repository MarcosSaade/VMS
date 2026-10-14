"""`python -m vms.worker bench`: model-only latency of the worker's per-frame work.

Times, frame by frame and in the worker's order, the three stages the worker runs in its
inference thread: JPEG decode, input statistics, and detection (Ultralytics pre-processing,
the forward pass and NMS). No database, object store or HTTP: this is the model-only number;
end-to-end capacity is task 2.5. Warmup frames are excluded; each repetition replays the same
frames, so repetitions differ only by run-to-run noise.

Writes a results JSON (vms.bench.results) under `results_dir`.
"""

from __future__ import annotations

import resource
import statistics
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from vms.bench.results import (
    DEFAULT_PACKAGES,
    RunResult,
    git_info,
    hardware,
    package_versions,
    write_result,
)
from vms.common.ids import new_ulid
from vms.common.logs import get_logger
from vms.common.settings import deep_merge, load_settings
from vms.loadgen.frames import load_clip
from vms.loadgen.settings import FrameSettings
from vms.worker.detector import build_detector
from vms.worker.service import decode_jpeg
from vms.worker.settings import WorkerSettings
from vms.worker.stats import input_stats

log = get_logger(__name__)

PACKAGES = (*DEFAULT_PACKAGES, "torch", "torchvision", "ultralytics", "onnxruntime")
STAGES = ("decode", "stats", "infer")


class BenchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: Path = Path("data")
    clips: list[str] = Field(min_length=1)
    frames_per_clip: int = Field(default=50, ge=1)
    warmup_frames: int = Field(default=10, ge=0)
    results_dir: Path = Path("bench/results")


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    kind: Literal["model"] = "model"
    repeats: int = Field(default=3, ge=1)
    bench: BenchConfig
    config: dict[str, Any] = Field(default_factory=dict)  # overrides of configs/worker.yaml


def load_scenario(path: Path) -> Scenario:
    return Scenario.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def percentiles_ms(values: Sequence[float]) -> dict[str, float]:
    ms = np.asarray(values) * 1000.0
    return {
        "p50": float(np.percentile(ms, 50)),
        "p95": float(np.percentile(ms, 95)),
        "p99": float(np.percentile(ms, 99)),
        "mean": float(ms.mean()),
        "std": float(ms.std(ddof=1)) if len(ms) > 1 else 0.0,
        "max": float(ms.max()),
    }


def _cpu_s() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def run(scenario: Scenario, scenario_raw: dict[str, Any]) -> Path:
    base = load_settings(WorkerSettings, "worker")
    # configs/worker.yaml (+ env), then the scenario's `config:` merged section by section.
    settings = WorkerSettings.model_validate(deep_merge(base.model_dump(), scenario.config))
    detector = build_detector(settings.detector)
    frames: list[bytes] = []
    clip_info: dict[str, Any] = {}
    for clip_id in scenario.bench.clips:
        # As sent by loadgen: the workload's 1280x720 q85 frames (configs/loadgen.yaml).
        clip = load_clip(
            scenario.bench.data_dir,
            clip_id,
            FrameSettings(),
            max_frames=scenario.bench.frames_per_clip,
        )
        frames.extend(clip.frames)
        clip_info[clip_id] = {"frames": len(clip.frames), "size": [clip.width, clip.height]}

    for jpeg in frames[: scenario.bench.warmup_frames]:
        detector.detect(decode_jpeg(jpeg))

    run_id = new_ulid()
    started = datetime.now(UTC)
    reps: list[dict[str, Any]] = []
    for rep in range(scenario.repeats):
        timings: dict[str, list[float]] = {s: [] for s in STAGES}
        n_detections = 0
        cpu0, wall0 = _cpu_s(), time.perf_counter()
        for jpeg in frames:
            t0 = time.perf_counter()
            image = decode_jpeg(jpeg)
            t1 = time.perf_counter()
            input_stats(image)
            t2 = time.perf_counter()
            n_detections += len(detector.detect(image))
            t3 = time.perf_counter()
            timings["decode"].append(t1 - t0)
            timings["stats"].append(t2 - t1)
            timings["infer"].append(t3 - t2)
        wall = time.perf_counter() - wall0
        cpu = _cpu_s() - cpu0
        total = [sum(v) for v in zip(*timings.values(), strict=True)]
        reps.append(
            {
                "frames": len(frames),
                "wall_s": wall,
                "frames_per_s": len(frames) / wall,
                "cpu_cores": cpu / wall,
                "detections_per_frame": n_detections / len(frames),
                "latency_ms": {s: percentiles_ms(v) for s, v in timings.items()}
                | {"total": percentiles_ms(total)},
            }
        )
        log.info(
            "repetition done",
            extra={
                "rep": rep,
                "frames_per_s": round(len(frames) / wall, 2),
                "infer_p50_ms": round(reps[-1]["latency_ms"]["infer"]["p50"], 1),
            },
        )

    fps = [r["frames_per_s"] for r in reps]
    aggregate = {
        "frames_per_s_mean": statistics.fmean(fps),
        "frames_per_s_std": statistics.stdev(fps) if len(fps) > 1 else 0.0,
        "infer_p50_ms_mean": statistics.fmean(r["latency_ms"]["infer"]["p50"] for r in reps),
        "infer_p95_ms_mean": statistics.fmean(r["latency_ms"]["infer"]["p95"] for r in reps),
        "total_p95_ms_mean": statistics.fmean(r["latency_ms"]["total"]["p95"] for r in reps),
        "error_rate": 0.0,
    }
    result = RunResult(
        run_id=run_id,
        name=scenario.name,
        kind="worker.model",
        started_at=started,
        ended_at=datetime.now(UTC),
        seed=settings.seed,
        git=git_info(),
        hardware=hardware(),
        packages=package_versions(PACKAGES),
        config=settings.model_dump(mode="json"),
        scenario=scenario_raw,
        metrics={
            "model_version": detector.model_version,
            "repetitions": reps,
            "aggregate": aggregate,
            "clips": clip_info,
        },
    )
    path = write_result(scenario.bench.results_dir, result)
    log.info("results written", extra={"path": str(path)} | aggregate)
    return path
