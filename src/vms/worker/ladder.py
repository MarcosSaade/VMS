"""`python -m vms.worker ladder`: speed side of the precision ladder (task 5.2; reused by the
batch sweep in 5.3).

A scenario (`bench/scenarios/ladder-*.yaml`, `kind: ladder`) lists steps, each a detector
configuration (backend, precision), and batch sizes. Each step runs in its own process, so
thread pools (PyTorch's and ONNX Runtime's) never share a process and the peak RSS belongs to
one backend. Repetitions rotate the step order (A B C, B C A, C A B, ...), so slow drift on the
host (thermal throttling of a laptop, background load) spreads over every step instead of
landing on the last one; a cooldown separates steps.

What is timed: `detect_batch` on frames already decoded, i.e. letterbox, forward pass and NMS
for a whole batch, the work that differs between backends. JPEG decode is the same for every
backend and is left out (it is in the worker's own stage metrics). Per batch size, the first
`warmup_batches` batches are run and discarded (ONNX Runtime allocates per input shape). The
last partial batch is dropped so every timed batch has the same size.

Quality: with a `quality:` section, after the speed runs each step is scored on the detection
eval subset (`python -m vms.eval.detection run`, its own process, the step's backend, precision
and device) and the listed pairs are compared with the paired bootstrap and the D32 verdict
(`python -m vms.eval.detection compare`); the result files are listed in the ladder's metrics.
On a GPU session this keeps speed and mAP from the same engines (D38). Without it, score each
step with `make eval SUITE=detection` as on the CPU ladder.
"""

from __future__ import annotations

import json
import os
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vms.common.logs import get_logger
from vms.worker.detector import Detection

log = get_logger(__name__)

PACKAGES = (
    "numpy",
    "torch",
    "torchvision",
    "ultralytics",
    "onnx",
    "onnxruntime",
    "onnxruntime-gpu",
    "tensorrt-cu13",
    "opencv-python",
)
STEP_TIMEOUT_S = 3600.0


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LadderBench(_Model):
    data_dir: Path = Path("data")
    clips: list[str] = Field(min_length=1)
    frames_per_clip: int = Field(default=100, ge=1)
    batch_sizes: list[int] = Field(min_length=1)
    warmup_batches: int = Field(default=2, ge=0)
    cooldown_s: float = Field(default=5.0, ge=0)  # idle time before each step's process
    results_dir: Path = Path("bench/results")

    @model_validator(mode="after")
    def _sizes(self) -> LadderBench:
        if any(b < 1 for b in self.batch_sizes) or len(set(self.batch_sizes)) != len(
            self.batch_sizes
        ):
            raise ValueError("batch_sizes must be distinct and >= 1")
        return self


class LadderStep(_Model):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    detector: dict[str, Any] = Field(default_factory=dict)  # overrides of the detector section


# Detector fields that pick the model and how it runs; the eval keeps its own thresholds
# (conf, iou, max_det from configs/eval-detection.yaml).
QUALITY_FIELDS = ("weights", "backend", "precision", "device", "imgsz")


class LadderQuality(_Model):
    # (base, other) step pairs for the paired comparison; D32 reads "other" against "base".
    compare: list[tuple[str, str]] = Field(default_factory=list)


class LadderScenario(_Model):
    name: str
    description: str = ""
    kind: Literal["ladder"] = "ladder"
    repeats: int = Field(default=3, ge=1)
    bench: LadderBench
    config: dict[str, Any] = Field(default_factory=dict)  # overrides of configs/worker.yaml
    steps: list[LadderStep] = Field(min_length=1)
    quality: LadderQuality | None = None

    @model_validator(mode="after")
    def _unique(self) -> LadderScenario:
        names = [s.name for s in self.steps]
        if len(set(names)) != len(names):
            raise ValueError("step names must be unique")
        for pair in self.quality.compare if self.quality else []:
            unknown = [n for n in pair if n not in names]
            if unknown:
                raise ValueError(f"quality.compare names unknown steps: {unknown}")
        return self

    def step(self, name: str) -> LadderStep:
        for s in self.steps:
            if s.name == name:
                return s
        raise KeyError(f"no step {name!r} in scenario {self.name}")


def load_scenario(path: Path) -> tuple[LadderScenario, dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return LadderScenario.model_validate(raw), raw


def step_order(names: Sequence[str], rep: int) -> list[str]:
    """Repetition `rep`'s order: the list rotated left by `rep`."""
    k = rep % len(names)
    return [*names[k:], *names[:k]]


# --- One step (child process) --------------------------------------------------------------


class BatchDetector(Protocol):
    @property
    def model_version(self) -> str: ...

    def detect_batch(self, images: list[NDArray[np.uint8]]) -> list[list[Detection]]: ...


def percentiles_ms(values_s: Sequence[float]) -> dict[str, float]:
    ms = np.asarray(values_s, dtype=np.float64) * 1000.0
    return {
        "p50": float(np.percentile(ms, 50)),
        "p95": float(np.percentile(ms, 95)),
        "p99": float(np.percentile(ms, 99)),
        "mean": float(ms.mean()),
        "std": float(ms.std(ddof=1)) if len(ms) > 1 else 0.0,
    }


def _cpu_s() -> float:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def measure(
    detector: BatchDetector,
    frames: list[NDArray[np.uint8]],
    batch_sizes: Sequence[int],
    warmup_batches: int,
    clock: Callable[[], float] = time.perf_counter,
    cpu_clock: Callable[[], float] = _cpu_s,
) -> dict[str, Any]:
    """Time `detect_batch` per batch size. Returns raw batch latencies (seconds) and totals."""
    out: dict[str, Any] = {}
    for b in batch_sizes:
        batches = [frames[i : i + b] for i in range(0, len(frames) - b + 1, b)]
        if len(batches) <= warmup_batches:
            raise ValueError(f"batch size {b}: {len(batches)} batches, need > {warmup_batches}")
        for batch in batches[:warmup_batches]:
            detector.detect_batch(batch)
        timed = batches[warmup_batches:]
        latencies: list[float] = []
        detections = 0
        cpu0, wall0 = cpu_clock(), clock()
        for batch in timed:
            t0 = clock()
            results = detector.detect_batch(batch)
            latencies.append(clock() - t0)
            detections += sum(len(r) for r in results)
        wall = clock() - wall0
        images = len(timed) * b
        out[str(b)] = {
            "batches": len(timed),
            "images": images,
            "wall_s": wall,
            "images_per_s": images / wall if wall > 0 else None,
            "cpu_cores": (cpu_clock() - cpu0) / wall if wall > 0 else None,
            "detections_per_image": detections / images,
            "batch_latency_s": latencies,
        }
    return out


def step_settings(scenario: LadderScenario, step: LadderStep) -> Any:
    """configs/worker.yaml (+ env), then the scenario's `config:`, then the step's detector."""
    from vms.common.settings import deep_merge, load_settings
    from vms.worker.settings import WorkerSettings

    base = load_settings(WorkerSettings, "worker").model_dump()
    merged = deep_merge(deep_merge(base, scenario.config), {"detector": step.detector})
    return WorkerSettings.model_validate(merged)


def load_frames(bench: LadderBench) -> tuple[list[NDArray[np.uint8]], dict[str, Any]]:
    """The clips' frames as loadgen sends them (1280x720 q85), decoded."""
    from vms.loadgen.frames import load_clip
    from vms.loadgen.settings import FrameSettings
    from vms.worker.service import decode_jpeg

    frames: list[NDArray[np.uint8]] = []
    info: dict[str, Any] = {}
    for clip_id in bench.clips:
        clip = load_clip(bench.data_dir, clip_id, FrameSettings(), max_frames=bench.frames_per_clip)
        frames.extend(decode_jpeg(j) for j in clip.frames)
        info[clip_id] = {"frames": len(clip.frames), "size": [clip.width, clip.height]}
    return frames, info


def run_step(scenario: LadderScenario, step: LadderStep) -> dict[str, Any]:
    from vms.worker.detector import build_detector

    settings = step_settings(scenario, step)
    t0 = time.perf_counter()
    detector = build_detector(settings.detector)
    load_s = time.perf_counter() - t0
    frames, clips = load_frames(scenario.bench)
    batches = measure(detector, frames, scenario.bench.batch_sizes, scenario.bench.warmup_batches)
    return {
        "step": step.name,
        "model_version": detector.model_version,
        "detector": settings.detector.model_dump(mode="json"),
        "load_s": load_s,
        # ru_maxrss is in KiB on Linux: the step's whole process, model and frames included.
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "clips": clips,
        "batch": batches,
    }


def run_step_main(scenario_path: Path, step_name: str, out: Path) -> int:
    scenario, _ = load_scenario(scenario_path)
    result = run_step(scenario, scenario.step(step_name))
    out.write_text(json.dumps(result), encoding="utf-8")
    return 0


# --- The ladder (parent process) -----------------------------------------------------------

StepRunner = Callable[[Path, str], dict[str, Any]]


def subprocess_step(scenario_path: Path, step_name: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "step.json"
        cmd = [sys.executable, "-m", "vms.worker", "ladder-step"]
        cmd += ["--scenario", str(scenario_path), "--step", step_name, "--out", str(out)]
        subprocess.run(cmd, check=True, timeout=STEP_TIMEOUT_S)
        result: dict[str, Any] = json.loads(out.read_text(encoding="utf-8"))
        return result


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """One step's repetitions -> per batch size: pooled batch-latency percentiles, per-image
    latency, and throughput mean and standard deviation across repetitions."""
    first = runs[0]
    out: dict[str, Any] = {
        "model_version": first["model_version"],
        "detector": first["detector"],
        "repetitions": len(runs),
        "peak_rss_mb_max": max(r["peak_rss_mb"] for r in runs),
        "load_s_mean": statistics.fmean(r["load_s"] for r in runs),
        "batch": {},
    }
    for b in first["batch"]:
        per = [r["batch"][b] for r in runs]
        pooled = [x for p in per for x in p["batch_latency_s"]]
        ips = [p["images_per_s"] for p in per]
        lat = percentiles_ms(pooled)
        out["batch"][b] = {
            "timed_batches": len(pooled),
            "batch_latency_ms": lat,
            "per_image_ms_p50": lat["p50"] / int(b),
            "images_per_s_mean": statistics.fmean(ips),
            "images_per_s_std": statistics.stdev(ips) if len(ips) > 1 else 0.0,
            "images_per_s_reps": ips,
            "rep_p50_ms": [percentiles_ms(p["batch_latency_s"])["p50"] for p in per],
            "cpu_cores_mean": statistics.fmean(p["cpu_cores"] for p in per),
            "detections_per_image": statistics.fmean(p["detections_per_image"] for p in per),
        }
    return out


def ladder(
    scenario: LadderScenario,
    scenario_path: Path,
    runner: StepRunner = subprocess_step,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[dict[str, Any], list[list[str]]]:
    """Run every step `repeats` times. A step whose process fails is recorded with its error
    and left out of later repetitions, so one broken backend does not cost the others' numbers
    (on a GPU session, the setup time already paid)."""
    names = [s.name for s in scenario.steps]
    runs: dict[str, list[dict[str, Any]]] = {n: [] for n in names}
    errors: dict[str, str] = {}
    orders = []
    for rep in range(scenario.repeats):
        order = [n for n in step_order(names, rep) if n not in errors]
        orders.append(order)
        for name in order:
            sleep(scenario.bench.cooldown_s)
            log.info("ladder step", extra={"rep": rep, "step": name})
            try:
                runs[name].append(runner(scenario_path, name))
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                log.error("ladder step failed", extra={"rep": rep, "step": name, "error": str(exc)})
                errors[name] = f"repetition {rep}: {exc}"
    steps = {n: aggregate(runs[n]) if n not in errors else {"error": errors[n]} for n in names}
    return steps, orders


# (argv, extra environment) -> stdout; raises on a non-zero exit.
CommandRunner = Callable[[list[str], dict[str, str]], str]


def subprocess_command(argv: list[str], env: dict[str, str]) -> str:
    done = subprocess.run(
        argv,
        env={**os.environ, **env},
        check=True,
        timeout=STEP_TIMEOUT_S,
        stdout=subprocess.PIPE,
        text=True,
    )
    return done.stdout


def _results_path(stdout: str) -> str:
    for line in reversed(stdout.splitlines()):
        if line.startswith("results: "):
            return line.removeprefix("results: ").strip()
    raise RuntimeError(f"no 'results:' line in the eval output: {stdout[-500:]}")


def detector_env(detector: dict[str, Any]) -> dict[str, str]:
    """VMS_DETECTOR__* overrides that make the eval run this step's model the same way."""
    return {f"VMS_DETECTOR__{k.upper()}": str(detector[k]) for k in QUALITY_FIELDS if k in detector}


def run_quality(
    scenario: LadderScenario,
    steps: dict[str, dict[str, Any]],
    runner: CommandRunner = subprocess_command,
) -> dict[str, Any]:
    """Score every step that ran on the detection subset, then compare the listed pairs whose
    steps both have a score. Failures are recorded, not raised."""
    assert scenario.quality is not None
    py = [sys.executable, "-m", "vms.eval.detection"]
    evals: dict[str, str] = {}
    errors: dict[str, str] = {}
    for step in scenario.steps:
        if "error" in steps[step.name]:
            continue
        name = f"{scenario.name}-{step.name}"
        log.info("ladder quality", extra={"step": step.name})
        try:
            out = runner([*py, "run", "--name", name], detector_env(steps[step.name]["detector"]))
            evals[step.name] = _results_path(out)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError) as exc:
            errors[f"eval {step.name}"] = str(exc)
    compares = []
    for base, other in scenario.quality.compare:
        if base not in evals or other not in evals:
            continue
        name = f"{scenario.name}-{other}-vs-{base}"
        try:
            out = runner([*py, "compare", evals[base], evals[other], "--name", name], {})
            compares.append({"base": base, "other": other, "results": _results_path(out)})
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, RuntimeError) as exc:
            errors[f"compare {other} vs {base}"] = str(exc)
    return {"eval": evals, "compare": compares, "errors": errors}


def run_ladder(scenario_path: Path) -> Path:
    from vms.bench.results import RunResult, git_info, hardware, package_versions, write_result
    from vms.common.ids import new_ulid

    scenario, raw = load_scenario(scenario_path)
    run_id = new_ulid()
    started = datetime.now(UTC)
    steps, orders = ladder(scenario, scenario_path)
    quality = run_quality(scenario, steps) if scenario.quality else None
    failed = [n for n, s in steps.items() if "error" in s]
    failed += list((quality or {}).get("errors", {}))
    settings = step_settings(scenario, scenario.steps[0])
    result = RunResult(
        run_id=run_id,
        name=scenario.name,
        kind="worker.ladder",
        started_at=started,
        ended_at=datetime.now(UTC),
        seed=settings.seed,
        git=git_info(),
        hardware=hardware(),
        packages=package_versions(PACKAGES),
        config={
            "worker": settings.model_dump(mode="json"),
            "steps": {n: s["detector"] for n, s in steps.items()},
        },
        scenario=raw,
        metrics={"steps": steps, "orders": orders, "quality": quality, "failed": failed},
        notes="Model-only: detect_batch (letterbox, forward, NMS) on decoded frames; JPEG "
        "decode excluded. Latency percentiles pool every timed batch of every repetition; "
        "throughput is the mean and std of per-repetition values. Each step ran in its own "
        "process, step order rotated per repetition.",
    )
    path = write_result(scenario.bench.results_dir, result)
    log.info("ladder results written", extra={"path": str(path), "failed": failed})
    if failed:
        raise LadderFailedError(f"results written to {path}, but these failed: {failed}")
    return path


class LadderFailedError(RuntimeError):
    """Some steps or quality runs failed; the results file has the rest and the errors."""
