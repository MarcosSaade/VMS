"""Precision ladder harness (task 5.2): scenario validation, rotated step order, batch timing
with an injected clock, and aggregation across repetitions. Each step's real run (a subprocess
with a real backend) is replaced by a fake runner."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from vms.worker.detector import Detection
from vms.worker.ladder import (
    LadderScenario,
    aggregate,
    ladder,
    measure,
    step_order,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def scenario(**overrides: Any) -> LadderScenario:
    raw: dict[str, Any] = {
        "name": "t",
        "kind": "ladder",
        "repeats": 3,
        "bench": {"clips": ["c"], "batch_sizes": [1, 2], "warmup_batches": 1, "cooldown_s": 2},
        "steps": [{"name": "a"}, {"name": "b"}, {"name": "c", "detector": {"backend": "x"}}],
    }
    return LadderScenario.model_validate(raw | overrides)


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class FakeDetector:
    """Each batch costs 0.01 s per image plus 0.005 s overhead; one detection per image."""

    model_version = "fake"

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls: list[int] = []

    def detect_batch(self, images: list[NDArray[np.uint8]]) -> list[list[Detection]]:
        self.calls.append(len(images))
        self.clock.now += 0.005 + 0.01 * len(images)
        return [[Detection("person", 0.9, 0, 0, 1, 1)] for _ in images]


def test_scenario_rejects_duplicate_steps_and_bad_batch_sizes() -> None:
    with pytest.raises(ValidationError, match="unique"):
        scenario(steps=[{"name": "a"}, {"name": "a"}])
    with pytest.raises(ValidationError, match="distinct"):
        scenario(bench={"clips": ["c"], "batch_sizes": [4, 4]})
    with pytest.raises(KeyError):
        scenario().step("zzz")


def test_committed_scenario_is_valid() -> None:
    import yaml

    raw = yaml.safe_load((REPO_ROOT / "bench/scenarios/ladder-cpu.yaml").read_text())
    s = LadderScenario.model_validate(raw)
    assert [x.name for x in s.steps] == ["pytorch-fp32", "ort-fp32", "ort-int8"]


def test_each_repetition_rotates_the_step_order() -> None:
    names = ["a", "b", "c"]
    assert [step_order(names, r) for r in range(4)] == [
        ["a", "b", "c"],
        ["b", "c", "a"],
        ["c", "a", "b"],
        ["a", "b", "c"],
    ]


def test_measure_times_full_batches_after_warmup() -> None:
    clock = FakeClock()
    detector = FakeDetector(clock)
    frames = [np.zeros((2, 2, 3), dtype=np.uint8)] * 9
    out = measure(detector, frames, [1, 4], warmup_batches=1, clock=clock, cpu_clock=clock)
    # bs 1: 9 batches, 1 warmup; bs 4: 2 full batches (the 9th frame dropped), 1 warmup.
    assert detector.calls == [1] * 9 + [4, 4]
    assert out["1"]["batches"] == 8
    assert out["1"]["batch_latency_s"] == pytest.approx([0.015] * 8)
    assert out["1"]["images_per_s"] == pytest.approx(1 / 0.015)
    assert out["4"]["images"] == 4
    assert out["4"]["batch_latency_s"] == pytest.approx([0.045])
    assert out["4"]["detections_per_image"] == 1.0
    with pytest.raises(ValueError, match="need > 1"):
        measure(detector, frames[:1], [1], warmup_batches=1, clock=clock, cpu_clock=clock)


def fake_run(ms_per_batch: float, rss: float) -> dict[str, Any]:
    return {
        "step": "a",
        "model_version": "m",
        "detector": {"backend": "x"},
        "load_s": 1.0,
        "peak_rss_mb": rss,
        "batch": {
            "2": {
                "images_per_s": 2 / (ms_per_batch / 1000),
                "cpu_cores": 4.0,
                "detections_per_image": 3.0,
                "batch_latency_s": [ms_per_batch / 1000] * 4,
            }
        },
    }


def test_aggregate_pools_latencies_and_spreads_throughput() -> None:
    agg = aggregate([fake_run(10, 500), fake_run(20, 600)])
    b = agg["batch"]["2"]
    assert agg["repetitions"] == 2
    assert agg["peak_rss_mb_max"] == 600
    assert b["timed_batches"] == 8
    assert b["batch_latency_ms"]["p50"] == pytest.approx(15.0)
    assert b["per_image_ms_p50"] == pytest.approx(7.5)
    assert b["images_per_s_reps"] == pytest.approx([200.0, 100.0])
    assert b["images_per_s_mean"] == pytest.approx(150.0)
    assert b["images_per_s_std"] == pytest.approx(70.71, abs=0.01)
    assert b["rep_p50_ms"] == pytest.approx([10.0, 20.0])


def test_ladder_runs_every_step_each_repetition_with_a_cooldown() -> None:
    calls: list[str] = []
    sleeps: list[float] = []

    def runner(path: Path, name: str) -> dict[str, Any]:
        calls.append(name)
        return fake_run(10, 100)

    steps, orders = ladder(scenario(), Path("s.yaml"), runner=runner, sleep=sleeps.append)
    assert calls == ["a", "b", "c", "b", "c", "a", "c", "a", "b"]
    assert orders[1] == ["b", "c", "a"]
    assert sleeps == [2.0] * 9
    assert set(steps) == {"a", "b", "c"}
    assert steps["a"]["repetitions"] == 3


def test_quality_scores_every_step_then_compares_the_pairs() -> None:
    from vms.worker.ladder import detector_env, run_quality

    s = scenario(
        steps=[
            {"name": "ort", "detector": {"backend": "onnxruntime"}},
            {"name": "trt", "detector": {"backend": "tensorrt", "precision": "fp16"}},
        ],
        quality={"compare": [["ort", "trt"]]},
    )
    steps = {
        "ort": {
            "detector": {
                "weights": "yolo11n",
                "backend": "onnxruntime",
                "precision": "fp32",
                "device": "cuda",
                "imgsz": 640,
                "conf": 0.1,
            }
        },
        "trt": {
            "detector": {
                "weights": "yolo11n",
                "backend": "tensorrt",
                "precision": "fp16",
                "device": "cuda",
                "imgsz": 640,
                "conf": 0.1,
            }
        },
    }
    calls: list[tuple[list[str], dict[str, str]]] = []

    def runner(argv: list[str], env: dict[str, str]) -> str:
        calls.append((argv, env))
        return f"log line\nresults: eval/results/{argv[-1]}.json\n"

    out = run_quality(s, steps, runner)
    assert [c[0][3] for c in calls] == ["run", "run", "compare"]
    assert calls[1][1] == {
        "VMS_DETECTOR__WEIGHTS": "yolo11n",
        "VMS_DETECTOR__BACKEND": "tensorrt",
        "VMS_DETECTOR__PRECISION": "fp16",
        "VMS_DETECTOR__DEVICE": "cuda",
        "VMS_DETECTOR__IMGSZ": "640",
    }  # the eval keeps its own thresholds: no conf
    assert calls[2][0][4:6] == [out["eval"]["ort"], out["eval"]["trt"]]
    assert out["compare"][0]["results"].endswith("-trt-vs-ort.json")
    assert detector_env({"backend": "onnxruntime"}) == {"VMS_DETECTOR__BACKEND": "onnxruntime"}


def test_quality_pairs_must_name_steps() -> None:
    with pytest.raises(ValidationError, match="unknown steps"):
        scenario(quality={"compare": [["pytorch", "nope"]]})


def test_gpu_scenarios_are_valid() -> None:
    import yaml

    for name in ("ladder-gpu-n", "sweep-gpu-trt-fp16-n", "sweep-gpu-trt-int8-n"):
        raw = yaml.safe_load((REPO_ROOT / f"bench/scenarios/{name}.yaml").read_text())
        s = LadderScenario.model_validate(raw)
        assert s.config["detector"]["device"] == "cuda"
    ladder_raw = yaml.safe_load((REPO_ROOT / "bench/scenarios/ladder-gpu-n.yaml").read_text())
    q = LadderScenario.model_validate(ladder_raw).quality
    assert q is not None
    assert ("trt-fp16", "trt-int8") in q.compare  # D32: INT8 against TensorRT FP16
