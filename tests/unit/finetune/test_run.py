"""Fine-tune scenarios (task 9.5): the committed scenario, training with a fake Ultralytics,
and the before/after pairing with fake eval runs."""

from __future__ import annotations

import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import pytest
import yaml
from pydantic import ValidationError

from vms.bench.results import GitInfo, Hardware, RunResult
from vms.finetune import run as ft

REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIO = REPO_ROOT / "bench/scenarios/finetune-blur-cpu.yaml"


def test_committed_scenario_runs_differ_only_in_blur() -> None:
    scenario, _ = ft.load_scenario(SCENARIO)
    assert scenario.base_weights == "yolo11n"  # the deployed CPU worker's model (D5)
    control = scenario.control
    assert control is not None
    assert control.blur.fraction == 0.0
    treatments = [r for r in scenario.runs if r.role == "treatment"]
    assert [r.blur.fraction for r in treatments] == [0.5]
    assert {c.drift.kind for c in scenario.conditions} >= {"blur"}
    assert scenario.train.close_mosaic >= scenario.train.epochs  # no mosaic at all


def base_raw() -> dict[str, Any]:
    raw = yaml.safe_load(SCENARIO.read_text())
    assert isinstance(raw, dict)
    return raw


def test_scenario_rejects_duplicate_runs_and_two_controls() -> None:
    raw = base_raw()
    raw["runs"] = [raw["runs"][0], raw["runs"][0]]
    with pytest.raises(ValidationError, match="unique"):
        ft.FinetuneScenario.model_validate(raw)
    raw = base_raw()
    raw["runs"] = [
        {"name": "a", "role": "control", "blur": {"fraction": 0}},
        {"name": "b", "role": "control", "blur": {"fraction": 0}},
    ]
    with pytest.raises(ValidationError, match="control"):
        ft.FinetuneScenario.model_validate(raw)


def test_last_epoch_losses(tmp_path: Path) -> None:
    csv = tmp_path / "results.csv"
    csv.write_text(
        "epoch,  train/box_loss,  train/cls_loss,  metrics/mAP50(B)\n"
        "1, 1.5, 2.0, 0.1\n"
        "2, 1.2, 1.7, 0.2\n"
    )
    assert ft.last_epoch_losses(csv) == {"train/box_loss": 1.2, "train/cls_loss": 1.7}
    csv.write_text("epoch,train/box_loss\n")
    assert ft.last_epoch_losses(csv) == {}


class FakeYolo:
    trained: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, path: str, task: str) -> None:
        self.names = {0: "person", 1: "car"}

    def train(self, **kwargs: Any) -> None:
        FakeYolo.trained.append(kwargs)
        out = Path(kwargs["project"]) / kwargs["name"]
        (out / "weights").mkdir(parents=True)
        (out / "weights/last.pt").write_bytes(b"trained " + kwargs["data"].encode())
        (out / "results.csv").write_text("epoch,train/box_loss\n1,0.9\n")


@pytest.fixture
def fake_ultralytics(monkeypatch: pytest.MonkeyPatch) -> type[FakeYolo]:
    module = types.ModuleType("ultralytics")
    module.YOLO = FakeYolo  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    FakeYolo.trained = []
    return FakeYolo


def test_train_run_keeps_the_last_epoch_and_records_it(  # type: ignore[no-untyped-def]
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_ultralytics
) -> None:
    raw = base_raw() | {"results_dir": str(tmp_path / "results")}
    scenario = ft.FinetuneScenario.model_validate(raw)
    monkeypatch.setattr(ft, "load_spec", lambda _p: object())
    monkeypatch.setattr(ft, "read_subset", lambda _p: [object(), object()])
    monkeypatch.setattr(ft, "image_ok", lambda _d, _i: True)
    seen: dict[str, Any] = {}

    def materialize(images, images_dir, out, spec, names, blur, val):  # type: ignore[no-untyped-def]
        seen.update(names=names, blur=blur, val=val)
        out.mkdir(parents=True)
        return {"data_yaml": str(out / "data.yaml"), "blurred": 1}

    monkeypatch.setattr(ft, "materialize", materialize)
    base = tmp_path / "base.pt"
    base.write_bytes(b"base")
    run = scenario.runs[0]
    path, sha, results = ft.train_run(scenario, run, base, tmp_path)

    assert path == tmp_path / "models/finetune/finetune-blur-cpu-blur-mix.pt"
    assert path.read_bytes().startswith(b"trained ")
    assert sha == ft.sha256_file(path)
    assert seen["names"] == {0: "person", 1: "car"}  # the base model's head, kept
    assert seen["blur"] == run.blur
    [args] = fake_ultralytics.trained
    assert args["epochs"] == scenario.train.epochs
    assert args["seed"] == scenario.seed
    assert args["deterministic"] is True
    assert args["plots"] is False
    record = RunResult.model_validate_json(results.read_text())
    assert record.kind == "train.detector"
    assert record.metrics["weights_sha256"] == sha
    assert record.metrics["base_weights_sha256"] == ft.sha256_file(base)
    assert record.metrics["last_epoch"] == {"train/box_loss": 0.9}
    assert record.metrics["images_per_s"] > 0


def result(name: str, metrics: dict[str, Any]) -> RunResult:
    now = datetime.now(UTC)
    return RunResult(
        run_id=name,
        name=name,
        kind="eval.detection",
        started_at=now,
        ended_at=now,
        seed=1,
        git=GitInfo(sha=None, dirty=None),
        hardware=Hardware(cpu="x", logical_cpus=1, ram_bytes=1, platform="p", gpus=[]),
        packages={},
        config={},
        metrics=metrics,
    )


def test_evaluate_pairs_each_run_with_the_base_and_the_treatment_with_the_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = ft.FinetuneScenario.model_validate(base_raw())
    score = {"base": 0.30, "blur-mix": 0.36, "control": 0.33}
    pairs: list[tuple[str, str]] = []

    def run_eval(settings, root, name, detector):  # type: ignore[no-untyped-def]
        label = detector.label
        cond = name.removeprefix(f"{scenario.name}-{label}-")
        value = score[label] - (0.1 if cond != "clean" else 0)
        out = root / f"{name}.json"
        out.write_text(
            result(
                f"{label}|{cond}", {"sources": {"mot17": {"map50_95": {"value": value}}}}
            ).model_dump_json()
        )
        return out

    def compare(base, other, root):  # type: ignore[no-untyped-def]
        pairs.append((base.run_id, other.run_id))
        b = base.metrics["sources"]["mot17"]["map50_95"]["value"]
        o = other.metrics["sources"]["mot17"]["map50_95"]["value"]
        cell = {"value": o - b, "lo": o - b - 0.01, "hi": o - b + 0.01}
        return result(
            "cmp",
            {
                "sources": {
                    "mot17": {
                        "base": {"map50_95": {"value": b}},
                        "other": {"map50_95": {"value": o}},
                        "delta_map50_95": cell,
                        "delta_map50": cell,
                    }
                }
            },
        )

    class FakeDetector:
        def __init__(self, path: Path, sha: str, settings: Any) -> None:
            self.label = path.stem

    import vms.eval.detection.run as eval_run
    import vms.worker.detector as detector_module

    monkeypatch.setattr(eval_run, "run", run_eval)
    monkeypatch.setattr(detector_module, "YoloDetector", FakeDetector)
    monkeypatch.setattr(ft, "compare", compare)
    settings = ft.eval_settings(scenario, REPO_ROOT)
    monkeypatch.setattr(ft, "eval_settings", lambda _s, _r: settings)
    models = {label: (tmp_path / f"{label}.pt", "0" * 64) for label in score}
    out = ft.evaluate_models(scenario, models, tmp_path)

    conds = ["clean", *(c.name for c in scenario.conditions)]
    assert set(out["runs"]) == {"blur-mix", "control"}
    assert out["runs"]["control"]["vs_control"] == {}
    mix = out["runs"]["blur-mix"]
    assert list(mix["vs_base"]) == conds
    assert mix["vs_base"]["blur-6"]["mot17"]["delta_map50_95"]["value"] == pytest.approx(0.06)
    assert mix["vs_control"]["blur-6"]["mot17"]["delta_map50_95"]["value"] == pytest.approx(0.03)
    assert ("control|blur-6", "blur-mix|blur-6") in pairs
    assert out["base"]["clean"]["mot17"] == {"value": 0.30}


def test_evaluate_only_reuses_weights_checked_against_their_newest_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = ft.FinetuneScenario.model_validate(base_raw())
    results = tmp_path / scenario.results_dir
    results.mkdir(parents=True)
    weights = {}
    for k, run in enumerate(scenario.runs):
        w = tmp_path / scenario.artifacts_dir / f"{scenario.name}-{run.name}.pt"
        w.parent.mkdir(parents=True, exist_ok=True)
        w.write_bytes(run.name.encode())
        weights[run.name] = w
        for age, sha in ((0, "stale"), (1, ft.sha256_file(w))):
            record = result(f"{scenario.name}-{run.name}", {}).model_copy(
                update={
                    "kind": "train.detector",
                    "started_at": datetime(2026, 10, 19, 10 + age, k, tzinfo=UTC),
                    "metrics": {
                        "weights": str(w.relative_to(tmp_path)),
                        "weights_sha256": sha,
                        "train_s": 10.0,
                        "images_per_s": 1.0,
                        "dataset": {"blurred": 0},
                    },
                }
            )
            (results / f"2026-10-19-{scenario.name}-{run.name}-{age}.json").write_text(
                record.model_dump_json()
            )
    base = tmp_path / "base.pt"
    base.write_bytes(b"base")
    seen: dict[str, Any] = {}
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/models.yaml").write_text((REPO_ROOT / "configs/models.yaml").read_text())

    def evaluate(scen, models, root):  # type: ignore[no-untyped-def]
        seen.update(models)
        return {"base": {}, "runs": {}, "eval_files": {}}

    import vms.worker.weights as weights_module

    monkeypatch.setattr(weights_module, "ensure_weights", lambda *_a, **_k: base)
    monkeypatch.setattr(ft, "evaluate_models", evaluate)
    settings = ft.eval_settings(scenario, REPO_ROOT)
    monkeypatch.setattr(ft, "eval_settings", lambda _s, _r: settings)
    monkeypatch.setattr(ft, "train_run", lambda *_a: pytest.fail("evaluate must not train"))
    scenario_file = tmp_path / "scenario.yaml"
    scenario_file.write_text(SCENARIO.read_text())
    out = ft.run_finetune(scenario_file, tmp_path, train=False)
    summary = RunResult.model_validate_json(out.read_text())
    assert seen["blur-mix"] == (weights["blur-mix"], ft.sha256_file(weights["blur-mix"]))
    assert summary.metrics["trained"]["control"]["results_file"].endswith("-control-1.json")

    weights["control"].write_bytes(b"retrained elsewhere")
    with pytest.raises(ValueError, match="does not match"):
        ft.run_finetune(scenario_file, tmp_path, train=False)
