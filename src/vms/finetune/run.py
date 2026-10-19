"""`kind: finetune` scenarios (task 9.5): train each run, then score every model before and
after on the clean and drifted eval subset.

A scenario (`bench/scenarios/finetune-*.yaml`) names the frozen training list, the base
weights, one fixed training configuration and the runs, which differ only in the share of
blurred training images (a blur-mix run, and a control with none, so the gain from blur can be
told apart from the gain from training on camera scenes at all). Per run: materialize its YOLO
dataset, fine-tune with Ultralytics from the base weights, keep the last epoch (no selection
on a validation set), copy the weights to `artifacts_dir` with their sha256, and write a
results JSON (`kind: train.detector`) with the training time and losses.

Then every model (base and each run) is scored with the detection eval (3.4) on the frozen
subset, clean and under each condition, and each run is compared with the base, and the
blur-mix run with the control, by the paired bootstrap of `vms.eval.detection.compare` (same
images, same resamples). A summary JSON (`kind: train.finetune`) holds the table.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vms.bench.results import RunResult, git_info, hardware, package_versions, write_result
from vms.common.ids import new_ulid
from vms.common.logs import get_logger
from vms.common.settings import deep_merge, load_settings
from vms.eval.detection.compare import compare, load_run
from vms.eval.detection.drift import Condition
from vms.eval.detection.settings import DetectionEvalSettings
from vms.eval.detection.subset import image_ok, read_subset
from vms.finetune.data import Blur, load_spec, materialize

log = get_logger(__name__)

KIND = "train.finetune"
PACKAGES = ("numpy", "torch", "torchvision", "ultralytics", "opencv-python")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrainConfig(_Model):
    """Ultralytics `train` arguments shared by every run (one fixed configuration, no sweep)."""

    epochs: int = Field(ge=1)
    imgsz: int = Field(default=640, ge=32, multiple_of=32)
    batch: int = Field(default=16, ge=1)
    workers: int = Field(default=0, ge=0)
    close_mosaic: int = Field(default=10, ge=0)
    val_images: int = Field(default=64, ge=1)
    extra: dict[str, Any] = Field(default_factory=dict)  # any other train() argument


class FinetuneRun(_Model):
    name: str = Field(pattern=r"^[a-z0-9][a-z0-9-]*$")
    blur: Blur
    role: Literal["treatment", "control"] = "treatment"


class FinetuneScenario(_Model):
    name: str
    description: str = ""
    kind: Literal["finetune"] = "finetune"
    seed: int
    data_spec: Path
    data_lock: Path
    images_dir: Path
    dataset_dir: Path  # materialized datasets and Ultralytics run folders (gitignored)
    artifacts_dir: Path  # fine-tuned weights (DVC-tracked)
    results_dir: Path = Path("bench/results")
    base_weights: str
    train: TrainConfig
    runs: list[FinetuneRun] = Field(min_length=1)
    eval: dict[str, Any] = Field(default_factory=dict)  # overrides of eval-detection.yaml
    conditions: list[Condition] = Field(min_length=1)

    @model_validator(mode="after")
    def _check(self) -> FinetuneScenario:
        names = [r.name for r in self.runs]
        if len(set(names)) != len(names):
            raise ValueError("run names must be unique")
        if sum(r.role == "control" for r in self.runs) > 1:
            raise ValueError("at most one control run")
        return self

    @property
    def control(self) -> FinetuneRun | None:
        return next((r for r in self.runs if r.role == "control"), None)


def load_scenario(path: Path) -> tuple[FinetuneScenario, dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return FinetuneScenario.model_validate(raw), raw


def _rel(path: Path, root: Path) -> str:
    """`path` relative to the repo root for results files; absolute when outside it."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def last_epoch_losses(results_csv: Path) -> dict[str, float]:
    """The last row of Ultralytics' results.csv, train losses only (column names stripped)."""
    with results_csv.open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}
    last = {k.strip(): v for k, v in rows[-1].items()}
    return {k: float(v) for k, v in last.items() if k.startswith("train/")}


def train_run(
    scenario: FinetuneScenario, run: FinetuneRun, base_path: Path, root: Path
) -> tuple[Path, str, Path]:
    """Materialize the run's dataset, fine-tune, copy the weights; returns (weights path,
    sha256, train results JSON path)."""
    import importlib

    import torch

    yolo: Any = importlib.import_module("ultralytics").YOLO
    spec = load_spec(root / scenario.data_spec)
    images = read_subset(root / scenario.data_lock)
    missing = [i.id for i in images if not image_ok(root / scenario.images_dir, i)]
    if missing:
        raise SystemExit(f"{len(missing)} training images missing; run `make finetune-data`")
    started = datetime.now(UTC)
    model = yolo(str(base_path), task="detect")
    names = {int(k): str(v) for k, v in model.names.items()}
    out = root / scenario.dataset_dir / scenario.name / run.name
    if out.exists():
        shutil.rmtree(out)
    t0 = time.perf_counter()
    dataset = materialize(
        images,
        root / scenario.images_dir,
        out / "dataset",
        spec,
        names,
        run.blur,
        scenario.train.val_images,
    )
    materialize_s = time.perf_counter() - t0
    t = scenario.train
    args = {
        "data": dataset["data_yaml"],
        "epochs": t.epochs,
        "imgsz": t.imgsz,
        "batch": t.batch,
        "workers": t.workers,
        "close_mosaic": t.close_mosaic,
        "device": "cpu" if not torch.cuda.is_available() else 0,
        "seed": scenario.seed,
        "deterministic": True,
        "plots": False,
        "val": False,  # Ultralytics still validates once after the last epoch (sanity only)
        "project": str((out / "ultralytics").resolve()),
        "name": "train",
        "exist_ok": True,
        "verbose": False,
        **t.extra,
    }
    log.info("training", extra={"run": run.name, "epochs": t.epochs, "images": len(images)})
    t1 = time.perf_counter()
    model.train(**args)
    train_s = time.perf_counter() - t1
    run_dir = out / "ultralytics/train"
    target = root / scenario.artifacts_dir / f"{scenario.name}-{run.name}.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run_dir / "weights/last.pt", target)
    sha = sha256_file(target)
    samples = len(images) * t.epochs * float(t.extra.get("fraction", 1.0))
    result = RunResult(
        run_id=new_ulid(),
        name=f"{scenario.name}-{run.name}",
        kind="train.detector",
        started_at=started,
        ended_at=datetime.now(UTC),
        seed=scenario.seed,
        git=git_info(root),
        hardware=hardware(),
        packages=package_versions(PACKAGES),
        config={"train_args": args, "run": run.model_dump(), "base_weights": scenario.base_weights},
        scenario=scenario.model_dump(mode="json"),
        metrics={
            "weights": _rel(target, root),
            "weights_sha256": sha,
            "base_weights_sha256": sha256_file(base_path),
            "dataset": dataset,
            "materialize_s": materialize_s,
            "train_s": train_s,
            "images_per_s": samples / train_s if train_s else None,
            "last_epoch": last_epoch_losses(run_dir / "results.csv"),
            "torch_threads": torch.get_num_threads(),
        },
        notes="Fine-tuned from the base weights; the last epoch is kept. The validation "
        "readout Ultralytics prints is on training images and is not a reported number. "
        "train_s (and images_per_s) include its one validation pass on val_images images.",
    )
    path = write_result(root / scenario.results_dir, result)
    log.info("trained", extra={"run": run.name, "train_s": round(train_s), "path": str(path)})
    return target, sha, path


def eval_settings(scenario: FinetuneScenario, root: Path) -> DetectionEvalSettings:
    base = load_settings(DetectionEvalSettings, "eval-detection", config_dir=root / "configs")
    merged = deep_merge(base.model_dump(), scenario.eval)
    merged = deep_merge(merged, {"detector": {"weights": scenario.base_weights}})
    settings = DetectionEvalSettings.model_validate(merged)
    if settings.max_images is not None:
        raise ValueError("a fine-tune eval scores the whole subset (max_images must be null)")
    return settings


def _cell(paired: RunResult) -> dict[str, Any]:
    return {
        source: {
            "base_map50_95": s["base"]["map50_95"]["value"],
            "map50_95": s["other"]["map50_95"],
            "delta_map50_95": s["delta_map50_95"],
            "delta_map50": s["delta_map50"],
        }
        for source, s in paired.metrics["sources"].items()
    }


def evaluate_models(
    scenario: FinetuneScenario,
    models: dict[str, tuple[Path, str]],
    root: Path,
) -> dict[str, Any]:
    """Score every model clean and under each condition; pair each run with the base, and the
    treatment runs with the control. `models` maps a label to (weights path, sha256); the base
    is the label `base`."""
    from vms.eval.detection.run import run as run_eval
    from vms.worker.detector import YoloDetector

    settings = eval_settings(scenario, root)
    conditions: list[tuple[str, Any]] = [("clean", None)] + [
        (c.name, c.drift) for c in scenario.conditions
    ]
    runs: dict[str, dict[str, RunResult]] = {}
    files: dict[str, dict[str, str]] = {}
    for label, (path, sha) in models.items():
        det_settings = (
            settings.detector
            if label == "base"
            else settings.detector.model_copy(update={"weights": f"{scenario.name}-{label}"})
        )
        detector = YoloDetector(path, sha, det_settings)
        runs[label], files[label] = {}, {}
        for cond, drift in conditions:
            out = run_eval(
                settings.model_copy(update={"drift": drift, "detector": det_settings}),
                root,
                name=f"{scenario.name}-{label}-{cond}",
                detector=detector,
            )
            runs[label][cond] = load_run(out)
            files[label][cond] = _rel(out, root)
            log.info("scored", extra={"model": label, "condition": cond})
    table: dict[str, Any] = {}
    control = scenario.control
    for run in scenario.runs:
        table[run.name] = {"vs_base": {}, "vs_control": {}}
        for cond, _ in conditions:
            paired = compare(runs["base"][cond], runs[run.name][cond], root)
            table[run.name]["vs_base"][cond] = _cell(paired)
            if control is not None and run.name != control.name:
                paired = compare(runs[control.name][cond], runs[run.name][cond], root)
                table[run.name]["vs_control"][cond] = _cell(paired)
    base_scores = {
        cond: {s: v["map50_95"] for s, v in r.metrics["sources"].items()}
        for cond, r in runs["base"].items()
    }
    return {"base": base_scores, "runs": table, "eval_files": files}


def latest_train_result(scenario: FinetuneScenario, run: FinetuneRun, root: Path) -> Path:
    """The newest `train.detector` results file of a run (for `evaluate` without training)."""
    pattern = f"*-{scenario.name}-{run.name}-*.json"
    found = sorted(
        (p for p in (root / scenario.results_dir).glob(pattern)),
        key=lambda p: load_run(p).started_at,
    )
    found = [p for p in found if load_run(p).kind == "train.detector"]
    if not found:
        raise FileNotFoundError(
            f"no train.detector results for {run.name} in {scenario.results_dir}"
        )
    return found[-1]


def run_finetune(scenario_path: Path, root: Path = Path(), *, train: bool = True) -> Path:
    """Train every run (unless `train` is False: reuse the weights in `artifacts_dir`, checked
    against their newest training record), then score and write the summary."""
    from vms.worker.weights import ensure_weights, load_models_config

    scenario, raw = load_scenario(scenario_path)
    started = datetime.now(UTC)
    models_config = load_models_config(root / "configs/models.yaml")
    base_path = ensure_weights(models_config, scenario.base_weights, root=root)
    base_sha = models_config.weights[scenario.base_weights].sha256
    models: dict[str, tuple[Path, str]] = {"base": (base_path, base_sha)}
    trained: dict[str, Any] = {}
    for run in scenario.runs:
        if train:
            path, sha, results = train_run(scenario, run, base_path, root)
        else:
            results = latest_train_result(scenario, run, root)
            path = root / load_run(results).metrics["weights"]
            sha = sha256_file(path)
            if sha != load_run(results).metrics["weights_sha256"]:
                raise ValueError(f"{path} does not match its training record {results}")
        models[run.name] = (path, sha)
        record = load_run(results)
        trained[run.name] = {
            "weights": _rel(path, root),
            "sha256": sha,
            "results_file": _rel(results, root),
            "train_s": record.metrics["train_s"],
            "images_per_s": record.metrics["images_per_s"],
            "blurred": record.metrics["dataset"]["blurred"],
        }
    scores = evaluate_models(scenario, models, root)
    result = RunResult(
        run_id=new_ulid(),
        name=scenario.name,
        kind=KIND,
        started_at=started,
        ended_at=datetime.now(UTC),
        seed=scenario.seed,
        git=git_info(root),
        hardware=hardware(),
        packages=package_versions(PACKAGES),
        config=eval_settings(scenario, root).model_dump(mode="json"),
        scenario=raw,
        metrics={
            "base": {"weights": scenario.base_weights, "sha256": base_sha},
            "trained": trained,
            **scores,
        },
        notes="delta = fine-tuned - base (vs_base) or treatment - control (vs_control), mAP "
        "points as fractions, paired percentile bootstrap over images. Blur is a synthetic "
        "augmentation, not real defocus or fog; the MOT17 gain includes adapting to MOT's "
        "amodal box convention, which the control run isolates.",
    )
    path = write_result(root / scenario.results_dir, result)
    log.info("fine-tune summary written", extra={"path": str(path)})
    return path
