"""Detection quality under drift (task 9.2): the frozen subset, clean and under each
photometric transform of a scenario, with paired intervals of the change.

`python -m vms.eval.detection drift --scenario bench/scenarios/drift-map-cpu.yaml` builds
the detector once, scores the clean subset, then the subset under each condition (one
`eval.detection` results JSON each, with its raw detections), and compares every condition
with the clean run by the paired bootstrap of `compare` (same images, same resamples). One
summary JSON (`kind: eval.detection.drift`) holds the table: mAP@50 and mAP@50-95 per source
and condition with 95% intervals, and the paired change against clean.

Photometric transforms keep every box valid, so the ground truth is the clean one; geometric
drift (camera shift) and occlusion change what is visible and are not scored here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from vms.bench.results import RunResult, git_info, hardware, package_versions, write_result
from vms.common.ids import new_ulid
from vms.common.logs import get_logger
from vms.common.settings import deep_merge, load_settings
from vms.eval.detection.compare import compare, load_run
from vms.eval.detection.run import PACKAGES, run
from vms.eval.detection.settings import DetectionEvalSettings, EvalDrift
from vms.worker.detector import Detector, build_detector

log = get_logger(__name__)


class Condition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    drift: EvalDrift


class DriftEvalScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    kind: str = "eval-drift"
    config: dict[str, Any] = Field(default_factory=dict)  # overrides eval-detection.yaml
    conditions: list[Condition] = Field(min_length=1)


def load_scenario(path: Path) -> DriftEvalScenario:
    return DriftEvalScenario.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def resolve(
    scenario: DriftEvalScenario, config_dir: Path = Path("configs")
) -> DetectionEvalSettings:
    base = load_settings(DetectionEvalSettings, "eval-detection", config_dir=config_dir)
    merged = deep_merge(base.model_dump(), scenario.config)
    return DetectionEvalSettings.model_validate(merged)


def _scores(run_result: RunResult) -> dict[str, Any]:
    return {
        source: {"map50": s["map50"], "map50_95": s["map50_95"]}
        for source, s in run_result.metrics["sources"].items()
    }


def run_drift(
    scenario: DriftEvalScenario,
    root: Path,
    settings: DetectionEvalSettings,
    detector: Detector | None = None,
) -> Path:
    if settings.max_images is not None:
        raise ValueError("a drift eval scores the whole subset (max_images must be null)")
    started = datetime.now(UTC)
    detector = detector or build_detector(settings.detector, root)
    clean_path = run(settings, root, name=f"{scenario.name}-clean", detector=detector)
    clean = load_run(clean_path)
    rows: list[dict[str, Any]] = []
    for cond in scenario.conditions:
        path = run(
            settings.model_copy(update={"drift": cond.drift}),
            root,
            name=f"{scenario.name}-{cond.name}",
            detector=detector,
        )
        other = load_run(path)
        paired = compare(clean, other, root, name=f"{scenario.name}-{cond.name}-vs-clean")
        sources = {}
        for source, s in paired.metrics["sources"].items():
            base_value = s["base"]["map50_95"]["value"]
            sources[source] = {
                "map50": other.metrics["sources"][source]["map50"],
                "map50_95": other.metrics["sources"][source]["map50_95"],
                "delta_map50": s["delta_map50"],
                "delta_map50_95": s["delta_map50_95"],
                "relative_change_map50_95": (
                    s["delta_map50_95"]["value"] / base_value if base_value else None
                ),
            }
        rows.append(
            {
                "name": cond.name,
                "drift": cond.drift.model_dump(),
                "run_id": other.run_id,
                "results_file": str(path),
                "sources": sources,
            }
        )
        log.info(
            "drift condition scored",
            extra={
                "condition": cond.name,
                **{
                    f"{src}_delta_map50_95": round(v["delta_map50_95"]["value"], 4)
                    for src, v in sources.items()
                },
            },
        )
    result = RunResult(
        run_id=new_ulid(),
        name=scenario.name,
        kind="eval.detection.drift",
        started_at=started,
        ended_at=datetime.now(UTC),
        seed=settings.bootstrap.seed,
        git=git_info(root),
        hardware=hardware(),
        packages=package_versions(PACKAGES),
        config=settings.model_dump(mode="json"),
        scenario=scenario.model_dump(),
        metrics={
            "model_version": clean.metrics["model_version"],
            "subset": clean.metrics["subset"],
            "clean": {
                "run_id": clean.run_id,
                "results_file": str(clean_path),
                "sources": _scores(clean),
            },
            "conditions": rows,
        },
        notes="Each condition applies one photometric transform (vms.drift.transforms) at full "
        "strength to every image; delta = drifted - clean, paired percentile bootstrap over "
        "images with the same resamples. Augmentation is not real domain shift: real night, "
        "rain or defocus also change colour, motion blur and compression in ways these "
        "transforms do not.",
    )
    path = write_result(root / settings.results_dir, result)
    log.info("drift eval written", extra={"path": str(path)})
    return path
