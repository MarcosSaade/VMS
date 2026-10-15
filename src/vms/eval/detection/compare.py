"""Paired comparison of two detection eval runs (`python -m vms.eval.detection compare`).

Two runs on the frozen subset are compared from their saved raw detections, without running
any model again: both are matched against the same ground truth and bootstrapped with the same
resamples (`metrics.paired_difference`), so the interval is for the difference itself. This is
how a backend change (ONNX, INT8; task 5.2) is judged against D32, where the unpaired intervals
(about 5.5 points wide on COCO) would hide any drop that matters.

The saved detections are rounded (boxes to 0.01 px, scores to 1e-5), so the re-scored point
estimates can differ from the runs' own in the last digits; both are recorded.
"""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from vms.bench.results import RunResult, git_info, hardware, package_versions, write_result
from vms.common.ids import new_ulid
from vms.eval.detection.metrics import ImageGt, ImagePred, match_dataset, paired_difference
from vms.eval.detection.run import ground_truth
from vms.eval.detection.settings import DetectionEvalSettings
from vms.eval.detection.subset import read_subset

# D32: INT8 is kept when, on mAP@50-95, the point drop is at most 1.0 point and the upper end
# of the paired 95% interval of the drop at most 2.0 points, on every source.
D32_MAX_DROP = 0.010
D32_MAX_DROP_UPPER = 0.020


class IncomparableRunsError(ValueError):
    pass


def load_run(path: Path) -> RunResult:
    return RunResult.model_validate_json(path.read_text(encoding="utf-8"))


def load_predictions(path: Path) -> dict[str, ImagePred]:
    out = {}
    for line in gzip.decompress(path.read_bytes()).decode().splitlines():
        if not line:
            continue
        row = json.loads(line)
        out[row["id"]] = ImagePred(
            boxes=np.asarray(row["boxes"], dtype=np.float64).reshape(-1, 4),
            scores=np.asarray(row["scores"], dtype=np.float64),
            classes=row["classes"],
        )
    return out


# Settings that must agree for two runs to be scored on the same terms.
_SAME = ("sources", "max_dets_per_class", "mot_frame", "bootstrap", "paths")


def check_comparable(base: RunResult, other: RunResult) -> None:
    for r in (base, other):
        if r.kind != "eval.detection":
            raise IncomparableRunsError(f"{r.run_id}: not a detection eval run ({r.kind})")
        if r.metrics["subset"]["partial"]:
            raise IncomparableRunsError(f"{r.run_id}: partial run (max_images); not comparable")
    if base.metrics["subset"]["subset_sha256"] != other.metrics["subset"]["subset_sha256"]:
        raise IncomparableRunsError("the runs used different subsets")
    for key in _SAME:
        if base.config.get(key) != other.config.get(key):
            raise IncomparableRunsError(f"the runs differ in {key}")


def _interval(i: Any) -> dict[str, float]:
    return dict(i.as_dict())


def compare(base: RunResult, other: RunResult, root: Path, name: str | None = None) -> RunResult:
    check_comparable(base, other)
    settings = DetectionEvalSettings.model_validate(
        {k: base.config[k] for k in (*_SAME, "results_dir", "predictions_dir")}
    )
    preds_base = load_predictions(root / base.metrics["predictions_file"])
    preds_other = load_predictions(root / other.metrics["predictions_file"])
    subset = [i for i in read_subset(root / settings.paths.subset) if i.source in settings.sources]
    b = settings.bootstrap
    sources: dict[str, Any] = {}
    for source in settings.sources:
        images = [i for i in subset if i.source == source]
        gts: list[ImageGt] = [ground_truth(i, settings) for i in images]
        classes = sorted({c for gt in gts for c in gt.classes})
        mb = match_dataset(
            gts, [preds_base[i.id] for i in images], classes, settings.max_dets_per_class
        )
        mo = match_dataset(
            gts, [preds_other[i.id] for i in images], classes, settings.max_dets_per_class
        )
        res = paired_difference(mb, mo, len(images), b.resamples, b.seed, b.confidence)
        drop = -res.delta50_95.value
        drop_upper = -res.delta50_95.lo  # upper end of the drop's interval
        sources[source] = {
            "images": len(images),
            "base": {"map50": _interval(res.base.map50), "map50_95": _interval(res.base.map50_95)},
            "other": {
                "map50": _interval(res.other.map50),
                "map50_95": _interval(res.other.map50_95),
            },
            "runs_reported_map50_95": {
                "base": base.metrics["sources"][source]["map50_95"]["value"],
                "other": other.metrics["sources"][source]["map50_95"]["value"],
            },
            "delta_map50": _interval(res.delta50),
            "delta_map50_95": _interval(res.delta50_95),
            "d32": {
                "drop_map50_95": drop,
                "drop_upper": drop_upper,
                "pass": bool(drop <= D32_MAX_DROP and drop_upper <= D32_MAX_DROP_UPPER),
            },
        }
    now = datetime.now(UTC)
    return RunResult(
        run_id=new_ulid(),
        name=name or "detection-compare",
        kind="eval.detection.compare",
        started_at=now,
        ended_at=datetime.now(UTC),
        seed=b.seed,
        git=git_info(root),
        hardware=hardware(),
        packages=package_versions(("numpy",)),
        config=settings.model_dump(mode="json"),
        metrics={
            "base": {"run_id": base.run_id, "model_version": base.metrics["model_version"]},
            "other": {"run_id": other.run_id, "model_version": other.metrics["model_version"]},
            "sources": sources,
            "d32_pass": all(s["d32"]["pass"] for s in sources.values()),
        },
        notes="delta = other - base, paired percentile bootstrap over images (same resamples "
        "for both). D32 drop = base - other on mAP@50-95; pass needs drop <= 1.0 point and the "
        "upper end of its 95% interval <= 2.0 points on every source.",
    )


def run_compare(base_path: Path, other_path: Path, root: Path, name: str | None) -> Path:
    result = compare(load_run(base_path), load_run(other_path), root, name)
    settings = DetectionEvalSettings.model_validate({"results_dir": result.config["results_dir"]})
    return write_result(root / settings.results_dir, result)
