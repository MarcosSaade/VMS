"""Score a detector on the frozen subset: mAP@50 and mAP@50-95 with bootstrap intervals.

The detector is anything behind the worker's `Detector` interface, built from settings by
`vms.worker.detector.build_detector` (`detector.backend`: Ultralytics PyTorch or ONNX
Runtime). COCO images go in as stored; MOT17 frames go through loadgen's resize and re-encode
first, with their boxes scaled to match, so the mot17 numbers describe the frames the worker
actually sees. With `drift` set, every image then goes through that photometric transform
(task 9.2) before detection.
"""

from __future__ import annotations

import gzip
import json
import time
import zlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

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
from vms.drift.transforms import apply as apply_drift
from vms.eval.detection.metrics import ImageGt, ImagePred, evaluate, match_dataset
from vms.eval.detection.settings import DetectionEvalSettings
from vms.eval.detection.subset import GtImage, SubsetManifest, image_ok, read_subset
from vms.loadgen.frames import prepare_frame
from vms.worker.detector import Detector, build_detector
from vms.worker.service import decode_jpeg

log = get_logger(__name__)

PACKAGES = (*DEFAULT_PACKAGES, "torch", "torchvision", "ultralytics", "onnxruntime")


class SubsetNotReadyError(RuntimeError):
    pass


def ground_truth(image: GtImage, settings: DetectionEvalSettings) -> ImageGt:
    """The image's ground truth in the pixels the detector sees (MOT boxes scaled)."""
    boxes = np.asarray(image.boxes, dtype=np.float64).reshape(-1, 4)
    if image.source == "mot17":
        f = settings.mot_frame
        boxes = boxes * np.array([f.width / image.width, f.height / image.height] * 2)
    return ImageGt(
        boxes=boxes,
        classes=image.classes,
        ignore=np.asarray(image.ignore, dtype=bool),
        crowd=np.asarray(image.crowd, dtype=bool),
    )


def load_image(
    image: GtImage, images_dir: Path, settings: DetectionEvalSettings
) -> tuple[np.ndarray, ImageGt]:
    """The image as the detector gets it, and its ground truth in that image's pixels."""
    data = image.path(images_dir).read_bytes()
    if image.source == "mot17":
        f = settings.mot_frame
        data = prepare_frame(data, (f.width, f.height), f.jpeg_quality)
    return decode_jpeg(data), ground_truth(image, settings)


def drifted(pixels: np.ndarray, image_id: str, settings: DetectionEvalSettings) -> np.ndarray:
    d = settings.drift
    if d is None:
        return pixels
    rng = np.random.default_rng([d.seed, zlib.crc32(image_id.encode())])
    return apply_drift(d.kind, pixels, 1.0, d.strength, rng=rng)


def predict(detector: Detector, pixels: np.ndarray) -> ImagePred:
    dets = detector.detect(pixels)
    return ImagePred(
        boxes=np.asarray([[d.x1, d.y1, d.x2, d.y2] for d in dets], dtype=np.float64).reshape(-1, 4),
        scores=np.asarray([d.confidence for d in dets], dtype=np.float64),
        classes=[d.class_name for d in dets],
    )


def prediction_line(image_id: str, pred: ImagePred) -> str:
    return json.dumps(
        {
            "id": image_id,
            "boxes": np.round(pred.boxes, 2).tolist(),
            "scores": np.round(pred.scores, 5).tolist(),
            "classes": list(pred.classes),
        },
        separators=(",", ":"),
    )


def score_source(
    gts: list[ImageGt], preds: list[ImagePred], settings: DetectionEvalSettings
) -> dict[str, Any]:
    classes = sorted({c for gt in gts for c in gt.classes})
    matches = match_dataset(gts, preds, classes, settings.max_dets_per_class)
    result = evaluate(
        matches,
        len(gts),
        settings.bootstrap.resamples,
        settings.bootstrap.seed,
        settings.bootstrap.confidence,
    )
    return {
        "images": len(gts),
        "gt_boxes": int(sum((~g.ignore).sum() for g in gts)),
        "ignore_regions": int(sum(g.ignore.sum() for g in gts)),
        "predictions": int(sum(len(p.scores) for p in preds)),
        "classes_scored": len(result.per_class),
        "map50": result.map50.as_dict(),
        "map50_95": result.map50_95.as_dict(),
        "per_class": result.per_class,
    }


def run(
    settings: DetectionEvalSettings,
    root: Path,
    name: str | None = None,
    detector: Detector | None = None,
) -> Path:
    p = settings.paths
    manifest = SubsetManifest.model_validate_json((root / p.manifest).read_text(encoding="utf-8"))
    images_dir = root / p.images_dir
    subset = [img for img in read_subset(root / p.subset) if img.source in settings.sources]
    if settings.max_images is not None:
        subset = subset[: settings.max_images]
    missing = [img.id for img in subset if not image_ok(images_dir, img)]
    if missing:
        raise SubsetNotReadyError(
            f"{len(missing)} images missing or corrupt under {images_dir} (first: {missing[0]}); "
            "run `make eval-detection-data`"
        )
    detector = detector or build_detector(settings.detector, root)
    run_id = new_ulid()
    started = datetime.now(UTC)

    gts: dict[str, list[ImageGt]] = {s: [] for s in settings.sources}
    preds: dict[str, list[ImagePred]] = {s: [] for s in settings.sources}
    infer_s = 0.0
    dump: list[str] = []
    for k, image in enumerate(subset):
        pixels, gt = load_image(image, images_dir, settings)
        pixels = drifted(pixels, image.id, settings)
        if k == 0:
            detector.detect(pixels)  # warmup, not timed
        t0 = time.perf_counter()
        pred = predict(detector, pixels)
        infer_s += time.perf_counter() - t0
        gts[image.source].append(gt)
        preds[image.source].append(pred)
        dump.append(prediction_line(image.id, pred))
        if (k + 1) % 100 == 0:
            log.info("inference progress", extra={"done": k + 1, "of": len(subset)})

    predictions = root / settings.predictions_dir / f"{run_id}.jsonl.gz"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_bytes(gzip.compress(("\n".join(dump) + "\n").encode(), mtime=0))
    scores = {
        source: score_source(gts[source], preds[source], settings)
        for source in settings.sources
        if gts[source]
    }
    result = RunResult(
        run_id=run_id,
        name=name or f"detection-{settings.detector.weights}",
        kind="eval.detection",
        started_at=started,
        ended_at=datetime.now(UTC),
        seed=settings.bootstrap.seed,
        git=git_info(root),
        hardware=hardware(),
        packages=package_versions(PACKAGES),
        config=settings.model_dump(mode="json"),
        metrics={
            "model_version": detector.model_version,
            "drift": settings.drift.model_dump() if settings.drift else None,
            "subset": {
                "version": manifest.version,
                "subset_sha256": manifest.subset_sha256,
                "spec_sha256": manifest.spec_sha256,
                "partial": settings.max_images is not None,
            },
            # Raw detections per image, for re-scoring without inference (not committed).
            "predictions_file": str(settings.predictions_dir / predictions.name),
            "infer_s_total": infer_s,
            "images_per_s": len(subset) / infer_s if infer_s else None,
            "sources": scores,
        },
        notes="mAP point estimates follow pycocotools (area all); intervals are percentile "
        "bootstrap over images. MOT17 frames are one per second of three sequences and still "
        "correlated within a sequence, so its interval is optimistic.",
    )
    path = write_result(root / settings.results_dir, result)
    log.info(
        "detection eval done",
        extra={
            "path": str(path),
            **{f"{s}_map50_95": round(v["map50_95"]["value"], 4) for s, v in scores.items()},
        },
    )
    return path
