"""ONNX Runtime vs PyTorch parity for the exported detector (task 5.1).

Three comparisons per image, each against Ultralytics as the reference:

1. pre-processing: our letterboxed input vs Ultralytics' LetterBox, byte for byte;
2. raw head output on the same input tensor: max |difference| of box coordinates (input
   pixels) and of class scores;
3. detections after NMS (`OnnxDetector.detect` vs `YoloDetector.detect`): one-to-one matches
   of the same class within the tolerances, plus any detection without a partner. An unmatched
   detection is explained only when its score is within `score_atol` of the confidence
   threshold (a threshold flip) or, through IoU within `iou_atol` of the NMS threshold, it
   was suppressed on one side only (an NMS flip); anything else fails parity.

Also checks that a batch through `detect_batch` gives the same detections as one image at a
time. `python -m vms.worker onnx-parity` runs it over the detection eval subset and the
data-mini frames and writes a results JSON; the tests call the same functions.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from vms.worker.detector import Detection, DetectorSettings, YoloDetector
from vms.worker.onnx_detector import OnnxDetector, letterbox, preprocess


class Tolerances(BaseModel):
    """Why these values: each sits about 10x or more above the largest difference observed
    between the two float32 runtimes (E-037) and far below anything that changes a decision.
    A box edge 0.1 px off is a tenth of a pixel: it moves no pixel of a crop and the IoU of a
    20 px box by about 1%. Scores: 1e-3 on raw anchors (most of which never pass NMS) and 1e-4
    on detections, 1/1500 of the gap between the worker's 0.1 and 0.25 thresholds. Detections
    whose score sits within the tolerance of a threshold, or whose IoU sits within `iou_atol`
    of the NMS threshold, may legitimately differ and are counted as flips."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    raw_box_atol: float = Field(default=0.1, gt=0)  # network-input pixels
    raw_score_atol: float = Field(default=1e-3, gt=0)
    box_atol: float = Field(default=0.1, gt=0)  # image pixels, after undoing the letterbox
    score_atol: float = Field(default=1e-4, gt=0)
    iou_atol: float = Field(default=1e-3, gt=0)  # NMS flips: IoU this close to the threshold


@dataclass
class DetectionDiff:
    matched: int = 0
    max_box_diff: float = 0.0
    max_score_diff: float = 0.0
    threshold_flips: int = 0  # unmatched, score within score_atol of conf
    nms_flips: int = 0  # unmatched, IoU with a kept box within iou_atol of the NMS threshold
    unexplained: list[dict[str, Any]] = field(default_factory=list)

    def add(self, other: DetectionDiff) -> None:
        self.matched += other.matched
        self.max_box_diff = max(self.max_box_diff, other.max_box_diff)
        self.max_score_diff = max(self.max_score_diff, other.max_score_diff)
        self.threshold_flips += other.threshold_flips
        self.nms_flips += other.nms_flips
        self.unexplained.extend(other.unexplained)

    def as_dict(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "max_box_diff_px": self.max_box_diff,
            "max_score_diff": self.max_score_diff,
            "threshold_flips": self.threshold_flips,
            "nms_flips": self.nms_flips,
            "unexplained": len(self.unexplained),
            "unexplained_examples": self.unexplained[:10],
        }


def _box(d: Detection) -> NDArray[np.float64]:
    return np.array([d.x1, d.y1, d.x2, d.y2], dtype=np.float64)


def _iou(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def compare_detections(
    ref: Sequence[Detection],
    got: Sequence[Detection],
    conf: float,
    iou: float,
    tol: Tolerances,
) -> DetectionDiff:
    """Match `got` to `ref` one to one (same class, within tolerances, closest first) and
    classify whatever is left over (see the module docstring)."""
    diff = DetectionDiff()
    a_box = np.array([_box(d) for d in ref]).reshape(-1, 4)
    b_box = np.array([_box(d) for d in got]).reshape(-1, 4)
    a_score = np.array([d.confidence for d in ref])
    b_score = np.array([d.confidence for d in got])
    # dtype explicit: an image without detections must still give a boolean (n, m) matrix.
    same_class = np.array(
        [[a.class_name == b.class_name for b in got] for a in ref], dtype=bool
    ).reshape(len(ref), len(got))
    box_diff = np.abs(a_box[:, None, :] - b_box[None, :, :]).max(axis=2, initial=0.0)
    score_diff = np.abs(a_score[:, None] - b_score[None, :])
    ok = same_class & (box_diff <= tol.box_atol) & (score_diff <= tol.score_atol)
    ii, jj = np.nonzero(ok)
    order = np.argsort(box_diff[ii, jj], kind="stable")
    used_ref: set[int] = set()
    used_got: set[int] = set()
    for k in order:
        i, j = int(ii[k]), int(jj[k])
        if i in used_ref or j in used_got:
            continue
        used_ref.add(i)
        used_got.add(j)
        diff.matched += 1
        diff.max_box_diff = max(diff.max_box_diff, float(box_diff[i, j]))
        diff.max_score_diff = max(diff.max_score_diff, float(score_diff[i, j]))

    for side, dets, used, other in (("ref", ref, used_ref, got), ("got", got, used_got, ref)):
        for k, d in enumerate(dets):
            if k in used:
                continue
            if abs(d.confidence - conf) <= tol.score_atol:
                diff.threshold_flips += 1
                continue
            # Suppressed on the other side by a box of the same class whose IoU with it is
            # within iou_atol of the NMS threshold.
            ious = [
                _iou(_box(d), _box(o))
                for o in (*dets, *other)
                if o is not d and o.class_name == d.class_name and o.confidence >= d.confidence
            ]
            if any(abs(v - iou) <= tol.iou_atol for v in ious):
                diff.nms_flips += 1
                continue
            diff.unexplained.append(
                {
                    "side": side,
                    "class": d.class_name,
                    "confidence": round(d.confidence, 6),
                    "box": [round(v, 2) for v in (d.x1, d.y1, d.x2, d.y2)],
                }
            )
    return diff


@dataclass
class ParityStats:
    images: int = 0
    letterbox_mismatches: int = 0
    raw_max_box_diff: float = 0.0
    raw_max_score_diff: float = 0.0
    detections: dict[str, DetectionDiff] = field(default_factory=dict)  # by conf threshold
    seconds: float = 0.0

    def ok(self, tol: Tolerances) -> bool:
        return (
            self.letterbox_mismatches == 0
            and self.raw_max_box_diff <= tol.raw_box_atol
            and self.raw_max_score_diff <= tol.raw_score_atol
            and all(not d.unexplained for d in self.detections.values())
        )

    def as_dict(self, tol: Tolerances) -> dict[str, Any]:
        return {
            "images": self.images,
            "letterbox_mismatches": self.letterbox_mismatches,
            "raw_max_box_diff_px": self.raw_max_box_diff,
            "raw_max_score_diff": self.raw_max_score_diff,
            "detections": {k: v.as_dict() for k, v in self.detections.items()},
            "ok": self.ok(tol),
            "seconds": self.seconds,
        }


def ultralytics_letterbox(image: NDArray[np.uint8], imgsz: int, stride: int) -> NDArray[np.uint8]:
    from ultralytics.data.augment import LetterBox

    out: Any = LetterBox((imgsz, imgsz), auto=True, stride=stride)(image=image)
    return np.asarray(out, dtype=np.uint8)


def torch_forward(reference: YoloDetector, x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Raw head output of the reference's (fused, eval-mode) PyTorch module on `x`."""
    import torch

    module = reference.torch_module()
    with torch.inference_mode():
        out = module(torch.from_numpy(x))
    out = out[0] if isinstance(out, list | tuple) else out
    return np.asarray(out.numpy(), dtype=np.float32)


class Comparator:
    """Runs the comparisons for one weights file at several confidence thresholds."""

    def __init__(
        self,
        weights_path: Path,
        onnx_path: Path,
        weights_sha256: str,
        settings: DetectorSettings,
        confs: Sequence[float],
        tol: Tolerances,
    ) -> None:
        self.settings = settings
        self.tol = tol
        self.confs = list(confs)
        self.reference = {
            c: YoloDetector(weights_path, weights_sha256, settings.model_copy(update={"conf": c}))
            for c in self.confs
        }
        self.onnx = {
            c: OnnxDetector(onnx_path, weights_sha256, settings.model_copy(update={"conf": c}))
            for c in self.confs
        }
        for det in self.reference.values():
            det.warmup()  # Ultralytics builds and fuses its module on the first predict

    def image(self, image: NDArray[np.uint8], stats: ParityStats) -> None:
        s = self.settings
        stride = 32
        x, _ = preprocess([image], s.imgsz, stride)
        ours = letterbox(image, s.imgsz, stride, auto=True)
        if not np.array_equal(ours, ultralytics_letterbox(image, s.imgsz, stride)):
            stats.letterbox_mismatches += 1
        first = self.confs[0]
        ref_raw = torch_forward(self.reference[first], x)
        got_raw = self.onnx[first].forward(x)
        d = np.abs(ref_raw - got_raw)
        stats.raw_max_box_diff = max(stats.raw_max_box_diff, float(d[:, :4].max()))
        stats.raw_max_score_diff = max(stats.raw_max_score_diff, float(d[:, 4:].max()))
        for c in self.confs:
            diff = compare_detections(
                self.reference[c].detect(image), self.onnx[c].detect(image), c, s.iou, self.tol
            )
            stats.detections.setdefault(f"conf={c}", DetectionDiff()).add(diff)
        stats.images += 1

    def run(self, images: Iterable[NDArray[np.uint8]]) -> ParityStats:
        stats = ParityStats()
        t0 = time.perf_counter()
        for image in images:
            self.image(image, stats)
        stats.seconds = time.perf_counter() - t0
        return stats

    def batch(self, images: Sequence[NDArray[np.uint8]], batch_size: int) -> DetectionDiff:
        """`detect_batch` in chunks of `batch_size` vs `detect` one image at a time."""
        c = self.confs[0]
        onnx = self.onnx[c]
        out = DetectionDiff()
        for k in range(0, len(images), batch_size):
            chunk = list(images[k : k + batch_size])
            for image, batched in zip(chunk, onnx.detect_batch(chunk), strict=True):
                out.add(
                    compare_detections(onnx.detect(image), batched, c, self.settings.iou, self.tol)
                )
        return out


class ParityConfig(BaseModel):
    """`python -m vms.worker onnx-parity` (make onnx-parity)."""

    model_config = ConfigDict(extra="forbid")

    weights: str = "yolo11n"
    confs: list[float] = Field(default=[0.1, 0.001], min_length=1)  # worker's, then eval's
    eval_images: int | None = Field(default=None, ge=1)  # None: the whole detection subset
    clips: list[str] = Field(default=["mot17-09-mini", "mot17-04-mini", "commons-roncesvalles"])
    frames_per_clip: int = Field(default=50, ge=1)
    batch_size: int = Field(default=8, ge=2)
    data_dir: Path = Path("data")
    results_dir: Path = Path("bench/results")
    tolerances: Tolerances = Field(default_factory=Tolerances)


def run_parity(config: ParityConfig, root: Path = Path()) -> tuple[Path, bool]:
    """Compare on the detection eval subset (COCO shapes vary; MOT17 as the worker sees it) and
    on data-mini frames (the 1280x720 workload, also batched). Returns the results path and
    whether parity holds."""
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
    from vms.common.settings import load_settings
    from vms.eval.detection.run import load_image
    from vms.eval.detection.settings import DetectionEvalSettings
    from vms.eval.detection.subset import read_subset
    from vms.loadgen.frames import load_clip
    from vms.loadgen.settings import FrameSettings
    from vms.worker.onnx_detector import verified_export
    from vms.worker.service import decode_jpeg
    from vms.worker.weights import ensure_weights, load_models_config

    log = get_logger(__name__)
    models = load_models_config(root / "configs/models.yaml")
    sha = models.weights[config.weights].sha256
    weights_path = ensure_weights(models, config.weights, root=root)
    onnx_path, export = verified_export(root / models.onnx_dir, config.weights, sha)
    settings = DetectorSettings(weights=config.weights)  # imgsz, iou, max_det: defaults
    comparator = Comparator(weights_path, onnx_path, sha, settings, config.confs, config.tolerances)

    started = datetime.now(UTC)
    eval_settings = load_settings(DetectionEvalSettings, "eval-detection")
    images_dir = root / eval_settings.paths.images_dir
    subset = read_subset(root / eval_settings.paths.subset)[: config.eval_images]
    by_source: dict[str, list[NDArray[np.uint8]]] = {}
    for item in subset:
        pixels, _ = load_image(item, images_dir, eval_settings)
        by_source.setdefault(item.source, []).append(pixels)
    frames = [
        decode_jpeg(f)
        for clip_id in config.clips
        for f in load_clip(
            root / config.data_dir, clip_id, FrameSettings(), max_frames=config.frames_per_clip
        ).frames
    ]
    sets = {**{f"eval-{k}": v for k, v in by_source.items()}, "data-mini": frames}
    results: dict[str, Any] = {}
    ok = True
    for name, images in sets.items():
        stats = comparator.run(images)
        results[name] = stats.as_dict(config.tolerances)
        ok &= stats.ok(config.tolerances)
        log.info(
            "parity set done",
            extra={"set": name} | {k: v for k, v in results[name].items() if k != "detections"},
        )
    batch = comparator.batch(frames, config.batch_size)
    results["data-mini-batched"] = {"batch_size": config.batch_size, **batch.as_dict()}
    ok &= not batch.unexplained

    result = RunResult(
        run_id=new_ulid(),
        name=f"onnx-parity-{config.weights}",
        kind="worker.onnx_parity",
        started_at=started,
        ended_at=datetime.now(UTC),
        seed=0,  # nothing random
        git=git_info(root),
        hardware=hardware(),
        packages=package_versions(
            (*DEFAULT_PACKAGES, "torch", "ultralytics", "onnx", "onnxruntime")
        ),
        config=config.model_dump(mode="json") | {"detector": settings.model_dump(mode="json")},
        metrics={"ok": ok, "export": export.model_dump(mode="json"), "sets": results},
        notes="ONNX Runtime CPU vs Ultralytics PyTorch CPU, float32. Raw diffs are on the same "
        "input tensor; detection diffs are over one-to-one matches (see vms.worker.parity).",
    )
    path = write_result(root / config.results_dir, result)
    log.info("parity results written", extra={"path": str(path), "ok": ok})
    return path, ok
