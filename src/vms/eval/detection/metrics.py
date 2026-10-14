"""COCO-style mAP with bootstrap confidence intervals over images.

The point estimate follows pycocotools' bbox evaluation (`COCOeval` with area range "all")
exactly, and a unit test checks that against pycocotools on random data:

- per image and class, detections are sorted by score and cut to `max_dets`; each is greedily
  matched, at each IoU threshold, to the unmatched ground-truth box it overlaps most (at least
  the threshold), preferring real boxes over ignore regions;
- a detection matched to an ignore region counts as neither true nor false positive; crowd
  regions use intersection over the detection's area and may absorb any number of detections;
- per class, detections of all images are ranked by score; precision is made monotone and read
  at 101 recall points (0, 0.01, ..., 1); AP is their mean; classes without real ground-truth
  boxes are left out; mAP@50-95 averages AP over classes and IoU 0.50:0.05:0.95.

Bootstrap: matching is per image, so it does not change when images are resampled. A resample
is a vector of integer image counts (multinomial), and its AP comes from weighted cumulative
sums over the same ranked detections. This is exact, not an approximation: repeated copies of a
true positive only add points of lower precision before the last copy, which the monotone
envelope removes, and copies of a false positive add points at unchanged recall. A unit test
checks it against scoring an explicitly duplicated dataset.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
RECALL_POINTS = np.linspace(0.0, 1.0, 101)
_EPS = np.spacing(1)

Boxes = NDArray[np.float64]  # (N, 4) x1, y1, x2, y2


@dataclass(frozen=True)
class ImageGt:
    boxes: Boxes
    classes: Sequence[str]
    ignore: NDArray[np.bool_]
    crowd: NDArray[np.bool_]


@dataclass(frozen=True)
class ImagePred:
    boxes: Boxes
    scores: NDArray[np.float64]
    classes: Sequence[str]


def box_iou(dt: Boxes, gt: Boxes, crowd: NDArray[np.bool_]) -> NDArray[np.float64]:
    """IoU matrix (len(dt), len(gt)); for crowd ground truth, intersection over the detection's
    area (pycocotools' convention)."""
    if len(dt) == 0 or len(gt) == 0:
        return np.zeros((len(dt), len(gt)))
    ix1 = np.maximum(dt[:, None, 0], gt[None, :, 0])
    iy1 = np.maximum(dt[:, None, 1], gt[None, :, 1])
    ix2 = np.minimum(dt[:, None, 2], gt[None, :, 2])
    iy2 = np.minimum(dt[:, None, 3], gt[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_dt = (dt[:, 2] - dt[:, 0]) * (dt[:, 3] - dt[:, 1])
    area_gt = (gt[:, 2] - gt[:, 0]) * (gt[:, 3] - gt[:, 1])
    union = np.where(crowd[None, :], area_dt[:, None], area_dt[:, None] + area_gt[None, :] - inter)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


@dataclass(frozen=True)
class ImageMatch:
    """Matching of one image and class: detections in score order, cut to max_dets."""

    scores: NDArray[np.float64]  # (D,)
    matched: NDArray[np.bool_]  # (T, D) matched to some ground truth (real or ignore)
    ignored: NDArray[np.bool_]  # (T, D) matched to an ignore region
    n_pos: int  # real (non-ignore) ground-truth boxes


def match_image(
    gt_boxes: Boxes,
    gt_ignore: NDArray[np.bool_],
    gt_crowd: NDArray[np.bool_],
    dt_boxes: Boxes,
    dt_scores: NDArray[np.float64],
    max_dets: int,
    iou_thresholds: NDArray[np.float64] = IOU_THRESHOLDS,
) -> ImageMatch:
    """pycocotools' `evaluateImg` for one image and class."""
    g_order = np.argsort(gt_ignore, kind="mergesort")  # real boxes first
    gt_boxes, gt_ignore, gt_crowd = gt_boxes[g_order], gt_ignore[g_order], gt_crowd[g_order]
    d_order = np.argsort(-dt_scores, kind="mergesort")[:max_dets]
    dt_boxes, dt_scores = dt_boxes[d_order], dt_scores[d_order]
    ious = box_iou(dt_boxes, gt_boxes, gt_crowd)
    n_t, n_d, n_g = len(iou_thresholds), len(dt_boxes), len(gt_boxes)
    matched = np.zeros((n_t, n_d), dtype=bool)
    ignored = np.zeros((n_t, n_d), dtype=bool)
    for t, thr in enumerate(iou_thresholds):
        gt_taken = np.zeros(n_g, dtype=bool)
        for d in range(n_d):
            best_iou = min(float(thr), 1 - 1e-10)
            m = -1
            for g in range(n_g):
                if gt_taken[g] and not gt_crowd[g]:
                    continue
                # Already matched a real box and only ignore regions are left: stop.
                if m > -1 and not gt_ignore[m] and gt_ignore[g]:
                    break
                if ious[d, g] < best_iou:
                    continue
                best_iou, m = ious[d, g], g
            if m == -1:
                continue
            matched[t, d] = True
            ignored[t, d] = bool(gt_ignore[m])
            gt_taken[m] = True
    return ImageMatch(dt_scores, matched, ignored, int((~gt_ignore).sum()))


@dataclass
class ClassMatches:
    """All images' matches for one class, detections ranked by score."""

    image: NDArray[np.int64]  # (D,) image index of each ranked detection
    tp: NDArray[np.bool_]  # (T, D)
    fp: NDArray[np.bool_]  # (T, D)
    n_pos: NDArray[np.int64]  # (n_images,) real ground-truth boxes per image


def match_dataset(
    gts: Sequence[ImageGt],
    preds: Sequence[ImagePred],
    classes: Sequence[str],
    max_dets: int,
    iou_thresholds: NDArray[np.float64] = IOU_THRESHOLDS,
) -> dict[str, ClassMatches]:
    """Match every image and class once; the result is all the bootstrap needs."""
    if len(gts) != len(preds):
        raise ValueError("one prediction set per ground-truth image")
    out = {}
    for cls in classes:
        scores, images, matched, ignored = [], [], [], []
        n_pos = np.zeros(len(gts), dtype=np.int64)
        for i, (gt, pr) in enumerate(zip(gts, preds, strict=True)):
            g = np.asarray([c == cls for c in gt.classes], dtype=bool)
            d = np.asarray([c == cls for c in pr.classes], dtype=bool)
            if not g.any() and not d.any():
                continue
            m = match_image(
                gt.boxes[g].reshape(-1, 4),
                gt.ignore[g],
                gt.crowd[g],
                pr.boxes[d].reshape(-1, 4),
                pr.scores[d],
                max_dets,
                iou_thresholds,
            )
            n_pos[i] = m.n_pos
            scores.append(m.scores)
            images.append(np.full(len(m.scores), i, dtype=np.int64))
            matched.append(m.matched)
            ignored.append(m.ignored)
        n_t = len(iou_thresholds)
        if scores:
            all_scores = np.concatenate(scores)
            order = np.argsort(-all_scores, kind="mergesort")
            mt = np.concatenate(matched, axis=1)[:, order]
            ig = np.concatenate(ignored, axis=1)[:, order]
            image = np.concatenate(images)[order]
        else:
            mt = ig = np.zeros((n_t, 0), dtype=bool)
            image = np.zeros(0, dtype=np.int64)
        out[cls] = ClassMatches(image=image, tp=mt & ~ig, fp=~mt & ~ig, n_pos=n_pos)
    return out


def ap_rows(
    tp_cum: NDArray[np.float64], fp_cum: NDArray[np.float64], n_pos: NDArray[np.float64]
) -> NDArray[np.float64]:
    """AP per row from cumulative (weighted) true and false positives over ranked detections.
    Rows with no positives are NaN (left out of the mean, like pycocotools' -1)."""
    rows, n = tp_cum.shape
    ap = np.full(rows, np.nan)
    if n == 0:
        ap[n_pos > 0] = 0.0
        return ap
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = tp_cum / n_pos[:, None]
    precision = tp_cum / (tp_cum + fp_cum + _EPS)
    envelope = np.maximum.accumulate(precision[:, ::-1], axis=1)[:, ::-1]
    for r in np.flatnonzero(n_pos > 0):
        idx = np.searchsorted(recall[r], RECALL_POINTS, side="left")
        q = np.where(idx < n, envelope[r, np.minimum(idx, n - 1)], 0.0)
        ap[r] = q.mean()
    return ap


def class_ap(m: ClassMatches, weights: NDArray[np.float64]) -> NDArray[np.float64]:
    """AP (rows of `weights`, one per resample) x (IoU thresholds) for one class.

    `weights` is (R, n_images): how many times each image is in each resample."""
    w_det = weights[:, m.image]  # (R, D)
    n_pos = weights @ m.n_pos.astype(np.float64)  # (R,)
    out = np.empty((weights.shape[0], m.tp.shape[0]))
    for t in range(m.tp.shape[0]):
        tp_cum = np.cumsum(w_det * m.tp[t], axis=1)
        fp_cum = np.cumsum(w_det * m.fp[t], axis=1)
        out[:, t] = ap_rows(tp_cum, fp_cum, n_pos)
    return out


@dataclass(frozen=True)
class Interval:
    value: float
    lo: float
    hi: float
    std: float
    mean: float  # of the bootstrap distribution; its distance from `value` is the bias

    def as_dict(self) -> dict[str, float]:
        return {
            "value": self.value,
            "ci_lo": self.lo,
            "ci_hi": self.hi,
            "boot_std": self.std,
            "boot_mean": self.mean,
        }


@dataclass
class MapResult:
    map50: Interval
    map50_95: Interval
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)  # ap50, ap50_95, n_pos
    resamples: int = 0
    confidence: float = 0.95


def _nanmean_rows(values: NDArray[np.float64]) -> NDArray[np.float64]:
    """Mean over classes per resample, ignoring classes absent from that resample."""
    counts = np.sum(~np.isnan(values), axis=1)
    sums = np.nansum(values, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)


def _interval(point: float, samples: NDArray[np.float64], confidence: float) -> Interval:
    samples = samples[~np.isnan(samples)]
    if len(samples) == 0:
        return Interval(point, float("nan"), float("nan"), float("nan"), float("nan"))
    alpha = (1 - confidence) / 2
    lo, hi = np.quantile(samples, [alpha, 1 - alpha])
    std = float(samples.std(ddof=1)) if len(samples) > 1 else 0.0
    return Interval(point, float(lo), float(hi), std, float(samples.mean()))


def evaluate(
    matches: dict[str, ClassMatches],
    n_images: int,
    resamples: int,
    seed: int,
    confidence: float = 0.95,
    chunk: int = 250,
) -> MapResult:
    """Point mAP (all images once) and percentile bootstrap intervals over images."""
    classes = list(matches)
    ones = np.ones((1, n_images))
    point = (
        np.stack([class_ap(matches[c], ones)[0] for c in classes])
        if classes
        else np.empty((0, len(IOU_THRESHOLDS)))
    )  # (C, T)
    per_class = {
        c: {
            "ap50": float(point[k, 0]),
            "ap50_95": float(np.mean(point[k])),
            "n_pos": int(matches[c].n_pos.sum()),
        }
        for k, c in enumerate(classes)
        if not np.isnan(point[k, 0])
    }
    valid = ~np.isnan(point[:, 0])
    map50 = float(point[valid, 0].mean()) if valid.any() else float("nan")
    map50_95 = float(point[valid].mean()) if valid.any() else float("nan")

    rng = np.random.default_rng(seed)
    if n_images == 0:
        resamples = 0
    boot50 = np.empty(resamples)
    boot5095 = np.empty(resamples)
    for start in range(0, resamples, chunk):
        size = min(chunk, resamples - start)
        weights = rng.multinomial(n_images, np.full(n_images, 1 / n_images), size=size).astype(
            np.float64
        )
        aps = np.stack([class_ap(matches[c], weights) for c in classes], axis=1)  # (R, C, T)
        boot50[start : start + size] = _nanmean_rows(aps[:, :, 0])
        boot5095[start : start + size] = _nanmean_rows(aps.mean(axis=2))
    return MapResult(
        map50=_interval(map50, boot50, confidence),
        map50_95=_interval(map50_95, boot5095, confidence),
        per_class=per_class,
        resamples=resamples,
        confidence=confidence,
    )
