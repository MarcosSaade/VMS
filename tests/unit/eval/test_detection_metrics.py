"""COCO-style mAP: equal to pycocotools on random data, exact weighted bootstrap, edge cases."""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from vms.eval.detection.metrics import (
    IOU_THRESHOLDS,
    ImageGt,
    ImagePred,
    box_iou,
    class_ap,
    evaluate,
    match_dataset,
    match_image,
)

CLASSES = ["cat", "dog", "person"]


def _boxes(rng: np.random.Generator, n: int, size: float = 100.0) -> np.ndarray:
    xy = rng.uniform(0, size, (n, 2))
    wh = rng.uniform(4, 40, (n, 2))
    return np.hstack([xy, xy + wh])


def random_dataset(
    seed: int, n_images: int = 12, crowd_rate: float = 0.1, ignore_rate: float = 0.1
) -> tuple[list[ImageGt], list[ImagePred]]:
    """Ground truth plus noisy detections near it, random false positives, an image with more
    than 100 detections of one class, and images with no ground truth or no detections."""
    rng = np.random.default_rng(seed)
    gts, preds = [], []
    for i in range(n_images):
        n_gt = 0 if i == 0 else int(rng.integers(1, 9))
        gt_boxes = _boxes(rng, n_gt)
        gt_classes = [CLASSES[int(c)] for c in rng.integers(0, len(CLASSES), n_gt)]
        flags = rng.uniform(size=n_gt)
        crowd = flags < crowd_rate
        ignore = crowd | ((flags >= crowd_rate) & (flags < crowd_rate + ignore_rate))
        gts.append(ImageGt(gt_boxes, gt_classes, ignore, crowd))

        hits = [k for k in range(n_gt) if rng.uniform() < 0.8] if i != 1 else []
        jitter = rng.normal(0, 3, (len(hits), 4))
        dt = [gt_boxes[k] + jitter[j] for j, k in enumerate(hits)]
        dt_classes = [gt_classes[k] for k in hits]
        n_fp = 130 if i == 2 else int(rng.integers(0, 6))
        if i == 1:
            n_fp = 0
        fp = _boxes(rng, n_fp)
        dt.extend(fp)
        dt_classes += ["person" if i == 2 else CLASSES[int(c)] for c in rng.integers(0, 3, n_fp)]
        boxes = np.asarray(dt, dtype=np.float64).reshape(-1, 4)
        boxes[:, 2:] = np.maximum(boxes[:, 2:], boxes[:, :2] + 1)
        preds.append(ImagePred(boxes, rng.uniform(0.01, 1.0, len(boxes)), dt_classes))
    return gts, preds


def pycocotools_maps(gts: list[ImageGt], preds: list[ImagePred]) -> tuple[float, float]:
    """mAP@50 and mAP@50-95 from pycocotools (area all, maxDets 100). Non-crowd ignore regions
    are expressed as boxes whose recorded area is outside the 'all' range."""
    cat_ids = {c: k + 1 for k, c in enumerate(CLASSES)}
    images, anns, dets = [], [], []
    for i, (gt, pr) in enumerate(zip(gts, preds, strict=True)):
        images.append({"id": i + 1, "width": 200, "height": 200})
        for b, c, ig, cr in zip(gt.boxes, gt.classes, gt.ignore, gt.crowd, strict=True):
            w, h = b[2] - b[0], b[3] - b[1]
            anns.append(
                {
                    "id": len(anns) + 1,
                    "image_id": i + 1,
                    "category_id": cat_ids[c],
                    "bbox": [b[0], b[1], w, h],
                    "area": 1e11 if ig and not cr else w * h,
                    "iscrowd": int(cr),
                }
            )
        for b, s, c in zip(pr.boxes, pr.scores, pr.classes, strict=True):
            dets.append(
                {
                    "image_id": i + 1,
                    "category_id": cat_ids[c],
                    "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                    "score": float(s),
                }
            )
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = {
            "images": images,
            "annotations": anns,
            "categories": [{"id": v, "name": k} for k, v in cat_ids.items()],
        }
        coco.createIndex()
        result = coco.loadRes(dets)
        ev = COCOeval(coco, result, "bbox")
        ev.evaluate()
        ev.accumulate()
    precision = ev.eval["precision"][:, :, :, 0, 2]  # T, R, K at area all, maxDets 100
    valid = precision[0, 0, :] > -1
    return float(precision[0][:, valid].mean()), float(precision[:, :, valid].mean())


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_point_estimate_equals_pycocotools(seed: int) -> None:
    gts, preds = random_dataset(seed)
    matches = match_dataset(gts, preds, CLASSES, max_dets=100)
    result = evaluate(matches, len(gts), resamples=5, seed=0)
    expected50, expected5095 = pycocotools_maps(gts, preds)
    assert result.map50.value == pytest.approx(expected50, abs=1e-12)
    assert result.map50_95.value == pytest.approx(expected5095, abs=1e-12)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_weighted_ap_equals_scoring_duplicated_images(seed: int) -> None:
    gts, preds = random_dataset(seed, crowd_rate=0.15, ignore_rate=0.15)
    counts = np.random.default_rng(100 + seed).multinomial(
        len(gts), np.full(len(gts), 1 / len(gts))
    )
    assert (counts == 0).any()
    assert (counts > 1).any()
    dup_gts = [g for g, c in zip(gts, counts, strict=True) for _ in range(c)]
    dup_preds = [p for p, c in zip(preds, counts, strict=True) for _ in range(c)]

    weighted = match_dataset(gts, preds, CLASSES, max_dets=100)
    explicit = match_dataset(dup_gts, dup_preds, CLASSES, max_dets=100)
    for cls in CLASSES:
        got = class_ap(weighted[cls], counts[None, :].astype(np.float64))
        want = class_ap(explicit[cls], np.ones((1, len(dup_gts))))
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12, equal_nan=True)


def test_crowd_iou_is_over_detection_area() -> None:
    dt = np.array([[0.0, 0.0, 10.0, 10.0]])
    gt = np.array([[0.0, 0.0, 100.0, 100.0], [0.0, 0.0, 100.0, 100.0]])
    iou = box_iou(dt, gt, np.array([False, True]))
    assert iou[0, 0] == pytest.approx(0.01)
    assert iou[0, 1] == pytest.approx(1.0)


def test_detections_on_ignore_regions_are_neither_hits_nor_misses() -> None:
    gt = np.array([[0.0, 0, 10, 10], [50, 50, 60, 60], [100, 100, 200, 200]])
    ignore = np.array([False, True, True])
    crowd = np.array([False, False, True])
    dt = np.array(
        [
            [0.0, 0, 10, 10],  # the real box
            [50, 50, 60, 60],  # the ignore box
            [50, 50, 60, 60],  # second hit on a non-crowd ignore box: a false positive
            [110, 110, 120, 120],  # inside the crowd region
            [150, 150, 160, 160],  # also inside: crowds absorb any number
        ]
    )
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    m = match_image(gt, ignore, crowd, dt, scores, max_dets=100, iou_thresholds=np.array([0.5]))
    assert m.n_pos == 1
    assert m.matched[0].tolist() == [True, True, False, True, True]
    assert m.ignored[0].tolist() == [False, True, False, True, True]


def test_real_box_is_preferred_over_an_ignore_region() -> None:
    gt = np.array([[0.0, 0, 10, 10], [0.0, 0, 10, 10]])
    m = match_image(
        gt,
        np.array([True, False]),
        np.array([False, False]),
        gt[:1],
        np.array([0.9]),
        100,
        np.array([0.5]),
    )
    assert m.matched[0, 0]
    assert not m.ignored[0, 0]


def test_max_dets_keeps_the_highest_scores() -> None:
    gt = np.array([[0.0, 0, 10, 10]])
    dt = np.array([[50.0, 50, 60, 60], [0.0, 0, 10, 10]])
    m = match_image(
        gt, np.array([False]), np.array([False]), dt, np.array([0.2, 0.9]), 1, np.array([0.5])
    )
    assert m.scores.tolist() == [0.9]
    assert m.matched[0].tolist() == [True]


def _one_image(
    gt_boxes: list[list[float]], dt_boxes: list[list[float]]
) -> tuple[list[ImageGt], list[ImagePred]]:
    g = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    d = np.asarray(dt_boxes, dtype=np.float64).reshape(-1, 4)
    gt = ImageGt(g, ["person"] * len(g), np.zeros(len(g), bool), np.zeros(len(g), bool))
    pr = ImagePred(d, np.linspace(0.9, 0.5, len(d)), ["person"] * len(d))
    return [gt], [pr]


def test_perfect_detector_scores_one_with_a_degenerate_interval() -> None:
    gts, preds = _one_image([[0, 0, 10, 10], [20, 20, 40, 40]], [[0, 0, 10, 10], [20, 20, 40, 40]])
    result = evaluate(match_dataset(gts * 5, preds * 5, ["person"], 100), 5, 200, seed=1)
    assert result.map50.value == 1.0
    assert result.map50_95.value == 1.0
    assert (result.map50_95.lo, result.map50_95.hi, result.map50_95.std) == (1.0, 1.0, 0.0)


def test_missed_class_scores_zero_and_unlabelled_class_is_left_out() -> None:
    gt = ImageGt(np.array([[0.0, 0, 10, 10]]), ["cat"], np.array([False]), np.array([False]))
    pr = ImagePred(np.array([[0.0, 0, 10, 10]]), np.array([0.9]), ["dog"])
    result = evaluate(match_dataset([gt], [pr], ["cat", "dog"], 100), 1, 10, seed=0)
    assert result.per_class == {"cat": {"ap50": 0.0, "ap50_95": 0.0, "n_pos": 1}}
    assert result.map50.value == 0.0


def test_iou_thresholds_separate_loose_from_tight_boxes() -> None:
    # IoU 0.64: a hit at 0.50 to 0.60, a miss above. AP50 = 1, AP50-95 = 3/10.
    gts, preds = _one_image([[0, 0, 10, 10]], [[0, 0, 8, 8]])
    result = evaluate(match_dataset(gts, preds, ["person"], 100), 1, 10, seed=0)
    assert result.map50.value == pytest.approx(1.0)
    assert result.map50_95.value == pytest.approx(0.3)
    assert len(IOU_THRESHOLDS) == 10


def test_bootstrap_is_seeded_and_brackets_the_estimate() -> None:
    gts, preds = random_dataset(7, n_images=30)
    matches = match_dataset(gts, preds, CLASSES, 100)
    a = evaluate(matches, len(gts), 300, seed=5)
    b = evaluate(matches, len(gts), 300, seed=5)
    c = evaluate(matches, len(gts), 300, seed=6)
    assert a.map50_95 == b.map50_95
    assert a.map50_95 != c.map50_95
    assert a.map50_95.lo < a.map50_95.value < a.map50_95.hi
    assert 0 < a.map50_95.std < 0.2


def test_empty_dataset_has_no_score() -> None:
    result = evaluate(match_dataset([], [], ["person"], 100), 0, 0, seed=0)
    assert np.isnan(result.map50.value)
    assert result.per_class == {}


def test_one_prediction_set_per_image_is_required() -> None:
    gts, preds = _one_image([[0, 0, 1, 1]], [])
    with pytest.raises(ValueError, match="one prediction set"):
        match_dataset(gts, preds * 2, ["person"], 100)
