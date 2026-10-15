"""ONNX backend (task 5.1): pre/post-processing against Ultralytics, export, parity.

Every reference here is Ultralytics' own code (LetterBox, non_max_suppression, scale_boxes) or
torchvision's NMS, run on the same inputs. No network: the end-to-end parity test exports a
randomly initialized YOLO11n built from its YAML (fixed seed); parity with the pretrained
weights is an integration test and `make onnx-parity`.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import torch
import torchvision
from ultralytics.data.augment import LetterBox
from ultralytics.utils import nms as ul_nms
from ultralytics.utils import ops as ul_ops

from vms.worker.detector import Detection, DetectorSettings, YoloDetector, build_detector
from vms.worker.onnx_detector import (
    Letterbox,
    OnnxDetector,
    StaleExportError,
    letterbox,
    nms,
    onnx_paths,
    postprocess,
    preprocess,
    scale_boxes,
    verified_export,
)
from vms.worker.parity import Comparator, Tolerances, compare_detections, torch_forward

SHAPES = [(720, 1280), (480, 640), (375, 500), (2000, 100), (641, 639), (32, 32), (1080, 1920)]


def noise(shape: tuple[int, int], seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (*shape, 3), dtype=np.uint8)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("auto", [True, False])
def test_letterbox_matches_ultralytics_byte_for_byte(shape: tuple[int, int], auto: bool) -> None:
    image = noise(shape)
    ours = letterbox(image, 640, 32, auto)
    theirs = LetterBox((640, 640), auto=auto, stride=32)(image=image)
    assert ours.shape == theirs.shape
    assert np.array_equal(ours, theirs)


def test_preprocess_uses_the_stride_rectangle_only_for_one_shape() -> None:
    same, placements = preprocess([noise((720, 1280)), noise((720, 1280), 1)], 640, 32)
    assert same.shape == (2, 3, 384, 640)
    assert placements[0] == Letterbox((720, 1280), (384, 640))
    mixed, _ = preprocess([noise((720, 1280)), noise((480, 640))], 640, 32)
    assert mixed.shape == (2, 3, 640, 640)  # as Ultralytics: squares when shapes differ


def test_preprocess_is_rgb_float_in_unit_range() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[..., 0] = 255  # blue in BGR
    x, _ = preprocess([image], 64, 32)
    assert x.dtype == np.float32
    assert x.flags.c_contiguous
    assert x[0, 2].min() == 1.0  # blue ends up in channel 2
    assert x[0, 0].max() == 0.0


def random_boxes(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, 600, (n, 2))
    wh = rng.uniform(2, 120, (n, 2))
    boxes = np.concatenate([xy, xy + wh], axis=1).astype(np.float32)
    return boxes, rng.uniform(0, 1, n).astype(np.float32)


@pytest.mark.parametrize("iou", [0.3, 0.5, 0.7, 0.9])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_nms_matches_torchvision(iou: float, seed: int) -> None:
    boxes, scores = random_boxes(800, seed)
    ref = torchvision.ops.nms(torch.from_numpy(boxes), torch.from_numpy(scores), iou).numpy()
    assert np.array_equal(nms(boxes, scores, iou), ref)
    assert np.array_equal(nms(boxes, scores, iou, max_keep=10), ref[:10])


def test_nms_keeps_empty_boxes_as_torchvision_does() -> None:
    # Two zero-area boxes: IoU is 0/0. torchvision suppresses nothing; a naive `iou <= t`
    # filter would drop the second box.
    boxes = np.array([[5, 5, 5, 5], [5, 5, 5, 5], [0, 0, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    ref = torchvision.ops.nms(torch.from_numpy(boxes), torch.from_numpy(scores), 0.5).numpy()
    assert np.array_equal(nms(boxes, scores, 0.5), ref)


def random_head(seed: int, anchors: int = 5040, classes: int = 80) -> np.ndarray:
    """A raw head output `(4 + classes, anchors)`: xywh in a 640x384 input, sparse scores."""
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0, [640, 384], (anchors, 2))
    wh = rng.uniform(4, 200, (anchors, 2))
    scores = rng.uniform(0, 0.05, (anchors, classes))
    hot = rng.choice(anchors, 400, replace=False)
    scores[hot, rng.integers(0, 4, 400)] = rng.uniform(0.05, 1, 400)  # a few busy classes
    return np.concatenate([xy, wh, scores], axis=1).T.astype(np.float32)


@pytest.mark.parametrize(("conf", "max_det"), [(0.25, 300), (0.1, 300), (0.001, 300), (0.1, 7)])
@pytest.mark.parametrize("seed", [0, 1])
def test_postprocess_matches_ultralytics(conf: float, max_det: int, seed: int) -> None:
    raw = random_head(seed)
    boxes, scores, classes = postprocess(raw, conf, 0.7, max_det)
    placement = Letterbox((720, 1280), (384, 640))
    ours = scale_boxes(boxes, placement)

    (ref,) = ul_nms.non_max_suppression(torch.from_numpy(raw[None]), conf, 0.7, max_det=max_det)
    ref[:, :4] = ul_ops.scale_boxes((384, 640), ref[:, :4], (720, 1280))
    ref_np = ref.numpy()
    assert len(ours) == len(ref_np) > 0
    np.testing.assert_array_equal(classes, ref_np[:, 5].astype(int))
    np.testing.assert_array_equal(scores, ref_np[:, 4])
    np.testing.assert_allclose(ours, ref_np[:, :4], atol=1e-4)


def test_postprocess_caps_candidates_before_nms(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("vms.worker.onnx_detector.MAX_NMS", 50)
    raw = random_head(3)
    _, scores, _ = postprocess(raw, 0.0, 0.7, 300)
    assert len(scores) <= 50
    best = np.sort(raw[4:].max(0))[::-1]
    assert scores[0] == best[0]


def det(cls: str, conf: float, x1: float, y1: float = 0.0, w: float = 10.0) -> Detection:
    return Detection(cls, conf, x1, y1, x1 + w, y1 + w)


def test_compare_matches_within_tolerance_and_classifies_leftovers() -> None:
    tol = Tolerances()
    ref = [det("person", 0.9, 0), det("person", 0.10003, 50), det("car", 0.5, 100)]
    got = [det("person", 0.90001, 0.01), det("car", 0.5, 100)]
    diff = compare_detections(ref, got, conf=0.1, iou=0.7, tol=tol)
    assert diff.matched == 2
    assert diff.threshold_flips == 1  # 0.10003 sits within score_atol of conf=0.1
    assert not diff.unexplained
    assert diff.max_box_diff == pytest.approx(0.01)


def test_compare_explains_an_nms_flip_and_reports_the_rest() -> None:
    tol = Tolerances()
    # A 10x10 box and one shifted so their IoU is ~0.7: kept on one side only.
    a, b = det("person", 0.9, 0.0), det("person", 0.8, 10 * (1 - 0.7) / (1 + 0.7))
    diff = compare_detections([a, b], [a], conf=0.1, iou=0.7, tol=tol)
    assert diff.nms_flips == 1
    assert not diff.unexplained
    lost = compare_detections([a, det("dog", 0.6, 300)], [a], conf=0.1, iou=0.7, tol=tol)
    assert [u["class"] for u in lost.unexplained] == ["dog"]


def test_export_writes_a_sidecar_that_verifies(random_yolo: tuple[Path, Path, str]) -> None:
    _, model, sha = random_yolo
    path, meta = verified_export(model.parent, "rand", sha)
    assert path == model
    assert meta.weights_sha256 == sha
    assert meta.dynamic
    assert meta.opset >= 11
    with pytest.raises(StaleExportError, match="other rand weights"):
        verified_export(model.parent, "rand", "0" * 64)
    with pytest.raises(FileNotFoundError, match="make onnx"):
        verified_export(model.parent, "missing", sha)


def test_tampered_export_is_rejected(random_yolo: tuple[Path, Path, str], tmp_path: Path) -> None:
    _, model, sha = random_yolo
    copy, sidecar = onnx_paths(tmp_path, "rand")
    copy.write_bytes(model.read_bytes() + b"\0")
    sidecar.write_bytes(onnx_paths(model.parent, "rand")[1].read_bytes())
    with pytest.raises(StaleExportError, match="sidecar hash"):
        verified_export(tmp_path, "rand", sha)


@pytest.mark.parametrize("shape", [(720, 1280), (480, 640), (500, 375)])
def test_raw_and_detection_parity_with_pytorch(
    random_yolo: tuple[Path, Path, str], shape: tuple[int, int]
) -> None:
    weights, model, sha = random_yolo
    # Random weights score every anchor below ~2e-4; this threshold keeps a few percent.
    tol = Tolerances()
    comparator = Comparator(weights, model, sha, DetectorSettings(weights="rand"), [1e-4], tol)
    stats = comparator.run([noise(shape, 7)])
    assert stats.letterbox_mismatches == 0
    assert stats.raw_max_box_diff <= tol.raw_box_atol
    assert stats.raw_max_score_diff <= tol.raw_score_atol
    diff = stats.detections["conf=0.0001"]
    assert diff.matched > 0
    assert not diff.unexplained
    assert stats.ok(tol)


def test_batch_gives_the_same_detections_as_one_at_a_time(
    random_yolo: tuple[Path, Path, str],
) -> None:
    _, model, sha = random_yolo
    onnx = OnnxDetector(model, sha, DetectorSettings(weights="rand", conf=1e-4))
    images = [noise((720, 1280), s) for s in range(3)]
    for image, batched in zip(images, onnx.detect_batch(images), strict=True):
        diff = compare_detections(onnx.detect(image), batched, 1e-4, 0.7, Tolerances())
        assert diff.matched == len(batched) > 0
        assert not diff.unexplained


def test_torch_forward_needs_a_warm_reference(random_yolo: tuple[Path, Path, str]) -> None:
    weights, _, sha = random_yolo
    reference = YoloDetector(weights, sha, DetectorSettings(weights="rand"))
    with pytest.raises(RuntimeError, match="warmup"):
        torch_forward(reference, np.zeros((1, 3, 64, 64), dtype=np.float32))


def test_onnx_detector_reports_its_backend_in_the_model_version(
    random_yolo: tuple[Path, Path, str],
) -> None:
    _, model, sha = random_yolo
    onnx = OnnxDetector(model, sha, DetectorSettings(weights="rand"))
    assert onnx.model_version == f"rand@{sha[:12]}+onnx"
    assert len(onnx.names) == 80  # from the export metadata (generic names for a YAML model)


def test_build_detector_needs_an_export_for_onnxruntime(tmp_path: Path) -> None:
    payload = b"not really weights"
    sha = hashlib.sha256(payload).hexdigest()
    (tmp_path / "configs").mkdir()
    (tmp_path / "models/weights").mkdir(parents=True)
    (tmp_path / "models/weights/tiny.pt").write_bytes(payload)
    (tmp_path / "configs/models.yaml").write_text(
        "weights_dir: models/weights\nonnx_dir: models/onnx\nweights:\n  tiny:\n"
        f"    url: https://example.invalid/tiny.pt\n    bytes: {len(payload)}\n"
        f"    sha256: {sha}\n    license: test\n",
        encoding="utf-8",
    )
    settings = DetectorSettings(weights="tiny", backend="onnxruntime")
    with pytest.raises(FileNotFoundError, match="make onnx"):
        build_detector(settings, tmp_path)


@pytest.mark.parametrize(
    ("ref", "got"), [([], []), ([det("person", 0.9, 0)], []), ([], [det("car", 0.5, 0)])]
)
def test_compare_handles_images_without_detections(
    ref: list[Detection], got: list[Detection]
) -> None:
    diff = compare_detections(ref, got, conf=0.25, iou=0.7, tol=Tolerances())
    assert diff.matched == 0
    assert len(diff.unexplained) == len(ref) + len(got)


def test_pytorch_batch_gives_the_same_detections_as_one_at_a_time(
    random_yolo: tuple[Path, Path, str],
) -> None:
    weights, _, sha = random_yolo
    detector = YoloDetector(weights, sha, DetectorSettings(weights="rand", conf=1e-4))
    images = [noise((720, 1280), s) for s in range(3)]
    batched = detector.detect_batch(images)
    assert len(batched) == 3
    for image, got in zip(images, batched, strict=True):
        diff = compare_detections(detector.detect(image), got, 1e-4, 0.7, Tolerances())
        assert diff.matched == len(got) > 0
        assert not diff.unexplained
