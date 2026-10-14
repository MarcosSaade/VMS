"""ONNX Runtime vs PyTorch parity with the pinned pretrained weights (task 5.1).

Integration because the weights are downloaded (cached in CI). The images are the two photos
shipped inside the ultralytics package, as stored and resized to the 1280x720 workload frame;
`make onnx-parity` runs the same comparison over the detection eval subset and data-mini.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import cv2
import numpy as np
import pytest

from vms.worker.detector import DetectorSettings
from vms.worker.onnx_detector import export_onnx
from vms.worker.parity import Comparator, Tolerances
from vms.worker.weights import ensure_weights, load_models_config

pytestmark = pytest.mark.integration


def sample_images() -> list[np.ndarray]:
    assets = Path(importlib.import_module("ultralytics").__file__).parent / "assets"
    images = [cv2.imread(str(assets / name)) for name in ("bus.jpg", "zidane.jpg")]
    return [*images, *(cv2.resize(im, (1280, 720)) for im in images)]


@pytest.mark.parametrize("weights", ["yolo11n", "yolo11s"])
def test_pretrained_onnx_export_matches_pytorch(weights: str, tmp_path: Path) -> None:
    models = load_models_config()
    sha = models.weights[weights].sha256
    weights_path = ensure_weights(models, weights)
    model = export_onnx(weights_path, sha, weights, tmp_path, 640)
    tol = Tolerances()
    comparator = Comparator(
        weights_path, model, sha, DetectorSettings(weights=weights), [0.25, 0.001], tol
    )
    stats = comparator.run(sample_images())
    assert stats.letterbox_mismatches == 0
    assert stats.raw_max_box_diff <= tol.raw_box_atol
    assert stats.raw_max_score_diff <= tol.raw_score_atol
    for diff in stats.detections.values():
        assert diff.matched > 0
        assert not diff.unexplained, diff.unexplained
    # The bus photo: people and a bus, found by both runtimes.
    found = {d.class_name for d in comparator.onnx[0.25].detect(sample_images()[0])}
    assert {"person", "bus"} <= found
