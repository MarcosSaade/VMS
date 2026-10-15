"""INT8 static quantization with the pinned pretrained weights and the real calibration set
(task 5.2). Integration because the weights and calibration images are downloaded (cached).
A smoke check that a quantized model still finds the obvious objects; its mAP drop is measured
on the eval subset (`make eval SUITE=detection PRECISION=int8`, D32)."""

from __future__ import annotations

import importlib
from pathlib import Path

import cv2
import pytest

from vms.common.settings import load_settings
from vms.datasets.http import RetryPolicy, UrllibTransport
from vms.worker import calibration as cal
from vms.worker.detector import DetectorSettings
from vms.worker.onnx_detector import OnnxDetector, export_onnx, sha256_file
from vms.worker.quantize import QuantizeSettings, quantize_int8, verified_int8
from vms.worker.weights import ensure_weights, load_models_config

pytestmark = pytest.mark.integration


def test_int8_yolo11n_still_finds_the_people_and_the_bus(tmp_path: Path) -> None:
    settings = load_settings(QuantizeSettings, "quantize")
    p = settings.calibration
    spec = cal.load_spec(p.spec)
    images = cal.read_lock(p.lock)[:8] + cal.read_lock(p.lock)[-8:]  # 8 COCO + 8 MOT17
    cal.fetch(spec, images, UrllibTransport(), RetryPolicy(), p.images_dir, p.fetch_workers)

    models = load_models_config()
    sha = models.weights["yolo11n"].sha256
    fp32 = export_onnx(ensure_weights(models, "yolo11n"), sha, "yolo11n", tmp_path, 640)
    source_sha = sha256_file(fp32)
    path = quantize_int8(
        fp32,
        source_sha,
        "yolo11n",
        sha,
        cal.load_pixels(images, p.images_dir, spec),
        "test",
        settings.int8,
        tmp_path,
        640,
    )
    _, meta = verified_int8(tmp_path, "yolo11n", sha, source_sha)
    assert meta.calibration_images == 16
    assert meta.quantized_ops["Conv"] > 50

    bus = cv2.imread(
        str(Path(importlib.import_module("ultralytics").__file__).parent / "assets/bus.jpg")
    )
    det = DetectorSettings(weights="yolo11n", backend="onnxruntime", precision="int8", conf=0.25)
    int8 = OnnxDetector(path, sha, det)
    ref = OnnxDetector(fp32, sha, det.model_copy(update={"precision": "fp32"}))
    got = [d.class_name for d in int8.detect(bus)]
    want = [d.class_name for d in ref.detect(bus)]
    assert {"person", "bus"} <= set(got)
    assert abs(got.count("person") - want.count("person")) <= 1
