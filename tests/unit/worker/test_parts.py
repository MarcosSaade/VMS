"""Input statistics, detector output conversion, weights fetching and worker settings."""

from __future__ import annotations

import hashlib
from pathlib import Path

import cv2
import numpy as np
import pytest
from pydantic import ValidationError

from tests.unit.datasets.conftest import FakeTransport
from vms.common.settings import load_settings
from vms.datasets.http import ChecksumError, RetryPolicy
from vms.worker.detector import DetectorSettings, to_detections
from vms.worker.settings import QueueSettings, WorkerSettings
from vms.worker.stats import input_stats
from vms.worker.weights import ModelsConfig, ensure_weights, load_models_config

REPO_ROOT = Path(__file__).resolve().parents[3]


def checkerboard(size: int = 64, cell: int = 4) -> np.ndarray:
    tile = (np.indices((size, size)).sum(axis=0) // cell) % 2
    gray = (tile * 255).astype(np.uint8)
    return np.dstack([gray] * 3)


def test_stats_of_a_flat_image() -> None:
    stats = input_stats(np.full((32, 32, 3), 100, dtype=np.uint8))
    assert stats.brightness == pytest.approx(100.0)
    assert stats.contrast == pytest.approx(0.0)
    assert stats.blur_var == pytest.approx(0.0)


def test_stats_react_to_darkening_and_blur() -> None:
    sharp = checkerboard()
    base = input_stats(sharp)
    darker = input_stats((sharp * 0.5).astype(np.uint8))
    blurred = input_stats(cv2.GaussianBlur(sharp, (9, 9), 3))
    assert darker.brightness < base.brightness
    assert darker.contrast < base.contrast
    assert blurred.blur_var < base.blur_var / 10
    assert blurred.brightness == pytest.approx(base.brightness, rel=0.05)


def test_to_detections_maps_classes_and_orders_by_confidence() -> None:
    xyxy = np.array([[0, 0, 10, 10], [5, 5, 20, 20], [1, 1, 2, 2]], dtype=np.float32)
    conf = np.array([0.4, 0.9, 0.4], dtype=np.float32)
    cls = np.array([2.0, 0.0, 0.0], dtype=np.float32)
    dets = to_detections(xyxy, conf, cls, {0: "person", 2: "car"})
    assert [(d.class_name, round(d.confidence, 2), d.x1) for d in dets] == [
        ("person", 0.9, 5.0),
        ("car", 0.4, 0.0),  # tie on confidence: leftmost first, so det_idx is stable
        ("person", 0.4, 1.0),
    ]
    assert all(isinstance(d.x2, float) for d in dets)
    assert to_detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0), {}) == []


def test_detector_settings_validation() -> None:
    with pytest.raises(ValidationError):
        DetectorSettings(imgsz=650)  # not a multiple of the model stride
    with pytest.raises(ValidationError):
        DetectorSettings(device="tpu")  # type: ignore[arg-type]


def test_committed_configs_load() -> None:
    settings = load_settings(WorkerSettings, "worker", config_dir=REPO_ROOT / "configs")
    assert (
        settings.detector.weights in load_models_config(REPO_ROOT / "configs/models.yaml").weights
    )
    assert settings.queue.renew_every_s < settings.queue.lease_ttl_s


def test_queue_settings_validation() -> None:
    with pytest.raises(ValidationError, match="renew_every_s"):
        QueueSettings(lease_ttl_s=5, renew_every_s=5)
    with pytest.raises(ValidationError, match="poll_min_s"):
        QueueSettings(poll_min_s=2, poll_max_s=1)


def models_config(data: bytes) -> ModelsConfig:
    return ModelsConfig.model_validate(
        {
            "weights_dir": "w",
            "weights": {
                "tiny": {
                    "url": "https://example.org/tiny.pt",
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "license": "test",
                }
            },
        }
    )


def test_ensure_weights_downloads_once_and_verifies(tmp_path: Path) -> None:
    data = b"weights" * 1000
    transport = FakeTransport({"https://example.org/tiny.pt": data})
    retry = RetryPolicy(attempts=2, base_s=0.0, sleep=lambda _: None)
    config = models_config(data)
    path = ensure_weights(config, "tiny", root=tmp_path, transport=transport, retry=retry)
    assert path == tmp_path / "w" / "tiny.pt"
    assert path.read_bytes() == data
    ensure_weights(config, "tiny", root=tmp_path, transport=transport, retry=retry)
    assert len(transport.requests) == 1  # the cached copy was re-verified, not re-downloaded


def test_ensure_weights_rejects_a_tampered_file(tmp_path: Path) -> None:
    config = models_config(b"the real weights")
    transport = FakeTransport({"https://example.org/tiny.pt": b"evil weights!!!!"})
    retry = RetryPolicy(attempts=1, base_s=0.0, sleep=lambda _: None)
    with pytest.raises(ChecksumError):
        ensure_weights(config, "tiny", root=tmp_path, transport=transport, retry=retry)
    assert not (tmp_path / "w" / "tiny.pt").exists()


def test_unknown_weights_name(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="configured: tiny"):
        ensure_weights(models_config(b"x"), "yolo99", root=tmp_path)
