"""Object detector behind a small interface, so the worker is tested with fakes.

`YoloDetector` runs Ultralytics YOLO11 in-process with PyTorch (the worker calls it at batch
size 1, the Phase 1 baseline). `OnnxDetector` (vms.worker.onnx_detector, task 5.1) runs the same
model exported to ONNX on ONNX Runtime, in FP32 or statically quantized to INT8
(vms.worker.quantize, task 5.2); `detector.backend` and `detector.precision` choose, and D5
decides the default. Both have `detect_batch` for the precision ladder (vms.worker.ladder).
`detect` is CPU or GPU bound and blocking: the worker calls it from its own single-thread
executor, never on the event loop. `build_detector` is the one place that turns settings into
a detector.

Detections are ordered by confidence (descending, ties by position), so `det_idx`, part of the
detections key, is stable for a given model output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from vms.worker.onnx_detector import OnnxDetector

# Before ultralytics is imported anywhere: no usage analytics, no pip installs at runtime, no
# console banners. ONLINE=False also stops its update checks.
os.environ.setdefault("YOLO_OFFLINE", "1")
os.environ.setdefault("YOLO_AUTOINSTALL", "0")
os.environ.setdefault("YOLO_VERBOSE", "0")


@dataclass(frozen=True)
class Detection:
    class_name: str
    confidence: float
    # Box corners in pixels of the input image.
    x1: float
    y1: float
    x2: float
    y2: float


class Detector(Protocol):
    @property
    def model_version(self) -> str:
        """Recorded on every frame and detection, e.g. `yolo11s@85a76fe86dd8`."""
        ...

    def detect(self, image: NDArray[np.uint8]) -> list[Detection]:
        """Detect objects in one BGR image (H x W x 3, uint8)."""
        ...


class DetectorSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    weights: str = "yolo11s"  # a key of configs/models.yaml
    # ultralytics: PyTorch through Ultralytics' predictor; onnxruntime: the ONNX export
    # (make onnx) with numpy pre/post-processing.
    backend: Literal["ultralytics", "onnxruntime"] = "ultralytics"
    # fp32, or int8: the ONNX export statically quantized on the calibration set
    # (vms.worker.quantize, make quantize; onnxruntime only).
    precision: Literal["fp32", "int8"] = "fp32"
    device: Literal["cpu", "cuda"] = "cpu"
    imgsz: int = Field(default=640, ge=32, multiple_of=32)
    conf: float = Field(default=0.25, ge=0.0, le=1.0)  # minimum confidence kept
    iou: float = Field(default=0.7, ge=0.0, le=1.0)  # NMS IoU threshold
    max_det: int = Field(default=300, ge=1)
    # Intra-op threads of the backend; None: its default (one per physical core for both).
    threads: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def _int8_needs_onnxruntime(self) -> DetectorSettings:
        if self.precision == "int8" and self.backend != "onnxruntime":
            raise ValueError("precision 'int8' needs backend 'onnxruntime'")
        return self


def to_detections(
    xyxy: NDArray[np.floating[Any]],
    conf: NDArray[np.floating[Any]],
    cls: NDArray[np.floating[Any]] | NDArray[np.integer[Any]],
    names: dict[int, str],
) -> list[Detection]:
    """Model outputs (N x 4 boxes, N scores, N class ids) to detections, highest score first."""
    rows = [
        Detection(names[int(c)], float(s), float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        for b, s, c in zip(xyxy, conf, cls, strict=True)
    ]
    rows.sort(key=lambda d: (-d.confidence, d.x1, d.y1, d.class_name))
    return rows


class YoloDetector:
    def __init__(self, weights_path: Path, sha256: str, settings: DetectorSettings) -> None:
        import importlib

        import torch

        # ultralytics is untyped where it matters (predict returns a union); treat it as Any
        # and convert its outputs in `to_detections`, which is typed and unit tested.
        yolo: Any = importlib.import_module("ultralytics").YOLO

        if settings.threads is not None:
            torch.set_num_threads(settings.threads)
        if settings.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("device 'cuda' requested but torch.cuda.is_available() is False")
        self._settings = settings
        self._model: Any = yolo(str(weights_path), task="detect")
        self._names: dict[int, str] = dict(self._model.names)
        self._version = f"{settings.weights}@{sha256[:12]}"

    @property
    def model_version(self) -> str:
        return self._version

    def detect(self, image: NDArray[np.uint8]) -> list[Detection]:
        return self.detect_batch([image])[0]

    def detect_batch(self, images: list[NDArray[np.uint8]]) -> list[list[Detection]]:
        """Several images in one forward pass: Ultralytics stacks a list source into one batch
        (letterboxed to the stride rectangle when all images share a shape, as OnnxDetector)."""
        s = self._settings
        results: Any = self._model.predict(
            images,
            imgsz=s.imgsz,
            conf=s.conf,
            iou=s.iou,
            max_det=s.max_det,
            device=s.device,
            batch=len(images),
            verbose=False,
        )
        out = []
        for result in results:
            boxes = result.boxes
            out.append(
                to_detections(
                    boxes.xyxy.cpu().numpy(),
                    boxes.conf.cpu().numpy(),
                    boxes.cls.cpu().numpy(),
                    self._names,
                )
            )
        return out

    def torch_module(self) -> Any:
        """The fused, eval-mode PyTorch module the predictor runs (built on the first predict;
        call `warmup` first). For parity checks against exported models."""
        predictor = self._model.predictor
        if predictor is None:
            raise RuntimeError("call warmup() first: the predictor is built on the first predict")
        return predictor.model.model

    def warmup(self, height: int = 720, width: int = 1280) -> None:
        """One inference on a blank frame: model fusion, allocator and kernel selection happen
        here instead of on the first real frame."""
        self.detect(np.zeros((height, width, 3), dtype=np.uint8))


def build_detector(settings: DetectorSettings, root: Path = Path()) -> YoloDetector | OnnxDetector:
    """The configured backend with verified weights (fetched if missing). The ONNX backend
    needs an export made from the same weights (`make onnx`); it is not exported here, because
    the worker's model directories are read-only in its container."""
    from vms.worker.weights import ensure_weights, load_models_config

    models = load_models_config(root / "configs/models.yaml")
    name = settings.weights
    path = ensure_weights(models, name, root=root)
    sha = models.weights[name].sha256
    if settings.backend == "ultralytics":
        return YoloDetector(path, sha, settings)
    from vms.worker.onnx_detector import OnnxDetector, verified_export

    model_path, meta = verified_export(root / models.onnx_dir, name, sha)
    if settings.precision == "int8":
        from vms.worker.quantize import verified_int8

        int8_path, int8_meta = verified_int8(root / models.onnx_dir, name, sha, meta.onnx_sha256)
        return OnnxDetector(int8_path, sha, settings, variant=f"int8@{int8_meta.int8_sha256[:8]}")
    return OnnxDetector(model_path, sha, settings)
