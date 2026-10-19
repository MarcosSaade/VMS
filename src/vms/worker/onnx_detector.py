"""YOLO11 exported to ONNX and run in-process with ONNX Runtime (task 5.1, D5).

Export goes through Ultralytics (`export_onnx`, `make onnx`): dynamic batch, height and width,
no graph simplifier, raw head output `(batch, 4 + classes, anchors)` without NMS. Inference
(`OnnxDetector`) does not use Ultralytics: pre- and post-processing are reimplemented in numpy
so the worker controls batching (5.3, 5.5) and the hot path has no PyTorch. Both follow
Ultralytics' predictor step by step, so detections match `YoloDetector` up to float error:

- pre: LetterBox to `imgsz` with the minimum stride-multiple rectangle when every image in the
  batch has the same shape (1280x720 -> 640x384), else a square; pad value 114; BGR -> RGB;
  uint8 -> float32 / 255.
- post: best class per anchor, keep confidence > `conf`, at most 30,000 candidates by score,
  class-aware NMS (boxes offset by class x 7680, greedy, suppress IoU > `iou`), first `max_det`,
  then undo the letterbox and clip to the image.

The parity tests (tests/unit/worker/test_onnx_detector.py, integration) and
`python -m vms.worker onnx-parity` check each step against Ultralytics.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import shutil
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from vms.worker.detector import Detection, DetectorSettings, to_detections

PAD_VALUE = 114
MAX_NMS = 30_000  # candidates kept for NMS, by score (Ultralytics' max_nms)
MAX_WH = 7_680  # per-class box offset for class-aware NMS (Ultralytics' max_wh)

F32 = NDArray[np.float32]


class OnnxExport(BaseModel):
    """Sidecar `<name>.onnx.json`: ties an exported file to the weights it came from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    weights: str
    weights_sha256: str
    onnx_sha256: str
    opset: int
    imgsz: int
    dynamic: bool
    half: bool = False  # FP16 weights and I/O (exported on a GPU, for TensorRT FP16; D38)
    packages: dict[str, str | None]
    exported_at: datetime


class StaleExportError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def onnx_paths(onnx_dir: Path, name: str) -> tuple[Path, Path]:
    model = onnx_dir / f"{name}.onnx"
    return model, model.with_name(model.name + ".json")


def export_stem(name: str, half: bool) -> str:
    """File stem of an export: `<name>` (FP32) or `<name>.fp16`."""
    return f"{name}.fp16" if half else name


def export_onnx(
    weights_path: Path,
    weights_sha256: str,
    name: str,
    onnx_dir: Path,
    imgsz: int,
    *,
    half: bool = False,
) -> Path:
    """Export `weights_path` to `<onnx_dir>/<name>.onnx` plus its sidecar; returns the model path.
    `half` exports FP16 weights and I/O to `<name>.fp16.onnx` instead; Ultralytics converts to
    FP16 only on a GPU, so that export runs on CUDA (D38).

    Ultralytics writes the export next to its input, so it runs on a copy in a temporary
    directory (the weights directory is read-only in the worker container). The file embeds
    an export timestamp, so two exports are not byte-identical; the sidecar records the hash.
    """
    yolo: Any = importlib.import_module("ultralytics").YOLO
    onnx_dir.mkdir(parents=True, exist_ok=True)
    model_path, meta_path = onnx_paths(onnx_dir, export_stem(name, half))
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / f"{name}.pt"
        shutil.copyfile(weights_path, src)
        model = yolo(str(src), task="detect")
        out = Path(
            model.export(
                format="onnx",
                dynamic=True,
                simplify=False,
                imgsz=imgsz,
                nms=False,
                half=half,
                device="cuda" if half else "cpu",
            )
        )
        shutil.move(out, model_path)
    import onnx

    opset = max(o.version for o in onnx.load(str(model_path)).opset_import if o.domain == "")
    from vms.bench.results import package_versions

    meta = OnnxExport(
        weights=name,
        weights_sha256=weights_sha256,
        onnx_sha256=sha256_file(model_path),
        opset=opset,
        imgsz=imgsz,
        dynamic=True,
        half=half,
        packages=package_versions(("torch", "ultralytics", "onnx", "onnxruntime")),
        exported_at=datetime.now(UTC),
    )
    meta_path.write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return model_path


def verified_export(
    onnx_dir: Path, name: str, weights_sha256: str, *, half: bool = False
) -> tuple[Path, OnnxExport]:
    """The exported model for `name` (`half`: the FP16 export), checked against its sidecar and
    the pinned weights."""
    model_path, meta_path = onnx_paths(onnx_dir, export_stem(name, half))
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"{model_path} or its sidecar is missing; run `make onnx`")
    meta = OnnxExport.model_validate_json(meta_path.read_text(encoding="utf-8"))
    if meta.weights_sha256 != weights_sha256:
        raise StaleExportError(
            f"{model_path} was exported from other {name} weights; run `make onnx` again"
        )
    if sha256_file(model_path) != meta.onnx_sha256:
        raise StaleExportError(f"{model_path} does not match its sidecar hash; run `make onnx`")
    return model_path, meta


# --- Pre-processing ------------------------------------------------------------------------


@dataclass(frozen=True)
class Letterbox:
    """How one image was placed in the network input."""

    orig_shape: tuple[int, int]  # (h, w) of the image
    input_shape: tuple[int, int]  # (h, w) of the padded network input


def letterbox(image: NDArray[np.uint8], imgsz: int, stride: int, auto: bool) -> NDArray[np.uint8]:
    """Ultralytics' LetterBox (center, scale up allowed, pad 114, bilinear resize)."""
    h, w = image.shape[:2]
    r = min(imgsz / h, imgsz / w)
    new_w, new_h = round(w * r), round(h * r)
    dw: float = imgsz - new_w
    dh: float = imgsz - new_h
    if auto:  # minimum rectangle: pad only up to the next stride multiple
        dw, dh = dw % stride, dh % stride
    dw, dh = dw / 2, dh / 2
    resized: Any = image
    if (w, h) != (new_w, new_h):
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    out = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3
    )
    return np.asarray(out, dtype=np.uint8)


def preprocess(
    images: list[NDArray[np.uint8]], imgsz: int, stride: int
) -> tuple[F32, list[Letterbox]]:
    """BGR uint8 images -> one float32 NCHW RGB batch in [0, 1], plus each image's placement.

    As Ultralytics: the stride rectangle only when all images share a shape, else squares.
    """
    auto = len({im.shape for im in images}) == 1
    padded = [letterbox(im, imgsz, stride, auto) for im in images]
    batch = np.stack(padded)[..., ::-1].transpose(0, 3, 1, 2)  # BHWC BGR -> BCHW RGB
    x = np.ascontiguousarray(batch).astype(np.float32) / np.float32(255.0)
    boxes = [Letterbox(im.shape[:2], p.shape[:2]) for im, p in zip(images, padded, strict=True)]
    return x, boxes


# --- Post-processing -----------------------------------------------------------------------


def nms(boxes: F32, scores: F32, iou_thres: float, max_keep: int | None = None) -> NDArray[np.intp]:
    """Greedy NMS as torchvision.ops.nms: highest score first, drop boxes with IoU > iou_thres
    against a kept box. Returns kept indices in score order (at most `max_keep`)."""
    order = np.argsort(-scores, kind="stable")
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    keep: list[int] = []
    with np.errstate(divide="ignore", invalid="ignore"):
        while order.size and (max_keep is None or len(keep) < max_keep):
            i = int(order[0])
            keep.append(i)
            rest = order[1:]
            w = np.maximum(np.float32(0), np.minimum(x2[i], x2[rest]) - np.maximum(x1[i], x1[rest]))
            h = np.maximum(np.float32(0), np.minimum(y2[i], y2[rest]) - np.maximum(y1[i], y1[rest]))
            inter = w * h
            iou = inter / (areas[i] + areas[rest] - inter)
            order = rest[~(iou > iou_thres)]  # a NaN IoU (two empty boxes) suppresses nothing
    return np.asarray(keep, dtype=np.intp)


def scale_boxes(boxes: F32, placement: Letterbox) -> F32:
    """Network-input xyxy -> image xyxy, clipped (Ultralytics' scale_boxes from the shapes)."""
    (h1, w1), (h0, w0) = placement.input_shape, placement.orig_shape
    gain = min(h1 / h0, w1 / w0)
    new_h, new_w = round(h0 * gain), round(w0 * gain)
    gain_y, gain_x = new_h / h0, new_w / w0
    pad_x, pad_y = round((w1 - new_w) / 2 - 0.1), round((h1 - new_h) / 2 - 0.1)
    out = boxes.copy()
    out[:, [0, 2]] = (out[:, [0, 2]] - pad_x) / gain_x
    out[:, [1, 3]] = (out[:, [1, 3]] - pad_y) / gain_y
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w0)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h0)
    return out.astype(np.float32, copy=False)


def postprocess(
    raw: F32, conf: float, iou: float, max_det: int
) -> tuple[F32, F32, NDArray[np.intp]]:
    """One image's raw head output `(4 + classes, anchors)` -> network-input xyxy boxes,
    scores and class ids after class-aware NMS, highest score first."""
    pred = raw.T  # (anchors, 4 + classes)
    cls_scores = pred[:, 4:]
    best = cls_scores.argmax(1)
    score = cls_scores[np.arange(len(best)), best]
    keep = score > conf
    xywh, score, best = pred[keep, :4], score[keep], best[keep]
    if len(score) > MAX_NMS:
        top = np.argsort(-score, kind="stable")[:MAX_NMS]
        xywh, score, best = xywh[top], score[top], best[top]
    half = xywh[:, 2:] / np.float32(2)
    xyxy = np.concatenate([xywh[:, :2] - half, xywh[:, :2] + half], axis=1)
    offset = (best.astype(np.float32) * np.float32(MAX_WH))[:, None]
    kept = nms(xyxy + offset, score, iou, max_det)
    return xyxy[kept], score[kept], best[kept]


def _ort() -> Any:
    # onnxruntime ships no type information; treat it as Any at this one boundary.
    return importlib.import_module("onnxruntime")


def onnx_session_options(threads: int | None, spin: bool = True) -> Any:
    ort = _ort()

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if threads is not None:
        opts.intra_op_num_threads = threads
    opts.add_session_config_entry("session.intra_op.allow_spinning", "1" if spin else "0")
    return opts


class OnnxDetector:
    """`Detector` on ONNX Runtime; `detect_batch` runs several images in one forward pass."""

    def __init__(
        self,
        model_path: Path,
        weights_sha256: str,
        settings: DetectorSettings,
        variant: str | None = None,
    ) -> None:
        """`variant` tags a derived model in `model_version` (e.g. `int8@<sha256 prefix>`)."""
        ort = _ort()
        providers = ["CPUExecutionProvider"]
        if settings.device == "cuda":
            # onnxruntime-gpu finds the CUDA and cuDNN libraries of the pip packages (the
            # ones PyTorch's CUDA build also uses) only once they are loaded (D38).
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise RuntimeError("device 'cuda' requested but onnxruntime has no CUDA provider")
            providers.insert(0, "CUDAExecutionProvider")
        self._settings = settings
        self._session: Any = ort.InferenceSession(
            str(model_path),
            onnx_session_options(settings.threads, settings.spin),
            providers=providers,
        )
        meta = self._session.get_modelmeta().custom_metadata_map
        self._names: dict[int, str] = {
            int(k): str(v) for k, v in ast.literal_eval(meta["names"]).items()
        }
        self._stride = int(meta.get("stride", 32))
        self._input = self._session.get_inputs()[0].name
        suffix = f"-{variant}" if variant else ""
        self._version = f"{settings.weights}@{weights_sha256[:12]}+onnx{suffix}"

    @property
    def model_version(self) -> str:
        return self._version

    @property
    def names(self) -> dict[int, str]:
        return dict(self._names)

    def forward(self, x: F32) -> F32:
        """Raw head output `(batch, 4 + classes, anchors)` for a preprocessed batch."""
        out = self._session.run(None, {self._input: x})[0]
        return np.asarray(out, dtype=np.float32)

    def detect_batch(self, images: list[NDArray[np.uint8]]) -> list[list[Detection]]:
        s = self._settings
        x, placements = preprocess(images, s.imgsz, self._stride)
        raw = self.forward(x)
        results = []
        for r, placement in zip(raw, placements, strict=True):
            boxes, scores, classes = postprocess(r, s.conf, s.iou, s.max_det)
            results.append(
                to_detections(scale_boxes(boxes, placement), scores, classes, self._names)
            )
        return results

    def detect(self, image: NDArray[np.uint8]) -> list[Detection]:
        return self.detect_batch([image])[0]

    def warmup(self, height: int = 720, width: int = 1280) -> None:
        self.detect(np.zeros((height, width, 3), dtype=np.uint8))
