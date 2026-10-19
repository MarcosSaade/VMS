"""YOLO11 as a TensorRT engine, built on the GPU it runs on (task 5.2 GPU rungs, D38).

TensorRT 11 builds strongly typed networks only: the precision of every layer comes from the
ONNX graph, not from builder flags. So each precision has its own source model:

- fp16: the FP16 ONNX export (`export_onnx(half=True)`, made on a GPU).
- int8: the FP32 export statically quantized with symmetric s8 Q/DQ pairs on the same
  calibration set as the CPU INT8 step (`quantize --target tensorrt`). Quantized convs run in
  INT8; the excluded head decode and anything else without Q/DQ runs in FP32.

`prepare` makes whatever is missing (export, quantized model, engine) and writes a sidecar per
engine; `verified_engine` refuses an engine whose source model, TensorRT version or GPU differ
from now, because an engine is only valid for the GPU and TensorRT version that built it.

`TrtDetector` keeps `OnnxDetector`'s numpy pre- and post-processing (letterbox, class-aware
NMS), so the backends differ only in the forward pass. I/O goes through PyTorch CUDA tensors,
already in the environment, instead of a second CUDA binding. `detect_batch` returns host
arrays, so its timing includes the copies and the synchronization.
"""

from __future__ import annotations

import importlib
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from vms.common.settings import ServiceSettings
from vms.worker.detector import Detection, DetectorSettings, to_detections
from vms.worker.onnx_detector import (
    F32,
    StaleExportError,
    postprocess,
    preprocess,
    scale_boxes,
    sha256_file,
)

TrtPrecision = Literal["fp16", "int8"]


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TrtBuildSettings(ServiceSettings):
    """Engine build settings (configs/tensorrt.yaml)."""

    service: str = "tensorrt"
    # Optimization profile of the dynamic input (batch, 3, height, width). TensorRT tunes its
    # kernels for `opt` and accepts anything between `min` and `max`.
    min_shape: tuple[int, int, int, int] = (1, 3, 64, 64)
    opt_shape: tuple[int, int, int, int] = (4, 3, 384, 640)
    max_shape: tuple[int, int, int, int] = (32, 3, 640, 640)
    workspace_gb: float = Field(default=4.0, gt=0)


class EngineMeta(BaseModel):
    """Sidecar `<name>.<precision>.engine.json`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    weights: str
    weights_sha256: str
    precision: TrtPrecision
    source_onnx: str  # file name under the ONNX directory
    source_onnx_sha256: str
    engine_sha256: str
    tensorrt: str
    gpu: str
    input_name: str
    output_name: str
    names: dict[int, str]
    stride: int
    profile: dict[str, list[int]]
    build_s: float
    created_at: datetime


def _trt() -> Any:
    # tensorrt is installed only on GPU hosts (scripts/pod_remote.sh) and ships no types.
    return importlib.import_module("tensorrt")


def engine_paths(trt_dir: Path, name: str, precision: TrtPrecision) -> tuple[Path, Path]:
    engine = trt_dir / f"{name}.{precision}.engine"
    return engine, engine.with_name(engine.name + ".json")


def gpu_name() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("TensorRT needs a CUDA GPU and torch.cuda.is_available() is False")
    return str(torch.cuda.get_device_name(0))


def source_model(
    onnx_dir: Path, name: str, weights_sha256: str, precision: TrtPrecision
) -> tuple[Path, str]:
    """The verified ONNX model an engine of `precision` is built from, and its sha256."""
    from vms.worker.onnx_detector import verified_export

    if precision == "fp16":
        path, meta = verified_export(onnx_dir, name, weights_sha256, half=True)
        return path, meta.onnx_sha256
    from vms.worker.quantize import verified_int8

    _, fp32 = verified_export(onnx_dir, name, weights_sha256)
    path, int8 = verified_int8(onnx_dir, name, weights_sha256, fp32.onnx_sha256, "tensorrt")
    return path, int8.int8_sha256


def model_metadata(onnx_path: Path) -> tuple[dict[int, str], int]:
    """Class names and stride from the export's metadata (Ultralytics writes both)."""
    import ast

    import onnx

    meta = {
        p.key: p.value for p in onnx.load(str(onnx_path), load_external_data=False).metadata_props
    }
    names = {int(k): str(v) for k, v in ast.literal_eval(meta["names"]).items()}
    return names, int(meta.get("stride", 32))


def build_engine(
    onnx_path: Path, engine_path: Path, settings: TrtBuildSettings
) -> tuple[str, str, float]:
    """Parse `onnx_path` into a strongly typed network and write the serialized engine.
    Returns the input and output tensor names and the build time in seconds."""
    trt = _trt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    if hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"):
        flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError(f"TensorRT could not parse {onnx_path}: {errors}")
    input_name = network.get_input(0).name
    output_name = network.get_output(0).name
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, int(settings.workspace_gb * (1 << 30))
    )
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, settings.min_shape, settings.opt_shape, settings.max_shape)
    config.add_optimization_profile(profile)
    t0 = time.perf_counter()
    plan = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - t0
    if plan is None:
        raise RuntimeError(f"TensorRT failed to build an engine from {onnx_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(bytes(plan))
    return input_name, output_name, build_s


def verified_engine(
    trt_dir: Path, onnx_dir: Path, name: str, weights_sha256: str, precision: str
) -> tuple[Path, EngineMeta]:
    """The engine for `name` and `precision`, checked against its source model, the installed
    TensorRT and this GPU."""
    if precision not in ("fp16", "int8"):
        raise ValueError(f"no TensorRT engine for precision {precision!r} (fp16 or int8)")
    prec: TrtPrecision = "fp16" if precision == "fp16" else "int8"
    engine_path, meta_path = engine_paths(trt_dir, name, prec)
    if not engine_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"{engine_path} or its sidecar is missing; run `make trt-prepare`")
    meta = EngineMeta.model_validate_json(meta_path.read_text(encoding="utf-8"))
    _, source_sha = source_model(onnx_dir, name, weights_sha256, prec)
    stale = [
        what
        for what, ok in (
            ("source model", meta.source_onnx_sha256 == source_sha),
            ("TensorRT version", meta.tensorrt == _trt().__version__),
            ("GPU", meta.gpu == gpu_name()),
            ("engine file", sha256_file(engine_path) == meta.engine_sha256),
        )
        if not ok
    ]
    if stale:
        raise StaleExportError(
            f"{engine_path} does not match this {', '.join(stale)}; run `make trt-prepare`"
        )
    return engine_path, meta


def prepare(name: str, precision: str, root: Path = Path()) -> Path:
    """Make what the `precision` engine of `name` needs and is missing or stale: the FP16
    export or the TensorRT INT8 model, then the engine. Returns the engine path."""
    from vms.common.settings import load_settings
    from vms.worker.onnx_detector import export_onnx, verified_export
    from vms.worker.quantize import quantize_from_config
    from vms.worker.weights import ensure_weights, load_models_config

    prec: TrtPrecision = "fp16" if precision == "fp16" else "int8"
    models = load_models_config(root / "configs/models.yaml")
    sha = models.weights[name].sha256
    weights = ensure_weights(models, name, root=root)
    onnx_dir, trt_dir = root / models.onnx_dir, root / models.trt_dir
    imgsz = DetectorSettings().imgsz

    def have(check: Any) -> bool:
        try:
            check()
        except (FileNotFoundError, StaleExportError):
            return False
        return True

    if prec == "fp16":
        if not have(lambda: verified_export(onnx_dir, name, sha, half=True)):
            export_onnx(weights, sha, name, onnx_dir, imgsz, half=True)
    else:
        if not have(lambda: verified_export(onnx_dir, name, sha)):
            export_onnx(weights, sha, name, onnx_dir, imgsz)
        if not have(lambda: source_model(onnx_dir, name, sha, "int8")):
            quantize_from_config([name], "tensorrt")
    if have(lambda: verified_engine(trt_dir, onnx_dir, name, sha, prec)):
        return engine_paths(trt_dir, name, prec)[0]

    settings = load_settings(TrtBuildSettings, "tensorrt", config_dir=root / "configs")
    source, source_sha = source_model(onnx_dir, name, sha, prec)
    engine_path, meta_path = engine_paths(trt_dir, name, prec)
    input_name, output_name, build_s = build_engine(source, engine_path, settings)
    names, stride = model_metadata(source)
    meta = EngineMeta(
        weights=name,
        weights_sha256=sha,
        precision=prec,
        source_onnx=source.name,
        source_onnx_sha256=source_sha,
        engine_sha256=sha256_file(engine_path),
        tensorrt=str(_trt().__version__),
        gpu=gpu_name(),
        input_name=input_name,
        output_name=output_name,
        names=names,
        stride=stride,
        profile={
            "min": list(settings.min_shape),
            "opt": list(settings.opt_shape),
            "max": list(settings.max_shape),
        },
        build_s=round(build_s, 1),
        created_at=datetime.now(UTC),
    )
    meta_path.write_text(meta.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return engine_path


class TrtDetector:
    """`Detector` on a TensorRT engine; `detect_batch` runs several images in one forward pass."""

    def __init__(
        self,
        engine_path: Path,
        meta: EngineMeta,
        weights_sha256: str,
        settings: DetectorSettings,
    ) -> None:
        import torch

        trt = _trt()
        self._torch = torch
        self._settings = settings
        self._meta = meta
        self._runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self._engine: Any = self._runtime.deserialize_cuda_engine(engine_path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"TensorRT could not load {engine_path}")
        self._context: Any = self._engine.create_execution_context()
        dtypes = {trt.DataType.FLOAT: torch.float32, trt.DataType.HALF: torch.float16}
        self._in_dtype = dtypes[self._engine.get_tensor_dtype(meta.input_name)]
        self._out_dtype = dtypes[self._engine.get_tensor_dtype(meta.output_name)]
        self._stream = torch.cuda.Stream()  # type: ignore[no-untyped-call]
        self._version = (
            f"{settings.weights}@{weights_sha256[:12]}+trt-{meta.precision}"
            f"@{meta.engine_sha256[:8]}"
        )

    @property
    def model_version(self) -> str:
        return self._version

    @property
    def names(self) -> dict[int, str]:
        return dict(self._meta.names)

    def forward(self, x: F32) -> F32:
        """Raw head output `(batch, 4 + classes, anchors)` for a preprocessed batch."""
        torch = self._torch
        m = self._meta
        with torch.cuda.stream(self._stream):
            inp = torch.from_numpy(x).to("cuda", non_blocking=True).to(self._in_dtype)
            self._context.set_input_shape(m.input_name, tuple(inp.shape))
            out = torch.empty(
                tuple(self._context.get_tensor_shape(m.output_name)),
                dtype=self._out_dtype,
                device="cuda",
            )
            self._context.set_tensor_address(m.input_name, inp.data_ptr())
            self._context.set_tensor_address(m.output_name, out.data_ptr())
            if not self._context.execute_async_v3(self._stream.cuda_stream):
                raise RuntimeError("TensorRT execution failed")
            host = out.float().cpu()
        self._stream.synchronize()
        return np.asarray(host.numpy(), dtype=np.float32)

    def detect_batch(self, images: list[NDArray[np.uint8]]) -> list[list[Detection]]:
        s = self._settings
        x, placements = preprocess(images, s.imgsz, self._meta.stride)
        raw = self.forward(x)
        results = []
        for r, placement in zip(raw, placements, strict=True):
            boxes, scores, classes = postprocess(r, s.conf, s.iou, s.max_det)
            results.append(
                to_detections(scale_boxes(boxes, placement), scores, classes, self._meta.names)
            )
        return results

    def detect(self, image: NDArray[np.uint8]) -> list[Detection]:
        return self.detect_batch([image])[0]

    def warmup(self, height: int = 720, width: int = 1280) -> None:
        self.detect(np.zeros((height, width, 3), dtype=np.uint8))
