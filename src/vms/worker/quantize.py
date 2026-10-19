"""Post-training static INT8 quantization of the ONNX export (task 5.2, D5, D32).

`quantize_int8` runs ONNX Runtime's static quantizer on `<name>.onnx` (make onnx): the
calibration images (vms.worker.calibration) go through the same letterbox as inference, the
FP32 model records each activation's range (`method`), and the quantizer writes
`<name>.int8.onnx` with QuantizeLinear/DequantizeLinear pairs around the quantized ops (QDQ),
which ONNX Runtime fuses into integer kernels. Nodes matching `exclude_nodes` stay FP32.

The sidecar `<name>.int8.onnx.json` ties the file to its FP32 export, the pinned weights, the
calibration list and the settings; `verified_int8` refuses a model whose inputs changed.
"""

from __future__ import annotations

import hashlib
import importlib
import re
import tempfile
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from vms.common.settings import ServiceSettings
from vms.worker.onnx_detector import StaleExportError, preprocess, sha256_file


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CalibrationPaths(_Section):
    spec: Path = Path("configs/calibration.v1.yaml")
    lock: Path = Path("configs/calibration.v1.lock.jsonl")
    images_dir: Path = Path("data/calibration/v1")
    fetch_workers: int = Field(default=8, ge=1)


class Int8Settings(_Section):
    format: Literal["qdq"] = "qdq"
    per_channel: bool = True
    activation_type: Literal["quint8", "qint8"] = "quint8"
    weight_type: Literal["qint8", "quint8"] = "qint8"
    method: Literal["minmax", "percentile", "entropy"] = "minmax"
    # Zero points of 0 for activations and weights: TensorRT reads Q/DQ pairs only when they
    # are symmetric and signed (D38).
    symmetric: bool = False
    # Op types to quantize; empty: every type ONNX Runtime can quantize. TensorRT takes Conv only
    # (D38): ORT also wraps scalar constants (attention scale) whose Q/DQ TensorRT rejects.
    op_types: list[str] = Field(default_factory=list)
    # Biases as int32 Q/DQ (ORT's default); TensorRT fuses FP32 biases itself.
    quantize_bias: bool = True
    exclude_nodes: list[str] = Field(default_factory=list)  # regexes on ONNX node names


Int8Target = Literal["onnxruntime", "tensorrt"]


class QuantizeSettings(ServiceSettings):
    service: str = "quantize"
    calibration: CalibrationPaths = Field(default_factory=CalibrationPaths)
    int8: Int8Settings = Field(default_factory=Int8Settings)  # for ONNX Runtime on the CPU
    # For TensorRT (D38): the same calibration set and excluded nodes, symmetric s8.
    int8_tensorrt: Int8Settings = Field(
        default_factory=lambda: Int8Settings(activation_type="qint8", symmetric=True)
    )

    def for_target(self, target: Int8Target) -> Int8Settings:
        return self.int8_tensorrt if target == "tensorrt" else self.int8


class Int8Export(BaseModel):
    """Sidecar `<name>.int8.onnx.json`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    weights: str
    target: Int8Target = "onnxruntime"
    weights_sha256: str
    source_onnx_sha256: str  # the FP32 export it was quantized from
    int8_sha256: str
    calibration_lock_sha256: str
    calibration_images: int
    settings: Int8Settings
    excluded_nodes: list[str]
    quantized_ops: dict[str, int]  # QDQ-wrapped op type -> count, from the written model
    packages: dict[str, str | None]
    created_at: datetime


def int8_paths(onnx_dir: Path, name: str, target: Int8Target = "onnxruntime") -> tuple[Path, Path]:
    """`<name>.int8.onnx` for ONNX Runtime, `<name>.int8-trt.onnx` for TensorRT."""
    suffix = "int8-trt" if target == "tensorrt" else "int8"
    model = onnx_dir / f"{name}.{suffix}.onnx"
    return model, model.with_name(model.name + ".json")


def excluded(node_names: Iterable[str], patterns: list[str]) -> list[str]:
    """Node names matching any pattern (`re.search`), in graph order."""
    compiled = [re.compile(p) for p in patterns]
    return [n for n in node_names if any(c.search(n) for c in compiled)]


def quantized_op_counts(model: Any) -> dict[str, int]:
    """Op types wrapped in QDQ (an input from a DequantizeLinear and the output into a
    QuantizeLinear), counted: the ops ONNX Runtime can run as integer kernels."""
    nodes = model.graph.node
    dq_outputs = {o for n in nodes if n.op_type == "DequantizeLinear" for o in n.output}
    q_inputs = {i for n in nodes if n.op_type == "QuantizeLinear" for i in n.input}
    counts: dict[str, int] = {}
    for n in nodes:
        if n.op_type in ("QuantizeLinear", "DequantizeLinear"):
            continue
        if any(i in dq_outputs for i in n.input) and any(o in q_inputs for o in n.output):
            counts[n.op_type] = counts.get(n.op_type, 0) + 1
    return dict(sorted(counts.items()))


class _Reader:
    """ONNX Runtime CalibrationDataReader over preprocessed images, one per call."""

    def __init__(
        self, input_name: str, images: Iterator[NDArray[np.uint8]], imgsz: int, stride: int
    ) -> None:
        self._input = input_name
        self._images = images
        self._imgsz = imgsz
        self._stride = stride
        self.count = 0

    def get_next(self) -> dict[str, NDArray[np.float32]] | None:
        image = next(self._images, None)
        if image is None:
            return None
        x, _ = preprocess([image], self._imgsz, self._stride)
        self.count += 1
        return {self._input: x}


def quantize_int8(
    source: Path,
    source_sha256: str,
    weights: str,
    weights_sha256: str,
    images: Iterator[NDArray[np.uint8]],
    lock_sha256: str,
    settings: Int8Settings,
    onnx_dir: Path,
    imgsz: int,
    target: Int8Target = "onnxruntime",
) -> Path:
    """Quantize `source` (the verified FP32 export) to `int8_paths(onnx_dir, weights, target)`."""
    import onnx

    q: Any = importlib.import_module("onnxruntime.quantization")
    quant_pre_process = importlib.import_module(
        "onnxruntime.quantization.shape_inference"
    ).quant_pre_process
    from vms.bench.results import package_versions

    model = onnx.load(str(source))
    meta = {p.key: p.value for p in model.metadata_props}
    stride = int(meta.get("stride", 32))
    input_name = model.graph.input[0].name
    types = {"quint8": q.QuantType.QUInt8, "qint8": q.QuantType.QInt8}
    methods = {
        "minmax": q.CalibrationMethod.MinMax,
        "percentile": q.CalibrationMethod.Percentile,
        "entropy": q.CalibrationMethod.Entropy,
    }
    out_path, meta_path = int8_paths(onnx_dir, weights, target)
    reader = _Reader(input_name, images, imgsz, stride)
    with tempfile.TemporaryDirectory() as tmp:
        # Shape inference and graph cleanup first, as ORT recommends; the symbolic pass is
        # skipped because the export's dynamic H and W are already named.
        pre = Path(tmp) / "pre.onnx"
        quant_pre_process(str(source), str(pre), skip_symbolic_shape=True)
        # Names after the pre-processing pass, which may rewrite the graph.
        skip = excluded((n.name for n in onnx.load(str(pre)).graph.node), settings.exclude_nodes)
        q.quantize_static(
            str(pre),
            str(out_path),
            reader,
            quant_format=q.QuantFormat.QDQ,
            per_channel=settings.per_channel,
            activation_type=types[settings.activation_type],
            weight_type=types[settings.weight_type],
            calibrate_method=methods[settings.method],
            nodes_to_exclude=skip,
            op_types_to_quantize=settings.op_types or None,
            extra_options=quantizer_options(settings),
        )
    if reader.count == 0:
        raise ValueError("no calibration images")
    written = onnx.load(str(out_path))
    sidecar = Int8Export(
        weights=weights,
        target=target,
        weights_sha256=weights_sha256,
        source_onnx_sha256=source_sha256,
        int8_sha256=sha256_file(out_path),
        calibration_lock_sha256=lock_sha256,
        calibration_images=reader.count,
        settings=settings,
        excluded_nodes=skip,
        quantized_ops=quantized_op_counts(written),
        packages=package_versions(("onnx", "onnxruntime")),
        created_at=datetime.now(UTC),
    )
    meta_path.write_text(sidecar.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return out_path


def quantizer_options(settings: Int8Settings) -> dict[str, Any]:
    """`extra_options` of ONNX Runtime's `quantize_static` for these settings."""
    options: dict[str, Any] = {}
    if settings.symmetric:
        options |= {"ActivationSymmetric": True, "WeightSymmetric": True}
    if not settings.quantize_bias:
        options["QuantizeBias"] = False
    return options


def verified_int8(
    onnx_dir: Path,
    name: str,
    weights_sha256: str,
    source_onnx_sha256: str,
    target: Int8Target = "onnxruntime",
) -> tuple[Path, Int8Export]:
    """The INT8 model for `name`, checked against its sidecar, the weights and the FP32 export."""
    model_path, meta_path = int8_paths(onnx_dir, name, target)
    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"{model_path} or its sidecar is missing; run `make quantize`")
    meta = Int8Export.model_validate_json(meta_path.read_text(encoding="utf-8"))
    if meta.weights_sha256 != weights_sha256 or meta.source_onnx_sha256 != source_onnx_sha256:
        raise StaleExportError(
            f"{model_path} was quantized from another export of {name}; run `make quantize`"
        )
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != meta.int8_sha256:
        raise StaleExportError(f"{model_path} does not match its sidecar hash; run `make quantize`")
    return model_path, meta


def quantize_from_config(names: list[str], target: Int8Target = "onnxruntime") -> list[Path]:
    """`make quantize` / `trt-prepare`: quantize each named model (default: every configured
    one) with configs/quantize.yaml's settings for `target`, from its verified FP32 export."""
    from vms.common.logs import get_logger
    from vms.common.settings import load_settings
    from vms.worker import calibration as cal
    from vms.worker.detector import DetectorSettings
    from vms.worker.onnx_detector import verified_export
    from vms.worker.weights import load_models_config

    log = get_logger(__name__)
    settings = load_settings(QuantizeSettings, "quantize")
    p = settings.calibration
    spec = cal.load_spec(p.spec)
    images = cal.read_lock(p.lock)
    missing = [i.id for i in images if not cal.image_ok(p.images_dir, i)]
    if missing:
        raise SystemExit(f"{len(missing)} calibration images missing; run `make calibration-data`")
    models = load_models_config()
    written = []
    for name in names or sorted(models.weights):
        sha = models.weights[name].sha256
        source, meta = verified_export(models.onnx_dir, name, sha)
        path = quantize_int8(
            source,
            meta.onnx_sha256,
            name,
            sha,
            cal.load_pixels(images, p.images_dir, spec),
            cal.lock_sha256(p.lock),
            settings.for_target(target),
            models.onnx_dir,
            DetectorSettings().imgsz,
            target,
        )
        log.info("int8 model written", extra={"weights": name, "target": target, "path": str(path)})
        written.append(path)
    return written
