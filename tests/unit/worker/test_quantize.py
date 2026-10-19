"""INT8 static quantization (task 5.2): node exclusion, the sidecar and its checks, and the INT8
model running behind the same detector interface. No network: a randomly initialized YOLO11n
(conftest `random_yolo`) is quantized on noise images. Accuracy with the real weights is the
detection eval (make eval SUITE=detection PRECISION=int8)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import onnx
import pytest

from vms.worker.detector import DetectorSettings, build_detector
from vms.worker.onnx_detector import OnnxDetector, StaleExportError, sha256_file
from vms.worker.quantize import (
    Int8Settings,
    excluded,
    int8_paths,
    quantize_int8,
    verified_int8,
)

HEAD = r"^/model\.23/(?!cv2\.|cv3\.)"


def noise_images(n: int, shape: tuple[int, int] = (360, 640)) -> list[np.ndarray]:
    rng = np.random.default_rng(3)
    return [rng.integers(0, 256, (*shape, 3), dtype=np.uint8) for _ in range(n)]


def test_exclusion_keeps_head_convs_and_drops_the_decode() -> None:
    names = [
        "/model.22/cv2/conv/Conv",
        "/model.23/cv2.0/cv2.0.0/conv/Conv",
        "/model.23/cv3.2/cv3.2.2/Conv",
        "/model.23/dfl/conv/Conv",
        "/model.23/Concat_26",
        "/model.23/Sigmoid",
    ]
    assert excluded(names, [HEAD]) == [
        "/model.23/dfl/conv/Conv",
        "/model.23/Concat_26",
        "/model.23/Sigmoid",
    ]
    assert excluded(names, []) == []


def test_int8_needs_the_onnxruntime_backend() -> None:
    with pytest.raises(ValueError, match="needs backend 'onnxruntime'"):
        DetectorSettings(precision="int8")
    assert DetectorSettings(backend="onnxruntime", precision="int8").precision == "int8"


@pytest.fixture(scope="module")
def int8_model(
    random_yolo: tuple[Path, Path, str], tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, str, str]:
    _, model, sha = random_yolo
    out_dir = tmp_path_factory.mktemp("int8")
    source_sha = sha256_file(model)
    path = quantize_int8(
        model,
        source_sha,
        "rand",
        sha,
        iter(noise_images(4)),
        "c" * 64,
        Int8Settings(exclude_nodes=[HEAD]),
        out_dir,
        640,
    )
    return path, sha, source_sha


def test_quantized_model_has_a_sidecar_that_verifies(int8_model: tuple[Path, str, str]) -> None:
    path, sha, source_sha = int8_model
    got, meta = verified_int8(path.parent, "rand", sha, source_sha)
    assert got == path
    assert meta.calibration_images == 4
    assert meta.calibration_lock_sha256 == "c" * 64
    assert meta.quantized_ops["Conv"] > 50  # the backbone and neck run in INT8
    assert meta.excluded_nodes
    assert all(n.startswith("/model.23/") for n in meta.excluded_nodes)
    assert not any("/cv2." in n or "/cv3." in n for n in meta.excluded_nodes)


def test_excluded_nodes_are_not_requantized(int8_model: tuple[Path, str, str]) -> None:
    """An excluded node may read a dequantized (float) input, but its output stays float."""
    path, sha, source_sha = int8_model
    _, meta = verified_int8(path.parent, "rand", sha, source_sha)
    graph = onnx.load(str(path)).graph
    q_in = {i for n in graph.node if n.op_type == "QuantizeLinear" for i in n.input}
    skip = set(meta.excluded_nodes)
    for node in graph.node:
        if node.name in skip:
            assert not q_in.intersection(node.output), node.name
    final = [n for n in graph.node if graph.output[0].name in n.output]
    assert final
    assert final[0].op_type == "Concat"  # output is not re-quantized


def test_stale_or_tampered_int8_models_are_refused(
    int8_model: tuple[Path, str, str], tmp_path: Path
) -> None:
    path, sha, source_sha = int8_model
    with pytest.raises(StaleExportError, match="another export"):
        verified_int8(path.parent, "rand", sha, "0" * 64)
    with pytest.raises(StaleExportError, match="another export"):
        verified_int8(path.parent, "rand", "0" * 64, source_sha)
    copy, sidecar = int8_paths(tmp_path, "rand")
    copy.write_bytes(path.read_bytes() + b"\0")
    sidecar.write_bytes(int8_paths(path.parent, "rand")[1].read_bytes())
    with pytest.raises(StaleExportError, match="sidecar hash"):
        verified_int8(tmp_path, "rand", sha, source_sha)
    with pytest.raises(FileNotFoundError, match="make quantize"):
        verified_int8(tmp_path, "missing", sha, source_sha)


def test_int8_model_runs_behind_the_detector_interface(
    int8_model: tuple[Path, str, str], random_yolo: tuple[Path, Path, str]
) -> None:
    path, sha, _ = int8_model
    _, fp32_path, _ = random_yolo
    settings = DetectorSettings(weights="rand", backend="onnxruntime", precision="int8", conf=0.0)
    int8 = OnnxDetector(path, sha, settings, variant="int8@abcd")
    fp32 = OnnxDetector(fp32_path, sha, settings.model_copy(update={"precision": "fp32"}))
    assert int8.model_version == f"rand@{sha[:12]}+onnx-int8@abcd"
    x = np.zeros((1, 3, 384, 640), dtype=np.float32)
    assert int8.forward(x).shape == fp32.forward(x).shape
    assert len(int8.detect_batch(noise_images(2))) == 2


def test_build_detector_needs_a_quantized_model_for_int8(
    random_yolo: tuple[Path, Path, str], tmp_path: Path
) -> None:
    weights, model, sha = random_yolo
    (tmp_path / "configs").mkdir()
    (tmp_path / "models/weights").mkdir(parents=True)
    (tmp_path / "models/weights/rand.pt").write_bytes(weights.read_bytes())
    onnx_dir = tmp_path / "models/onnx"
    onnx_dir.mkdir()
    for f in (model, model.with_name(model.name + ".json")):
        (onnx_dir / f.name).write_bytes(f.read_bytes())
    (tmp_path / "configs/models.yaml").write_text(
        "weights_dir: models/weights\nonnx_dir: models/onnx\nweights:\n  rand:\n"
        f"    url: https://example.invalid/rand.pt\n    bytes: {weights.stat().st_size}\n"
        f"    sha256: {hashlib.sha256(weights.read_bytes()).hexdigest()}\n    license: test\n",
        encoding="utf-8",
    )
    settings = DetectorSettings(weights="rand", backend="onnxruntime", precision="int8")
    with pytest.raises(FileNotFoundError, match="make quantize"):
        build_detector(settings, tmp_path)
    quantize_int8(
        model,
        sha256_file(model),
        "rand",
        sha,
        iter(noise_images(2)),
        "c" * 64,
        Int8Settings(exclude_nodes=[HEAD]),
        onnx_dir,
        640,
    )
    detector = build_detector(settings, tmp_path)
    assert "+onnx-int8@" in detector.model_version


def test_backend_and_precision_combinations() -> None:
    assert DetectorSettings(backend="tensorrt", precision="int8", device="cuda").precision == "int8"
    assert (
        DetectorSettings(backend="tensorrt", precision="fp16", device="cuda").backend == "tensorrt"
    )
    with pytest.raises(ValueError, match="'fp16' needs backend 'tensorrt'"):
        DetectorSettings(backend="onnxruntime", precision="fp16")
    with pytest.raises(ValueError, match="'tensorrt' needs device 'cuda'"):
        DetectorSettings(backend="tensorrt", precision="fp16")


def test_tensorrt_target_is_symmetric_with_the_cpu_exclusions() -> None:
    from vms.common.settings import load_settings
    from vms.worker.quantize import QuantizeSettings, quantizer_options

    s = load_settings(QuantizeSettings, "quantize", config_dir=Path("configs"))
    trt, cpu = s.for_target("tensorrt"), s.for_target("onnxruntime")
    assert (trt.activation_type, trt.weight_type, trt.symmetric) == ("qint8", "qint8", True)
    assert trt.exclude_nodes == cpu.exclude_nodes  # same head decode in FP32
    assert (trt.method, trt.per_channel) == (cpu.method, cpu.per_channel)
    assert quantizer_options(trt) == {
        "ActivationSymmetric": True,
        "WeightSymmetric": True,
        "QuantizeBias": False,
    }
    assert trt.op_types == ["Conv"]
    assert quantizer_options(cpu) == {}
    assert int8_paths(Path("m"), "yolo11n", "tensorrt")[0] == Path("m/yolo11n.int8-trt.onnx")
    assert int8_paths(Path("m"), "yolo11n")[0] == Path("m/yolo11n.int8.onnx")


def test_tensorrt_int8_model_has_only_zero_points_of_zero(
    random_yolo: tuple[Path, Path, str], tmp_path: Path
) -> None:
    from onnx import numpy_helper

    _, model, sha = random_yolo
    source_sha = sha256_file(model)
    path = quantize_int8(
        model,
        source_sha,
        "rand",
        sha,
        iter(noise_images(2)),
        "c" * 64,
        Int8Settings(
            activation_type="qint8",
            symmetric=True,
            op_types=["Conv"],
            quantize_bias=False,
            exclude_nodes=[HEAD],
        ),
        tmp_path,
        640,
        "tensorrt",
    )
    assert path.name == "rand.int8-trt.onnx"
    m = onnx.load(str(path))
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    zps = [
        inits[n.input[2]]
        for n in m.graph.node
        if n.op_type == "QuantizeLinear" and len(n.input) > 2 and n.input[2] in inits
    ]
    assert zps
    assert all(z.dtype == np.int8 for z in zps)
    assert all(not z.any() for z in zps)
    # What TensorRT rejects (E-062): Q/DQ on scalars, and int32 bias Q/DQ.
    for n in m.graph.node:
        if n.op_type == "DequantizeLinear" and n.input[0] in inits:
            x = inits[n.input[0]]
            assert x.ndim > 0, n.name
            assert x.dtype == np.int8, n.name
    _, meta = verified_int8(tmp_path, "rand", sha, source_sha, "tensorrt")
    assert meta.target == "tensorrt"
    with pytest.raises(FileNotFoundError):
        verified_int8(tmp_path, "rand", sha, source_sha)  # the ONNX Runtime model is another file
