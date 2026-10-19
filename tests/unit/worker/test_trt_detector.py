"""TensorRT engine bookkeeping (D38) without a GPU: build settings, engine paths, and the checks
that refuse an engine built from another model, by another TensorRT, or on another GPU. The
build and the forward pass need a GPU and run on a Runpod session (ladder-gpu-n)."""

from __future__ import annotations

import types
from datetime import UTC, datetime
from pathlib import Path

import pytest

from vms.common.settings import load_settings
from vms.worker import trt_detector as trt
from vms.worker.onnx_detector import StaleExportError, sha256_file


def test_committed_build_settings_cover_the_worker_and_the_sweep() -> None:
    s = load_settings(trt.TrtBuildSettings, "tensorrt", config_dir=Path("configs"))
    assert s.opt_shape == (4, 3, 384, 640)  # deployed batch, 1280x720 letterboxed
    assert s.max_shape[0] >= 32  # the batch sweep
    assert s.max_shape[2:] == (640, 640)  # the eval's square inputs
    assert all(
        lo <= op <= hi for lo, op, hi in zip(s.min_shape, s.opt_shape, s.max_shape, strict=True)
    )


def test_engine_paths_name_the_precision() -> None:
    engine, meta = trt.engine_paths(Path("models/trt"), "yolo11n", "int8")
    assert engine == Path("models/trt/yolo11n.int8.engine")
    assert meta == Path("models/trt/yolo11n.int8.engine.json")


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path, meta_path = trt.engine_paths(tmp_path, "yolo11n", "fp16")
    path.write_bytes(b"plan")
    meta = trt.EngineMeta(
        weights="yolo11n",
        weights_sha256="w" * 64,
        precision="fp16",
        source_onnx="yolo11n.fp16.onnx",
        source_onnx_sha256="s" * 64,
        engine_sha256=sha256_file(path),
        tensorrt="11.3.0.99",
        gpu="NVIDIA A40",
        input_name="images",
        output_name="output0",
        names={0: "person"},
        stride=32,
        profile={"min": [1, 3, 64, 64], "opt": [4, 3, 384, 640], "max": [32, 3, 640, 640]},
        build_s=60.0,
        created_at=datetime(2026, 10, 19, tzinfo=UTC),
    )
    meta_path.write_text(meta.model_dump_json())
    monkeypatch.setattr(trt, "_trt", lambda: types.SimpleNamespace(__version__="11.3.0.99"))
    monkeypatch.setattr(trt, "gpu_name", lambda: "NVIDIA A40")
    monkeypatch.setattr(trt, "source_model", lambda *a: (Path("x.onnx"), "s" * 64))
    return path


def test_a_matching_engine_verifies(engine: Path) -> None:
    path, meta = trt.verified_engine(engine.parent, Path("onnx"), "yolo11n", "w" * 64, "fp16")
    assert path == engine
    assert meta.names == {0: "person"}


@pytest.mark.parametrize(
    ("attr", "value", "reason"),
    [
        ("_trt", lambda: types.SimpleNamespace(__version__="11.4.0.106"), "TensorRT version"),
        ("gpu_name", lambda: "NVIDIA L4", "GPU"),
        ("source_model", lambda *a: (Path("x.onnx"), "t" * 64), "source model"),
    ],
)
def test_an_engine_from_another_model_trt_or_gpu_is_refused(
    engine: Path, monkeypatch: pytest.MonkeyPatch, attr: str, value: object, reason: str
) -> None:
    monkeypatch.setattr(trt, attr, value)
    with pytest.raises(StaleExportError, match=reason):
        trt.verified_engine(engine.parent, Path("onnx"), "yolo11n", "w" * 64, "fp16")


def test_a_tampered_or_missing_engine_is_refused(engine: Path) -> None:
    engine.write_bytes(b"other")
    with pytest.raises(StaleExportError, match="engine file"):
        trt.verified_engine(engine.parent, Path("onnx"), "yolo11n", "w" * 64, "fp16")
    with pytest.raises(FileNotFoundError, match="trt-prepare"):
        trt.verified_engine(engine.parent, Path("onnx"), "yolo11n", "w" * 64, "int8")
    with pytest.raises(ValueError, match="fp16 or int8"):
        trt.verified_engine(engine.parent, Path("onnx"), "yolo11n", "w" * 64, "fp32")
