"""Pretrained weights: pinned in `configs/models.yaml`, fetched and verified by checksum.

Weights are never committed (AGENTS.md section 7). `ensure_weights` downloads a file once into
`weights_dir`, verifies size and sha256, and on later calls only re-verifies the cached copy.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from vms.datasets.http import RetryPolicy, Transport, UrllibTransport, download_file

DEFAULT_MODELS_CONFIG = Path("configs/models.yaml")


class WeightSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    url: HttpUrl
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    license: str


class ModelsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    weights_dir: Path
    onnx_dir: Path = Path("models/onnx")  # exports made by `make onnx` (vms.worker.onnx_detector)
    trt_dir: Path = Path("models/trt")  # TensorRT engines built on the GPU they run on (D38)
    weights: dict[str, WeightSpec]


def load_models_config(path: Path = DEFAULT_MODELS_CONFIG) -> ModelsConfig:
    with path.open(encoding="utf-8") as fh:
        return ModelsConfig.model_validate(yaml.safe_load(fh))


def ensure_weights(
    config: ModelsConfig,
    name: str,
    *,
    root: Path = Path(),
    transport: Transport | None = None,
    retry: RetryPolicy | None = None,
) -> Path:
    """Path to verified weights for `name`, downloading them first if needed."""
    try:
        spec = config.weights[name]
    except KeyError:
        known = ", ".join(sorted(config.weights))
        raise KeyError(f"unknown weights {name!r}; configured: {known}") from None
    dest = root / config.weights_dir / f"{name}.pt"
    return download_file(
        transport or UrllibTransport(),
        retry or RetryPolicy(),
        str(spec.url),
        dest,
        f"sha256:{spec.sha256}",
        spec.bytes,
    )
