"""Settings for the detection eval, loaded from `configs/eval-detection.yaml`."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vms.common.settings import ServiceSettings
from vms.drift.transforms import check_strength
from vms.worker.detector import DetectorSettings


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SubsetPaths(_Section):
    spec: Path = Path("configs/eval/detection.v1.yaml")
    subset: Path = Path("eval/detection/subset.v1.jsonl")
    manifest: Path = Path("eval/detection/subset.v1.json")
    images_dir: Path = Path("data/eval/detection/v1")
    fetch_workers: int = Field(default=8, ge=1)


class WorkerFrame(_Section):
    """How MOT17 frames are fed to the detector: as loadgen sends them to the worker
    (configs/loadgen.yaml, frame), so the mot17 numbers describe the camera path."""

    width: int = Field(default=1280, ge=16)
    height: int = Field(default=720, ge=16)
    jpeg_quality: int = Field(default=85, ge=1, le=100)


class BootstrapSettings(_Section):
    # Its own seed, not the shared VMS_SEED that .env sets for every service: changing a
    # global default must not move a reported interval.
    seed: int = 20261014
    resamples: int = Field(default=2000, ge=1)
    confidence: float = Field(default=0.95, gt=0, lt=1)


class EvalDrift(_Section):
    """A photometric drift applied to every image before detection (task 9.2), at full
    intensity, with the same transform loadgen injects (vms.drift.transforms). Boxes stay
    valid, so the ground truth is unchanged. Noise is seeded per image from `seed` and the
    image id."""

    kind: Literal["darken", "blur", "noise"]
    strength: float
    seed: int = 20261016

    @model_validator(mode="after")
    def _check(self) -> EvalDrift:
        check_strength(self.kind, self.strength)
        return self


class DetectionEvalSettings(ServiceSettings):
    service: str = "eval-detection"
    paths: SubsetPaths = Field(default_factory=SubsetPaths)
    results_dir: Path = Path("eval/results")
    predictions_dir: Path = Path("data/eval/detection/predictions")
    sources: list[Literal["coco", "mot17"]] = Field(default=["coco", "mot17"], min_length=1)
    detector: DetectorSettings = Field(
        default_factory=lambda: DetectorSettings(conf=0.001, max_det=300)
    )
    max_dets_per_class: int = Field(default=100, ge=1)  # per image, as in the COCO protocol
    mot_frame: WorkerFrame = Field(default_factory=WorkerFrame)
    bootstrap: BootstrapSettings = Field(default_factory=BootstrapSettings)
    drift: EvalDrift | None = None  # none: the clean subset
    max_images: int | None = Field(
        default=None, ge=1
    )  # smoke runs only; never for reported numbers
