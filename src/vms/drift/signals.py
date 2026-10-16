"""Per-camera drift signals over a time range, and their comparison with a reference.

A `Sample` is one camera's frames and detections over one capture-time range, reduced to the
arrays the statistics need. The reference is a `Sample` too (the calibration period), and
`compare` turns a window and a reference into `DriftStatistics`:

Input signals (one value per frame, from the worker's input statistics, vms.worker.stats):

- brightness: mean luma. Darkening (dusk, a failing light, an underexposed camera).
- contrast: luma standard deviation. Fog, haze, glare, a washed-out exposure.
- log_blur: ln(1 + variance of the Laplacian). Defocus, a dirty or wet lens; logged because
  the raw variance is heavy-tailed and scene-dependent.
- noise: estimated noise sigma. Sensor gain at night, interference.

Output signals (from stored detections with confidence >= `min_conf`, so the tracker's
low-confidence boxes kept only as part of a track do not mix in):

- detections_per_frame: one value per frame. Occlusion, an emptier or busier scene.
- confidence: one value per detection. A degraded input lowers it before boxes disappear.
  Compared with a standardized shift of its mean and, as a distribution, with the KS
  statistic and the Wasserstein-1 distance.
- class mix: detections per class over fixed bins (`classes` plus "other"), compared with
  PSI. A camera now pointed at a road instead of a sidewalk, or a different clip.

For each per-frame or per-detection signal `compare` reports the window mean and its
standardized shift; detection-based statistics need at least `min_detections` in both the
window and the reference, and everything needs `min_frames`; otherwise the value is NaN.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Final

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vms.drift.stats import FloatArray, ks_statistic, psi, standardized_shift, wasserstein_1

INPUT_SIGNALS: Final = ("brightness", "contrast", "log_blur", "noise")
OUTPUT_SIGNALS: Final = ("detections_per_frame", "confidence")
SIGNALS: Final = (*INPUT_SIGNALS, *OUTPUT_SIGNALS)
OTHER_CLASS: Final = "other"


class SignalSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Stored detections below this are left out (the tracker keeps some down to its low
    # threshold); matches tracker.store_min_conf in configs/worker.yaml.
    min_conf: float = Field(default=0.25, ge=0, le=1)
    # Class-mix bins; any other class falls in "other".
    classes: list[str] = Field(
        default_factory=lambda: ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
    )
    min_frames: int = Field(default=20, ge=1)
    min_detections: int = Field(default=20, ge=1)
    psi_eps: float = Field(default=1e-4, gt=0)
    # Floor of the reference std per signal, in the signal's units, so a signal that was
    # constant during calibration (an empty scene: 0 detections on every frame) gives a
    # finite shift.
    min_std: dict[str, float] = Field(
        default_factory=lambda: {
            "brightness": 1.0,
            "contrast": 1.0,
            "log_blur": 0.05,
            "noise": 0.1,
            "detections_per_frame": 0.5,
            "confidence": 0.01,
        }
    )

    @model_validator(mode="after")
    def _check(self) -> SignalSettings:
        missing = set(SIGNALS) - set(self.min_std)
        if missing:
            raise ValueError(f"min_std has no floor for {sorted(missing)}")
        if OTHER_CLASS in self.classes or len(set(self.classes)) != len(self.classes):
            raise ValueError(f"classes must be distinct and not include {OTHER_CLASS!r}")
        return self

    @property
    def bins(self) -> tuple[str, ...]:
        return (*self.classes, OTHER_CLASS)


@dataclass(frozen=True)
class FrameRow:
    camera_id: str
    frame_id: str
    ts: datetime
    brightness: float | None
    contrast: float | None
    blur_var: float | None
    noise: float | None  # NULL on frames processed before migration 0007


@dataclass(frozen=True)
class DetectionRow:
    camera_id: str
    frame_id: str
    class_name: str
    confidence: float


@dataclass(frozen=True)
class Sample:
    """One camera's signals over one time range. `values` holds per-frame arrays for the
    input signals and detections_per_frame (NaN where a frame lacks the statistic) and the
    per-detection confidences; `class_counts` is aligned with `SignalSettings.bins`."""

    frames: int
    values: Mapping[str, FloatArray]
    class_counts: FloatArray

    @property
    def detections(self) -> int:
        return int(self.values["confidence"].size)

    def mean(self, signal: str) -> float:
        v = self.values[signal]
        v = v[~np.isnan(v)]
        return float(v.mean()) if v.size else math.nan

    def std(self, signal: str) -> float:
        v = self.values[signal]
        v = v[~np.isnan(v)]
        return float(v.std()) if v.size else math.nan

    @classmethod
    def build(
        cls,
        frames: Sequence[FrameRow],
        detections: Iterable[DetectionRow],
        settings: SignalSettings,
    ) -> Sample:
        """Detections of frames not in `frames`, or below `min_conf`, are ignored."""
        index = {f.frame_id: i for i, f in enumerate(frames)}
        counts = np.zeros(len(frames))
        confidences: list[float] = []
        bins = {name: i for i, name in enumerate(settings.bins)}
        other = bins[OTHER_CLASS]
        class_counts = np.zeros(len(bins))
        for d in detections:
            i = index.get(d.frame_id)
            if i is None or d.confidence < settings.min_conf:
                continue
            counts[i] += 1
            confidences.append(d.confidence)
            class_counts[bins.get(d.class_name, other)] += 1

        def column(values: Iterable[float | None]) -> FloatArray:
            return np.array([math.nan if v is None else v for v in values], dtype=np.float64)

        blur = column(f.blur_var for f in frames)
        return cls(
            frames=len(frames),
            values={
                "brightness": column(f.brightness for f in frames),
                "contrast": column(f.contrast for f in frames),
                "log_blur": np.log1p(np.maximum(blur, 0.0)),
                "noise": column(f.noise for f in frames),
                "detections_per_frame": counts,
                "confidence": np.array(confidences, dtype=np.float64),
            },
            class_counts=class_counts,
        )


@dataclass(frozen=True)
class DriftStatistics:
    window_frames: int
    window_detections: int
    window_mean: dict[str, float] = field(default_factory=dict)
    shift: dict[str, float] = field(default_factory=dict)
    confidence_ks: float = math.nan
    confidence_wasserstein: float = math.nan
    class_psi: float = math.nan


def compare(window: Sample, reference: Sample, settings: SignalSettings) -> DriftStatistics:
    nan = dict.fromkeys(SIGNALS, math.nan)
    if window.frames < settings.min_frames or reference.frames < settings.min_frames:
        return DriftStatistics(
            window.frames,
            window.detections,
            {s: window.mean(s) for s in SIGNALS},
            nan,
        )
    enough = min(window.detections, reference.detections) >= settings.min_detections
    means = {s: window.mean(s) for s in SIGNALS}
    shift = {
        s: standardized_shift(means[s], reference.mean(s), reference.std(s), settings.min_std[s])
        for s in SIGNALS
    }
    if not enough:
        shift["confidence"] = math.nan
    conf_w, conf_r = window.values["confidence"], reference.values["confidence"]
    return DriftStatistics(
        window_frames=window.frames,
        window_detections=window.detections,
        window_mean=means,
        shift=shift,
        confidence_ks=ks_statistic(conf_w, conf_r) if enough else math.nan,
        confidence_wasserstein=wasserstein_1(conf_w, conf_r) if enough else math.nan,
        class_psi=(
            psi(window.class_counts, reference.class_counts, settings.psi_eps)
            if enough
            else math.nan
        ),
    )
