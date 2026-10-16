"""Sequential change detection on the drift statistics (task 9.3).

The monitor produces, per camera and every update, a vector of drift statistics
(vms.drift.signals.compare). Each entry is a *channel*:

- `shift:<signal>` for the six standardized shifts (two-sided: darker or brighter);
- `confidence_ks`, `confidence_wasserstein`, `class_psi` (one-sided: they are >= 0 and only
  their increase means drift).

A channel's value is first standardized against clean data, z = (x - mu) / sigma, with mu and
sigma calibrated per channel on clean runs (`Calibration`). Then one of two detectors watches
each channel:

- threshold (`kind: threshold`): alarm when z > h (or |z| > h, two-sided). Memoryless:
  reacts to a large step at once and to a small persistent change never.
- CUSUM (`kind: cusum`, Page 1954): accumulates evidence, S = max(0, S + z - k) (and S- for
  the other side), alarm when S > h. `k` is the allowance: drift smaller than k clean
  standard deviations is ignored and S drains by k per update on clean data; `h` trades
  delay for false alarms. A
  persistent shift d > k reaches h after about h / (d - k) updates, so small drifts are caught
  eventually and large ones about as fast as a threshold.

Undefined statistics (NaN) leave the state unchanged and never alarm. A camera alarms when any
of its channels does (`CameraDetector`).

A channel alarms while S stays above h; S is capped at `cap * h` so an alarm clears soon after
the drift ends. The live monitor reports exactly that, and resets a camera's state when its
reference changes. Offline scoring (vms.drift.score) counts alarm events, so after an alarm
it resets the whole camera, as an operator acknowledging the alarm would.

Updates overlap (a 30 s window every 5 s), so consecutive z are strongly correlated and the
textbook average-run-length formulas for CUSUM do not hold; the false-alarm rate is measured
on clean runs instead (task 9.3).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from vms.drift.signals import SIGNALS, DriftStatistics

TWO_SIDED: Final = tuple(f"shift:{s}" for s in SIGNALS)
ONE_SIDED: Final = ("confidence_ks", "confidence_wasserstein", "class_psi")
CHANNELS: Final = (*TWO_SIDED, *ONE_SIDED)


def channel_values(stats: DriftStatistics) -> dict[str, float]:
    values = {f"shift:{s}": stats.shift.get(s, math.nan) for s in SIGNALS}
    values["confidence_ks"] = stats.confidence_ks
    values["confidence_wasserstein"] = stats.confidence_wasserstein
    values["class_psi"] = stats.class_psi
    return values


class ChannelCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mean: float
    std: float = Field(gt=0)


class Calibration(BaseModel):
    """Per-channel mean and std of the statistic on clean runs."""

    model_config = ConfigDict(extra="forbid")

    channels: dict[str, ChannelCalibration]

    def z(self, channel: str, value: float) -> float:
        c = self.channels.get(channel)
        if c is None or math.isnan(value):
            return math.nan
        return (value - c.mean) / c.std

    @classmethod
    def fit(
        cls, samples: Mapping[str, Iterable[float]], min_std: Mapping[str, float]
    ) -> Calibration:
        """mean and std of each channel's finite values, std floored at `min_std` (looping
        clips can make clean statistics almost constant)."""
        channels = {}
        for channel, values in samples.items():
            finite = [v for v in values if math.isfinite(v)]
            if len(finite) < 2:
                continue
            mean = sum(finite) / len(finite)
            var = sum((v - mean) ** 2 for v in finite) / (len(finite) - 1)
            floor = min_std.get(channel, 1e-6)
            channels[channel] = ChannelCalibration(mean=mean, std=max(math.sqrt(var), floor))
        return cls(channels=channels)


class DetectorSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["threshold", "cusum"] = "cusum"
    h: float = Field(default=10.0, gt=0)  # threshold on z, or CUSUM decision interval
    k: float = Field(default=1.0, ge=0)  # CUSUM allowance (ignored by threshold)
    cap: float = Field(default=2.0, ge=1)  # live: S is capped at cap * h
    channels: list[str] = Field(default_factory=lambda: list(CHANNELS))


@dataclass
class _Channel:
    two_sided: bool
    up: float = 0.0
    down: float = 0.0


@dataclass
class ChannelDetector:
    """One channel's state. `update(z)` returns whether it alarms at this update."""

    settings: DetectorSettings
    two_sided: bool
    state: _Channel = field(init=False)

    def __post_init__(self) -> None:
        self.state = _Channel(self.two_sided)

    @property
    def statistic(self) -> float:
        """CUSUM: the larger side's S; threshold: 0."""
        return max(self.state.up, self.state.down)

    def reset(self) -> None:
        self.state = _Channel(self.two_sided)

    def update(self, z: float) -> bool:
        if math.isnan(z):
            return False
        s = self.settings
        if s.kind == "threshold":
            return (abs(z) if self.two_sided else z) > s.h
        cap = s.cap * s.h
        self.state.up = min(max(0.0, self.state.up + z - s.k), cap)
        if self.two_sided:
            self.state.down = min(max(0.0, self.state.down - z - s.k), cap)
        return self.statistic > s.h


class CameraDetector:
    """All channels of one camera; alarms when any channel does."""

    def __init__(self, settings: DetectorSettings, calibration: Calibration) -> None:
        self.settings = settings
        self.calibration = calibration
        self.channels = {
            c: ChannelDetector(settings, c in TWO_SIDED)
            for c in settings.channels
            if c in calibration.channels
        }

    def update(self, values: Mapping[str, float]) -> list[str]:
        """The channels alarming at this update."""
        return [
            c
            for c, det in self.channels.items()
            if det.update(self.calibration.z(c, values.get(c, math.nan)))
        ]

    def reset(self) -> None:
        for det in self.channels.values():
            det.reset()
