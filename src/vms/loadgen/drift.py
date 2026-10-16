"""Drift injection: scheduled, seeded changes to what some cameras send (task 9.2).

A scenario lists drift events under `drift.events`. Each applies to cameras by index and
starts at a known onset, `onset_s` after the run start plus a seeded jitter in
[0, onset_jitter_s) drawn per camera and repetition, so repetitions differ while every onset
is recorded in the results. `ramp_s` 0 is an abrupt change; otherwise the intensity rises
linearly from 0 at the onset to 1 after `ramp_s` and stays there (a gradual drift). A drift
lasts until the end of the run.

Kinds: the pixel transforms of vms.drift.transforms (darken, blur, noise, shift, occlusion)
with their `strength`, and `switch_clip`, which makes the camera play `clip` from the onset
on (a class-mix change: the same camera now sees another scene). Frames before the onset are
sent exactly as without drift; transformed frames are decoded, transformed and re-encoded at
`frame.jpeg_quality` in a thread pool (see vms.loadgen.runner).

Each event is also posted to Grafana as an annotation when the repetition starts
(`ScheduledDrift.annotation`, task 9.4), so the Drift dashboard marks the onset that an
alarm should answer.

Noise is drawn from a generator seeded per camera and frame, and the occluder's position per
camera, so a repetition's pixels are reproducible from its seed.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vms.datasets.clips import decode_jpeg, encode_jpeg
from vms.drift.transforms import TransformKind, apply, check_strength
from vms.loadgen.schedule import CameraPlan
from vms.observe.annotations import Annotation

# Grafana tag of onset annotations; the Drift dashboard queries it (vms.observe.dashboards).
ONSET_TAG = "drift-onset"

DriftKind = Literal["darken", "blur", "noise", "shift", "occlusion", "switch_clip"]


class DriftEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: DriftKind
    cameras: list[int] = Field(min_length=1)  # 0-based camera indices
    onset_s: float = Field(ge=0)
    onset_jitter_s: float = Field(default=0.0, ge=0)
    ramp_s: float = Field(default=0.0, ge=0)  # 0: abrupt
    strength: float | None = None  # pixel transforms (vms.drift.transforms)
    clip: str | None = None  # switch_clip

    @model_validator(mode="after")
    def _check(self) -> DriftEvent:
        if self.kind == "switch_clip":
            if self.clip is None or self.strength is not None:
                raise ValueError("switch_clip takes a clip and no strength")
            if self.ramp_s:
                raise ValueError("switch_clip is abrupt (ramp_s 0)")
        else:
            if self.strength is None or self.clip is not None:
                raise ValueError(f"{self.kind} takes a strength and no clip")
            check_strength(self.kind, self.strength)
        if len(set(self.cameras)) != len(self.cameras) or min(self.cameras) < 0:
            raise ValueError("cameras must be distinct, non-negative indices")
        return self


class DriftInjectionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    events: list[DriftEvent] = Field(default_factory=list)
    render_workers: int = Field(default=2, ge=1)  # threads that transform and re-encode

    def clips(self) -> list[str]:
        return [e.clip for e in self.events if e.clip is not None]


@dataclass(frozen=True)
class ScheduledDrift:
    """One event on one camera, with its onset and seeds resolved."""

    camera_id: str
    kind: DriftKind
    onset_s: float
    ramp_s: float
    strength: float | None
    clip: str | None
    anchor: tuple[float, float]  # occluder centre, normalized
    noise_seed: int

    def intensity(self, t: float) -> float:
        if t < self.onset_s:
            return 0.0
        if self.ramp_s == 0:
            return 1.0
        return min(1.0, (t - self.onset_s) / self.ramp_s)

    def record(self, wall0: datetime) -> dict[str, Any]:
        """For the results JSON: what changed on which camera, and exactly when."""
        return {
            "camera_id": self.camera_id,
            "kind": self.kind,
            "onset_s": round(self.onset_s, 3),
            "onset_at": (wall0 + timedelta(seconds=self.onset_s)).isoformat(),
            "ramp_s": self.ramp_s,
            "strength": self.strength,
            "clip": self.clip,
        }

    def annotation(self, wall0: datetime, run_id: str) -> Annotation:
        """The onset marker for the dashboards: a point if abrupt, the ramp as a region."""
        onset = wall0 + timedelta(seconds=self.onset_s)
        what = f"clip {self.clip}" if self.kind == "switch_clip" else f"strength {self.strength:g}"
        how = f"ramp {self.ramp_s:g} s" if self.ramp_s else "abrupt"
        return Annotation(
            time=onset,
            time_end=onset + timedelta(seconds=self.ramp_s) if self.ramp_s else None,
            text=f"Injected {self.kind} on {self.camera_id} ({what}, {how}); run {run_id}",
            tags=(ONSET_TAG, self.kind, self.camera_id),
        )


def schedule_drift(
    settings: DriftInjectionSettings, plans: Sequence[CameraPlan], seed: int
) -> dict[str, list[ScheduledDrift]]:
    """Each camera's events in scenario order; cameras without drift are absent."""
    rng = random.Random(f"drift:{seed}")
    out: dict[str, list[ScheduledDrift]] = {}
    for event in settings.events:
        for index in event.cameras:
            if index >= len(plans):
                raise ValueError(f"drift event on camera {index}, but the run has {len(plans)}")
            camera_id = plans[index].camera_id
            out.setdefault(camera_id, []).append(
                ScheduledDrift(
                    camera_id=camera_id,
                    kind=event.kind,
                    onset_s=event.onset_s + rng.uniform(0.0, event.onset_jitter_s),
                    ramp_s=event.ramp_s,
                    strength=event.strength,
                    clip=event.clip,
                    anchor=(rng.uniform(0.2, 0.8), rng.uniform(0.3, 0.8)),
                    noise_seed=rng.getrandbits(63),
                )
            )
    return out


def active_clip(events: Sequence[ScheduledDrift], t: float) -> str | None:
    """The clip a camera plays at `t` if a switch_clip event has started (the latest wins)."""
    clip = None
    for e in events:
        if e.kind == "switch_clip" and t >= e.onset_s:
            clip = e.clip
    return clip


def active_transforms(
    events: Sequence[ScheduledDrift], t: float
) -> list[tuple[ScheduledDrift, float]]:
    return [(e, i) for e in events if e.kind != "switch_clip" and (i := e.intensity(t)) > 0]


def render(
    jpeg: bytes, transforms: Sequence[tuple[ScheduledDrift, float]], k: int, quality: int
) -> bytes:
    """Frame k of a camera with its active transforms applied, in event order."""
    image = decode_jpeg(jpeg)
    for event, intensity in transforms:
        kind: TransformKind = event.kind  # type: ignore[assignment]  # not switch_clip here
        assert event.strength is not None
        image = apply(
            kind,
            image,
            intensity,
            event.strength,
            rng=np.random.default_rng([event.noise_seed, k]),
            anchor=event.anchor,
        )
    return encode_jpeg(image, quality)
