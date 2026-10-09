"""Per-camera schedules: which clip a camera plays, from where, and when each frame is due.

Pure functions of the settings and the seed, so a run's offered load is fully reproducible.

- Each camera gets a seeded start frame, so cameras playing the same clip show different
  moments instead of identical frames.
- Each camera gets a seeded phase in [0, 1/fps), so N cameras spread their sends across the
  frame interval instead of firing in one burst at every tick (which would measure burst
  handling, not steady load).
- Playback is real time: at `fps` sends per second, a camera steps through a clip recorded at
  `clip_fps` by clip_fps/fps frames per send, looping at the end.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

# Absorbs float error so k * ratio landing exactly on an integer is not floored to the one below.
_EPS = 1e-9


@dataclass(frozen=True)
class CameraPlan:
    camera_id: str
    clip_id: str
    start_frame: int
    phase_s: float


def camera_id(prefix: str, index: int, total: int) -> str:
    """`cam-001` style ids, zero-padded to at least 3 digits so they sort."""
    width = max(3, len(str(total)))
    return f"{prefix}{index + 1:0{width}d}"


def plan_cameras(
    cameras: int,
    clip_ids: Sequence[str],
    clip_lengths: Mapping[str, int],
    fps: float,
    seed: int,
    *,
    assignment: Literal["round_robin", "random"] = "round_robin",
    prefix: str = "cam-",
) -> list[CameraPlan]:
    if not clip_ids:
        raise ValueError("need at least one clip")
    rng = random.Random(seed)
    plans = []
    for i in range(cameras):
        clip = clip_ids[i % len(clip_ids)] if assignment == "round_robin" else rng.choice(clip_ids)
        plans.append(
            CameraPlan(
                camera_id=camera_id(prefix, i, cameras),
                clip_id=clip,
                start_frame=rng.randrange(clip_lengths[clip]),
                phase_s=rng.uniform(0.0, 1.0 / fps),
            )
        )
    return plans


def due_time(plan: CameraPlan, k: int, fps: float) -> float:
    """Seconds after the run start at which frame k of this camera is due."""
    return plan.phase_s + k / fps


def clip_frame(plan: CameraPlan, k: int, fps: float, clip_fps: float, clip_len: int) -> int:
    """Index into the clip of the k-th frame this camera sends."""
    return (plan.start_frame + math.floor(k * clip_fps / fps + _EPS)) % clip_len
