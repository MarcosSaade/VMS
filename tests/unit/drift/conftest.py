"""An in-memory DriftStore and synthetic frame streams with known shifts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta

import numpy as np

from vms.drift.signals import DetectionRow, FrameRow
from vms.drift.store import ReferenceRange

T0 = datetime(2026, 10, 16, 9, 0, tzinfo=UTC)
FPS = 5


@dataclass(frozen=True)
class Scene:
    """Per-frame distributions of one synthetic camera."""

    brightness: float = 120.0
    contrast: float = 50.0
    blur_var: float = 400.0
    noise: float = 2.0
    detections: int = 4  # per frame
    confidence: float = 0.7
    classes: tuple[str, ...] = ("person", "car")


def stream(
    camera_id: str,
    start: datetime,
    seconds: float,
    scene: Scene,
    rng: np.random.Generator,
    *,
    fps: int = FPS,
) -> tuple[list[FrameRow], list[DetectionRow]]:
    frames: list[FrameRow] = []
    detections: list[DetectionRow] = []
    for k in range(int(seconds * fps)):
        frame_id = f"{camera_id}-{start.timestamp():.0f}-{k}"
        frames.append(
            FrameRow(
                camera_id,
                frame_id,
                start + timedelta(seconds=k / fps),
                float(rng.normal(scene.brightness, 2.0)),
                float(rng.normal(scene.contrast, 1.5)),
                float(rng.lognormal(np.log(scene.blur_var), 0.1)),
                float(abs(rng.normal(scene.noise, 0.2))),
            )
        )
        for j in range(scene.detections):
            conf = float(np.clip(rng.normal(scene.confidence, 0.08), 0.26, 0.99))
            detections.append(
                DetectionRow(camera_id, frame_id, scene.classes[j % len(scene.classes)], conf)
            )
    return frames, detections


@dataclass
class FakeDriftStore:
    frame_rows: list[FrameRow] = field(default_factory=list)
    detection_rows: list[DetectionRow] = field(default_factory=list)
    refs: dict[str, ReferenceRange] = field(default_factory=dict)
    fail: bool = False
    reads: int = 0

    def add(self, rows: tuple[list[FrameRow], list[DetectionRow]]) -> None:
        self.frame_rows.extend(rows[0])
        self.detection_rows.extend(rows[1])

    def _check(self) -> None:
        from vms.storage.frames import FrameStoreError

        if self.fail:
            raise FrameStoreError("down")

    async def cameras(self) -> list[str]:
        self._check()
        return sorted({f.camera_id for f in self.frame_rows})

    async def references(self) -> dict[str, ReferenceRange]:
        self._check()
        return dict(self.refs)

    async def create_reference(self, ref: ReferenceRange) -> ReferenceRange:
        self._check()
        return self.refs.setdefault(ref.camera_id, ref)

    async def set_reference(self, ref: ReferenceRange) -> None:
        self._check()
        self.refs[ref.camera_id] = replace(ref)

    async def delete_reference(self, camera_id: str) -> bool:
        return self.refs.pop(camera_id, None) is not None

    async def frames(self, cameras: list[str], start: datetime, end: datetime) -> list[FrameRow]:
        self._check()
        self.reads += 1
        return sorted(
            (f for f in self.frame_rows if f.camera_id in cameras and start <= f.ts < end),
            key=lambda f: (f.camera_id, f.ts),
        )

    async def detections(
        self, cameras: list[str], start: datetime, end: datetime, min_conf: float
    ) -> list[DetectionRow]:
        self._check()
        ts = {f.frame_id: f.ts for f in self.frame_rows}
        return [
            d
            for d in self.detection_rows
            if d.camera_id in cameras and start <= ts[d.frame_id] < end and d.confidence >= min_conf
        ]

    async def ready(self) -> bool:
        return not self.fail

    async def aclose(self) -> None:
        pass
