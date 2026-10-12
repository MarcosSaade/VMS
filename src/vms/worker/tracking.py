"""Per-camera multi-object tracking with ByteTrack (the Ultralytics implementation).

ByteTrack predicts where each track should be in the next frame (a Kalman filter on the box)
and matches detections to tracks by box overlap (IoU) in two rounds: confident detections
first, then the leftover tracks against low-confidence detections. The second round is what
keeps a track alive through partial occlusion, when the detector is unsure; it is why the
detector threshold is set low (0.1) and storage keeps low-confidence boxes only when they
belong to a track. A track missing for `buffer_s` is dropped; seen again before that, it
keeps its id.

A tracker's state is the history of one camera, so it lives in the one worker that leases
that camera (D20) and must see the camera's frames in capture order. Tracker ids restart at
1 for each tracker; the worker turns them into globally unique ids (`TrackIds`).

ByteTrack associates boxes by position, not class, so a track keeps the class it was created
with; each detection row keeps its own class.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from vms.worker.detector import Detection


class TrackerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fps: float = Field(default=5.0, gt=0)  # the cameras' analysis rate (ARCHITECTURE 2)
    high_thresh: float = Field(default=0.25, ge=0, le=1)  # first association round
    low_thresh: float = Field(default=0.1, ge=0, le=1)  # second round, for unsure detections
    new_track_thresh: float = Field(default=0.25, ge=0, le=1)  # start a track from this score
    match_thresh: float = Field(default=0.8, ge=0, le=1)  # max association cost (1 - IoU)
    fuse_score: bool = True  # weight association cost by detection score
    buffer_s: float = Field(default=2.0, gt=0)  # a lost track is kept this long
    # Classes that get tracks and zone events; others are stored as plain detections.
    classes: list[str] = Field(
        default_factory=lambda: ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
    )
    # Detections below this are stored only when they belong to a track.
    store_min_conf: float = Field(default=0.25, ge=0, le=1)

    @model_validator(mode="after")
    def _check(self) -> TrackerSettings:
        if self.low_thresh > self.high_thresh:
            raise ValueError("low_thresh must not exceed high_thresh")
        return self

    @property
    def buffer_frames(self) -> int:
        return max(1, round(self.buffer_s * self.fps))


class _Boxes:
    """The minimal results-like object BYTETracker reads: conf, cls, xywh, mask indexing."""

    def __init__(
        self, xywh: NDArray[np.float32], conf: NDArray[np.float32], cls: NDArray[np.float32]
    ) -> None:
        self.xywh, self.conf, self.cls = xywh, conf, cls

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, mask: NDArray[np.bool_]) -> _Boxes:
        return _Boxes(self.xywh[mask], self.conf[mask], self.cls[mask])


@dataclass(frozen=True)
class TrackedBox:
    local_id: int  # unique within this tracker
    det_index: int  # which input detection it was matched to in this frame


class CameraTracker:
    """ByteTrack for one camera. Call `update` once per frame, in capture order."""

    def __init__(self, settings: TrackerSettings) -> None:
        bytetrack: Any = importlib.import_module("ultralytics.trackers.byte_tracker")
        self._settings = settings
        self._classes = {name: i for i, name in enumerate(settings.classes)}
        self._tracker: Any = bytetrack.BYTETracker(
            SimpleNamespace(
                track_high_thresh=settings.high_thresh,
                track_low_thresh=settings.low_thresh,
                new_track_thresh=settings.new_track_thresh,
                track_buffer=settings.buffer_frames,
                match_thresh=settings.match_thresh,
                fuse_score=settings.fuse_score,
            )
        )

    def update(self, detections: list[Detection]) -> list[TrackedBox]:
        """Feed one frame's detections; return the ones assigned to a confirmed track.

        Detections of classes not in `settings.classes` are ignored (never tracked). A new
        object gets its track from its second frame on: ByteTrack confirms a track only when
        it is matched again (except on the tracker's very first frame).
        """
        indices = [i for i, d in enumerate(detections) if d.class_name in self._classes]
        if indices:
            boxes = np.array(
                [
                    [
                        (detections[i].x1 + detections[i].x2) / 2,
                        (detections[i].y1 + detections[i].y2) / 2,
                        detections[i].x2 - detections[i].x1,
                        detections[i].y2 - detections[i].y1,
                    ]
                    for i in indices
                ],
                dtype=np.float32,
            )
            conf = np.array([detections[i].confidence for i in indices], dtype=np.float32)
            cls = np.array(
                [self._classes[detections[i].class_name] for i in indices], dtype=np.float32
            )
        else:
            boxes = np.zeros((0, 4), dtype=np.float32)
            conf = np.zeros(0, dtype=np.float32)
            cls = np.zeros(0, dtype=np.float32)
        out = self._tracker.update(_Boxes(boxes, conf, cls))
        # Rows: x1, y1, x2, y2, track_id, score, cls, idx (idx into the boxes passed in).
        return [TrackedBox(int(row[4]), indices[int(row[7])]) for row in out]
