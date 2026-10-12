"""The per-frame tracking step, shared by the worker and the renderer (`vms.worker.render`), so
what a review video shows is exactly what the worker stores.

Given one frame's detections and the camera's state, `track_frame` runs the tracker, turns
tracker ids into global track ids, moves the zone state machine, and applies the storage rule
(weak detections are kept only when tracked). See vms.worker.service for when it runs and
vms.worker.tracking / vms.worker.zones for the algorithms.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from vms.storage.work import DetectionRow, TrackUpdate
from vms.worker.detector import Detection
from vms.worker.tracking import TrackedBox
from vms.worker.zones import ZoneEvent, ZoneTracker, anchor


class Tracker(Protocol):
    def update(self, detections: list[Detection]) -> list[TrackedBox]: ...


@dataclass
class CameraState:
    """What a worker remembers about one camera it leases."""

    session: str  # part of every track id, new for each tracker
    tracker: Tracker
    zones: ZoneTracker
    zones_loaded_at: float  # monotonic; -inf forces a load
    last_key: tuple[datetime, int] | None = None  # (ts, seq) of the last tracked frame


@dataclass(frozen=True)
class TrackedFrame:
    detections: tuple[DetectionRow, ...]  # what is stored, with track ids where tracked
    tracks: tuple[TrackUpdate, ...]
    zone_events: tuple[ZoneEvent, ...]


def track_frame(
    found: list[Detection],
    *,
    camera_id: str,
    ts: datetime,
    seq: int,
    state: CameraState | None,
    width: int,
    height: int,
    store_min_conf: float,
) -> TrackedFrame:
    """Track one frame (if `state` is given; None means store without tracking)."""
    track_of: dict[int, str] = {}  # detection index -> global track id
    zone_events: list[ZoneEvent] = []
    if state is not None:
        for box in state.tracker.update(found):
            track_of[box.det_index] = f"{camera_id}:{state.session}:{box.local_id}"
        state.last_key = (ts, seq)
        anchors = {
            track_id: anchor(found[i].x1, found[i].y1, found[i].x2, found[i].y2, width, height)
            for i, track_id in track_of.items()
        }
        zone_events = state.zones.update(ts, anchors)

    kept = [i for i, d in enumerate(found) if d.confidence >= store_min_conf or i in track_of]
    detections = tuple(
        DetectionRow(
            n,
            found[i].class_name,
            found[i].confidence,
            found[i].x1,
            found[i].y1,
            found[i].x2,
            found[i].y2,
            track_of.get(i),
        )
        for n, i in enumerate(kept)
    )
    tracks = tuple(TrackUpdate(t, found[i].class_name) for i, t in track_of.items())
    return TrackedFrame(detections, tracks, tuple(zone_events))
