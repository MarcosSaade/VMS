"""Zone events from tracks: enter, exit and loiter, per camera.

Each track is reduced to one anchor point per frame: the bottom centre of its box, roughly
where a person's feet or a car's wheels touch the ground. Zones are drawn on the ground, so
this is the point that "is in" a zone; the box centre would put a tall person standing in
front of a zone inside it. Polygons are in normalized frame coordinates (vms.storage.catalog).

Per (track, zone), a small state machine decides when the track is inside:

- Debounce: a change of side counts only after `min_frames` consecutive frames on the new
  side. A box jittering across the boundary therefore emits nothing, at the cost of reporting
  each crossing `min_frames - 1` frames late. The event's timestamp is the first frame on the
  new side, not the frame that confirmed it.
- Every track starts outside, so one first seen inside a zone (it walked in while occluded,
  or the camera just started) gets an enter like any other.
- Loiter: inside for `loiter_s` since the enter emits one loiter event per visit.
- Lost tracks: a track not seen for `lost_after_s` (the tracker's buffer, so a track the
  tracker could still re-find keeps its zone state) is finished: if inside, it exits at the
  last frame it was seen. Time here is frame capture time, never the wall clock, so the result
  depends only on the frames.

Visits open when a worker loses a camera (crash, takeover, shutdown) are left without an exit:
the next worker starts new tracks. Queries that pair enters with exits must allow for that.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

Point = tuple[float, float]


class ZoneSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_frames: int = Field(default=2, ge=1)  # consecutive frames to confirm a crossing
    loiter_s: float | None = Field(default=30.0, gt=0)  # null: no loiter events
    refresh_s: float = Field(default=60.0, gt=0)  # how often the worker reloads a camera's zones


class EventType(StrEnum):
    ENTER = "zone_enter"
    EXIT = "zone_exit"
    LOITER = "loiter"


@dataclass(frozen=True)
class ZoneDef:
    zone_id: str
    name: str
    polygon: tuple[Point, ...]


@dataclass(frozen=True)
class ZoneEvent:
    ts: datetime
    track_id: str
    zone_id: str
    event_type: EventType


EDGE_EPS = 1e-9


def point_in_polygon(point: Point, polygon: tuple[Point, ...]) -> bool:
    """Ray casting (even-odd rule); points on an edge count as inside. That matters for zones
    drawn to the frame border: a box touching the bottom of the frame has its anchor at y = 1.0,
    exactly on such a zone's edge."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if (
            abs(cross) <= EDGE_EPS
            and min(x1, x2) - EDGE_EPS <= x <= max(x1, x2) + EDGE_EPS
            and min(y1, y2) - EDGE_EPS <= y <= max(y1, y2) + EDGE_EPS
        ):
            return True
        if (y1 > y) != (y2 > y):
            x_cross = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < x_cross:
                inside = not inside
    return inside


def anchor(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Point:
    """Bottom centre of a pixel box, normalized to the frame."""
    return ((x1 + x2) / 2 / width, y2 / height)


@dataclass
class _Presence:
    inside: bool = False
    pending_side: bool | None = None
    pending_since: datetime | None = None
    pending_frames: int = 0
    entered_at: datetime | None = None
    loitered: bool = False


@dataclass
class _Track:
    last_seen: datetime
    zones: dict[str, _Presence] = field(default_factory=dict)


class ZoneTracker:
    """Zone state for the tracks of one camera. Call `update` once per tracked frame."""

    def __init__(self, zones: Iterable[ZoneDef], settings: ZoneSettings, lost_after_s: float):
        self._zones = list(zones)
        self._settings = settings
        self._lost_after = timedelta(seconds=lost_after_s)
        self._tracks: dict[str, _Track] = {}

    def set_zones(self, zones: Iterable[ZoneDef]) -> None:
        """Replace the zone set (config reload). State for removed zones is dropped silently."""
        self._zones = list(zones)
        ids = {z.zone_id for z in self._zones}
        for track in self._tracks.values():
            track.zones = {k: v for k, v in track.zones.items() if k in ids}

    @property
    def open_tracks(self) -> int:
        return len(self._tracks)

    def update(self, ts: datetime, anchors: dict[str, Point]) -> list[ZoneEvent]:
        """Advance to frame time `ts` with each visible track's anchor point."""
        events: list[ZoneEvent] = []
        for track_id, point in anchors.items():
            track = self._tracks.setdefault(track_id, _Track(last_seen=ts))
            track.last_seen = ts
            for zone in self._zones:
                presence = track.zones.setdefault(zone.zone_id, _Presence())
                observed = point_in_polygon(point, zone.polygon)
                events.extend(self._step(presence, observed, ts, track_id, zone.zone_id))
        events.extend(self._expire(ts))
        return events

    def _step(
        self, p: _Presence, observed: bool, ts: datetime, track_id: str, zone_id: str
    ) -> list[ZoneEvent]:
        events: list[ZoneEvent] = []
        if observed == p.inside:
            p.pending_side, p.pending_since, p.pending_frames = None, None, 0
        else:
            if p.pending_side is not observed:
                p.pending_side, p.pending_since, p.pending_frames = observed, ts, 0
            p.pending_frames += 1
            if p.pending_frames >= self._settings.min_frames:
                assert p.pending_since is not None
                since = p.pending_since
                p.inside = observed
                p.pending_side, p.pending_since, p.pending_frames = None, None, 0
                kind = EventType.ENTER if observed else EventType.EXIT
                events.append(ZoneEvent(since, track_id, zone_id, kind))
                p.entered_at = since if observed else None
                p.loitered = False
        loiter_s = self._settings.loiter_s
        if (
            p.inside
            and loiter_s is not None
            and not p.loitered
            and p.entered_at is not None
            and (ts - p.entered_at).total_seconds() >= loiter_s
        ):
            p.loitered = True
            events.append(ZoneEvent(ts, track_id, zone_id, EventType.LOITER))
        return events

    def _expire(self, now: datetime) -> list[ZoneEvent]:
        events: list[ZoneEvent] = []
        for track_id in [
            t for t, s in self._tracks.items() if now - s.last_seen > self._lost_after
        ]:
            track = self._tracks.pop(track_id)
            for zone_id, presence in track.zones.items():
                if presence.inside:
                    events.append(ZoneEvent(track.last_seen, track_id, zone_id, EventType.EXIT))
        return events
