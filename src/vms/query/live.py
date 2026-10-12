"""The camera wall: per catalog camera, the latest processed frame, its detections and zones.

Fixed queries written here (no user input besides catalog camera ids, bound as parameters),
run on the read-only login like everything else in the query service. "Processed" means the
worker stored a model version on the frame. The latest frame per camera is found with one
index probe each on `frames (camera_id, ts)`, newest first, rather than a DISTINCT ON over
the whole table.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel

from vms.query.catalog import QueryCatalog
from vms.query.engine import frame_image_path
from vms.query.store import Reader

LATEST_FRAME_SQL = """
SELECT frame_id, ts, width, height FROM frames
WHERE camera_id = %(camera_id)s AND model_version IS NOT NULL
ORDER BY ts DESC LIMIT 1
"""
DETECTIONS_SQL = """
SELECT frame_id, class_name, confidence, track_id, x1, y1, x2, y2 FROM detections
WHERE frame_id = ANY(%(frame_ids)s)
ORDER BY frame_id, confidence DESC
"""
ZONES_SQL = "SELECT camera_id, name, polygon FROM zones ORDER BY camera_id, name"


class LiveDetection(BaseModel):
    class_name: str
    confidence: float
    track_id: str | None
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels


class LiveZone(BaseModel):
    name: str
    polygon: list[tuple[float, float]]  # normalized 0..1


class LiveFrame(BaseModel):
    frame_id: str
    ts: dt.datetime
    width: int | None
    height: int | None
    image_url: str
    detections: list[LiveDetection]


class LiveCamera(BaseModel):
    camera_id: str
    name: str
    timezone: str
    zones: list[LiveZone]
    frame: LiveFrame | None


async def live_view(reader: Reader, catalog: QueryCatalog) -> list[LiveCamera]:
    latest: dict[str, tuple[Any, ...]] = {}
    for cam in catalog.cameras:
        rows = await reader.fetch(LATEST_FRAME_SQL, {"camera_id": cam.camera_id})
        if rows.rows:
            latest[cam.camera_id] = rows.rows[0]

    detections: dict[str, list[LiveDetection]] = {}
    if latest:
        frame_ids = [row[0] for row in latest.values()]
        found = await reader.fetch(DETECTIONS_SQL, {"frame_ids": frame_ids})
        for frame_id, cls, conf, track, x1, y1, x2, y2 in found.rows:
            detections.setdefault(frame_id, []).append(
                LiveDetection(
                    class_name=cls,
                    confidence=float(conf),
                    track_id=track,
                    box=(float(x1), float(y1), float(x2), float(y2)),
                )
            )

    zones: dict[str, list[LiveZone]] = {}
    for camera_id, name, polygon in (await reader.fetch(ZONES_SQL)).rows:
        points = [(float(p[0]), float(p[1])) for p in polygon]
        zones.setdefault(camera_id, []).append(LiveZone(name=name, polygon=points))

    cameras = []
    for cam in catalog.cameras:
        row = latest.get(cam.camera_id)
        frame = None
        if row is not None:
            frame_id, ts, width, height = row
            frame = LiveFrame(
                frame_id=frame_id,
                ts=ts,
                width=width,
                height=height,
                image_url=frame_image_path(frame_id),
                detections=detections.get(frame_id, []),
            )
        cameras.append(
            LiveCamera(
                camera_id=cam.camera_id,
                name=cam.name,
                timezone=cam.timezone,
                zones=zones.get(cam.camera_id, []),
                frame=frame,
            )
        )
    return cameras
