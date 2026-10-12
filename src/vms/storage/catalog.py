"""The camera catalog: cameras and their zones, versioned in `configs/cameras.yaml`.

The YAML is the source of truth; `sync_catalog` upserts it into the `cameras` and `zones`
tables (`python -m vms.storage bootstrap`, run by `make migrate`). Rows missing from the YAML
are reported, not deleted: detections, tracks and events keep referring to them by id.

Names are what questions and `QueryPlan` filters use (`entrance`, `front_door`), so they are
unique: camera names across the catalog, and zone names across the catalog too, because a plan
names a zone without its camera. A zone's id is derived as `<camera_id>:<name>`, so renaming a
zone creates a new zone rather than relabeling old events.

Polygons are in normalized frame coordinates (0..1, origin top-left), so a zone does not depend
on the resolution a camera sends. They must be simple (no self-intersection) with non-zero area,
or point-in-polygon tests in the zone-event logic (task 1.6) become ambiguous.
"""

from __future__ import annotations

import itertools
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

import psycopg
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vms.common.ingest_api import SAFE_ID_PATTERN

DEFAULT_CATALOG = Path("configs/cameras.yaml")
NAME_PATTERN = r"^[a-z][a-z0-9_]{0,63}$"
MAX_POLYGON_POINTS = 64

Point = tuple[float, float]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Zone(_Model):
    name: str = Field(pattern=NAME_PATTERN)
    polygon: tuple[Point, ...] = Field(min_length=3, max_length=MAX_POLYGON_POINTS)

    @field_validator("polygon")
    @classmethod
    def _check_polygon(cls, polygon: tuple[Point, ...]) -> tuple[Point, ...]:
        for x, y in polygon:
            if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
                raise ValueError(f"point ({x}, {y}) is outside the normalized frame [0, 1]")
        if not is_simple(polygon):
            raise ValueError("polygon edges intersect each other or a vertex repeats")
        if signed_area(polygon) == 0.0:
            raise ValueError("polygon has zero area")
        return polygon


class Camera(_Model):
    camera_id: str = Field(pattern=SAFE_ID_PATTERN)
    name: str = Field(pattern=NAME_PATTERN)
    site: str = Field(min_length=1)
    timezone: str
    zones: tuple[Zone, ...] = ()

    @field_validator("timezone")
    @classmethod
    def _check_timezone(cls, value: str) -> str:
        # ZoneInfo alone also accepts file paths and odd aliases; require a listed IANA name.
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown IANA timezone {value!r}") from exc
        if value not in available_timezones():
            raise ValueError(f"unknown IANA timezone {value!r}")
        return value

    def zone_id(self, zone: Zone) -> str:
        return f"{self.camera_id}:{zone.name}"


class CameraCatalog(_Model):
    version: int = Field(ge=1)
    cameras: tuple[Camera, ...]

    @model_validator(mode="after")
    def _unique(self) -> CameraCatalog:
        _require_unique("camera_id", [c.camera_id for c in self.cameras])
        _require_unique("camera name", [c.name for c in self.cameras])
        _require_unique("zone name", [z.name for c in self.cameras for z in c.zones])
        return self


def _require_unique(what: str, values: list[str]) -> None:
    dupes = sorted(v for v, n in Counter(values).items() if n > 1)
    if dupes:
        raise ValueError(f"duplicate {what}: {', '.join(dupes)}")


def load_catalog(path: Path = DEFAULT_CATALOG) -> CameraCatalog:
    with path.open(encoding="utf-8") as fh:
        data: Any = yaml.safe_load(fh)
    return CameraCatalog.model_validate(data)


# --- Geometry (small polygons from config; O(n^2) is fine) --------------------------------


def signed_area(polygon: tuple[Point, ...]) -> float:
    """Shoelace formula; the sign gives the winding order."""
    pairs = zip(polygon, polygon[1:] + polygon[:1], strict=True)
    return 0.5 * sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in pairs)


def is_simple(polygon: tuple[Point, ...]) -> bool:
    """True if no two non-adjacent edges touch and no vertex repeats."""
    if len(set(polygon)) != len(polygon):
        return False
    n = len(polygon)
    edges = [(polygon[i], polygon[(i + 1) % n]) for i in range(n)]
    for i, j in itertools.combinations(range(n), 2):
        adjacent = j == i + 1 or (i == 0 and j == n - 1)
        if not adjacent and _segments_touch(*edges[i], *edges[j]):
            return False
    return True


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, p: Point) -> bool:
    return min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])


def _segments_touch(a: Point, b: Point, c: Point, d: Point) -> bool:
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)
    if ((o1 > 0 > o2) or (o1 < 0 < o2)) and ((o3 > 0 > o4) or (o3 < 0 < o4)):
        return True
    return (
        (o1 == 0 and _on_segment(a, b, c))
        or (o2 == 0 and _on_segment(a, b, d))
        or (o3 == 0 and _on_segment(c, d, a))
        or (o4 == 0 and _on_segment(c, d, b))
    )


# --- Sync to Postgres ------------------------------------------------------------------------


@dataclass(frozen=True)
class SyncReport:
    cameras: int  # rows upserted
    zones: int
    stale_cameras: tuple[str, ...]  # in the database but not in the catalog (kept)
    stale_zones: tuple[str, ...]


UPSERT_CAMERA_SQL = """
INSERT INTO cameras (camera_id, name, site, timezone)
VALUES (%(camera_id)s, %(name)s, %(site)s, %(timezone)s)
ON CONFLICT (camera_id) DO UPDATE
SET name = EXCLUDED.name, site = EXCLUDED.site, timezone = EXCLUDED.timezone
"""

UPSERT_ZONE_SQL = """
INSERT INTO zones (zone_id, camera_id, name, polygon)
VALUES (%(zone_id)s, %(camera_id)s, %(name)s, %(polygon)s::jsonb)
ON CONFLICT (zone_id) DO UPDATE SET polygon = EXCLUDED.polygon
"""


def sync_catalog(conn: psycopg.Connection[Any], catalog: CameraCatalog) -> SyncReport:
    """Upsert the catalog in one transaction; nothing is deleted."""
    cameras = [
        {"camera_id": c.camera_id, "name": c.name, "site": c.site, "timezone": c.timezone}
        for c in catalog.cameras
    ]
    zones = [
        {
            "zone_id": c.zone_id(z),
            "camera_id": c.camera_id,
            "name": z.name,
            "polygon": json.dumps([list(p) for p in z.polygon]),
        }
        for c in catalog.cameras
        for z in c.zones
    ]
    with conn.transaction(), conn.cursor() as cur:
        if cameras:
            cur.executemany(UPSERT_CAMERA_SQL, cameras)
        if zones:
            cur.executemany(UPSERT_ZONE_SQL, zones)
        db_cameras = {r[0] for r in cur.execute("SELECT camera_id FROM cameras").fetchall()}
        db_zones = {r[0] for r in cur.execute("SELECT zone_id FROM zones").fetchall()}
    return SyncReport(
        cameras=len(cameras),
        zones=len(zones),
        stale_cameras=tuple(sorted(db_cameras - {c["camera_id"] for c in cameras})),
        stale_zones=tuple(sorted(db_zones - {z["zone_id"] for z in zones})),
    )
