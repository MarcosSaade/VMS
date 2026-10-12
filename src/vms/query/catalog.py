"""What the query service knows about the deployment: cameras, zones and object classes.

The planner prompt lists these names, semantic validation checks plans against them, and the
compiler resolves names to ids (bound as parameters) and picks each camera's timezone for time
ranges. The service loads cameras and zones from Postgres through the read-only login
(`load_query_catalog`), so it sees exactly what `make migrate` synced from
`configs/cameras.yaml`; the class list comes from `configs/query.yaml`.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class CameraInfo:
    camera_id: str
    name: str
    timezone: str

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


@dataclass(frozen=True)
class ZoneInfoRow:
    zone_id: str
    camera_id: str
    name: str


@dataclass(frozen=True)
class QueryCatalog:
    cameras: tuple[CameraInfo, ...]
    zones: tuple[ZoneInfoRow, ...]
    classes: tuple[str, ...]
    _by_name: dict[str, CameraInfo] = field(init=False, repr=False, compare=False)
    _by_id: dict[str, CameraInfo] = field(init=False, repr=False, compare=False)
    _zones: dict[str, ZoneInfoRow] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_by_name", {c.name: c for c in self.cameras})
        object.__setattr__(self, "_by_id", {c.camera_id: c for c in self.cameras})
        object.__setattr__(self, "_zones", {z.name: z for z in self.zones})

    def camera(self, name: str) -> CameraInfo | None:
        return self._by_name.get(name)

    def camera_by_id(self, camera_id: str) -> CameraInfo | None:
        return self._by_id.get(camera_id)

    def zone(self, name: str) -> ZoneInfoRow | None:
        return self._zones.get(name)

    def zones_of(self, camera_id: str) -> list[ZoneInfoRow]:
        return [z for z in self.zones if z.camera_id == camera_id]


CAMERAS_SQL = "SELECT camera_id, name, timezone FROM cameras ORDER BY name"
ZONES_SQL = "SELECT zone_id, camera_id, name FROM zones ORDER BY name"


def catalog_from_rows(
    cameras: Iterable[tuple[Any, ...]], zones: Iterable[tuple[Any, ...]], classes: Iterable[str]
) -> QueryCatalog:
    return QueryCatalog(
        cameras=tuple(CameraInfo(str(r[0]), str(r[1]), str(r[2])) for r in cameras),
        zones=tuple(ZoneInfoRow(str(r[0]), str(r[1]), str(r[2])) for r in zones),
        classes=tuple(classes),
    )
