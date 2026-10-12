"""Deterministic `QueryPlan` -> parameterized Postgres SQL (one compiler per backend, D1).

The compiler is the only place SQL text is produced for the plan engine, and it builds that
text only from constants in this module. Every value that came from the plan (camera and zone
ids, class names, thresholds, time bounds) is bound as a `%(pN)s` parameter, so a question can
never become SQL syntax. The one exception is LIMIT: the validator requires an integer literal
(D14), and the plan's limit is an `int` already bounded by the schema and the configured cap.

Semantics, so the same plan means the same thing on every backend (equivalence tests, D1):

- Camera scope: every query is restricted to catalog cameras: the plan's cameras, else the
  cameras of the plan's zones, else all of them. Frames from cameras not in the catalog (load
  tests) are never counted.
- Time: ranges are camera-local (vms.query.timerange). Cameras are grouped by timezone and each
  group gets its own UTC bounds, OR-ed together. Detections, events and frames match on `ts`;
  a track matches if it was visible during the range (`last_seen >= start AND first_seen <
  end`), so a track spanning midnight counts on both days.
- Tracks filtered or grouped by zone are the tracks that entered the zone during the range
  (zone_enter events), counted once per zone.
- Events filtered or grouped by class use their track's class.
- Frames filtered by class or confidence are frames with at least one such detection.
- Dwell without zones is time on camera (`last_seen - first_seen`). Dwell in zones is first
  zone_enter to last zone_exit per track and zone, from events inside the range; visits with no
  exit (still inside, or the camera stopped) are left out.
- Trend buckets are camera-local wall-clock times; a track falls in the bucket of `first_seen`.
- Ordering is total (ties broken by the group keys or ids), so results are reproducible.
"""

from __future__ import annotations

import datetime as dt
import itertools
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any
from zoneinfo import ZoneInfo

from vms.query.catalog import CameraInfo, QueryCatalog, ZoneInfoRow
from vms.query.plan import (
    Direction,
    GroupKey,
    Intent,
    MeasureField,
    OrderBy,
    QueryPlan,
    ResultKind,
    Target,
    result_kind,
)
from vms.query.timerange import resolve


class CompileError(ValueError):
    """The plan cannot be compiled (it should have failed semantic validation first)."""


@dataclass(frozen=True)
class CompiledQuery:
    sql: str  # with %(pN)s placeholders
    params: dict[str, Any]
    result_kind: ResultKind
    limit: int


@dataclass(frozen=True)
class _Table:
    name: str
    alias: str
    order_col: str  # the "time" column for ordering and buckets
    id_col: str  # tie-breaker


_TABLES = {
    Target.TRACKS: _Table("tracks", "t", "first_seen", "track_id"),
    Target.DETECTIONS: _Table("detections", "d", "ts", "frame_id"),
    Target.EVENTS: _Table("events", "e", "ts", "event_id"),
    Target.FRAMES: _Table("frames", "f", "ts", "frame_id"),
}

# Columns returned by `list` and `frames`, after the select of the group keys.
_LIST_COLUMNS = {
    Target.TRACKS: [
        "t.track_id",
        "c.name AS camera",
        "t.class_name",
        "t.first_seen",
        "t.last_seen",
        "t.n_frames",
    ],
    Target.EVENTS: [
        "e.event_id",
        "e.ts",
        "c.name AS camera",
        "z.name AS zone",
        "e.event_type",
        "e.track_id",
    ],
    Target.DETECTIONS: [
        "d.ts",
        "c.name AS camera",
        "d.frame_id",
        "d.track_id",
        "d.class_name",
        "d.confidence",
        "d.x1",
        "d.y1",
        "d.x2",
        "d.y2",
    ],
    Target.FRAMES: [
        "f.frame_id",
        "f.ts",
        "c.name AS camera",
        "f.object_key",
        "f.width",
        "f.height",
    ],
}

_DWELL_ON_CAMERA = "EXTRACT(EPOCH FROM t.last_seen - t.first_seen)"
_DWELL_IN_ZONE = "EXTRACT(EPOCH FROM v.exited - v.entered)"


class _Params:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self._n = itertools.count()

    def add(self, value: Any) -> str:
        name = f"p{next(self._n)}"
        self.values[name] = value
        return f"%({name})s"

    def in_list(self, column: str, values: Sequence[Any]) -> str:
        if not values:
            return "FALSE"
        return f"{column} IN ({', '.join(self.add(v) for v in values)})"


@dataclass
class _Query:
    select: list[str] = field(default_factory=list)
    from_: str = ""
    joins: list[str] = field(default_factory=list)
    where: list[str] = field(default_factory=list)
    group_by: list[str] = field(default_factory=list)
    order_by: list[str] = field(default_factory=list)
    limit: int | None = None

    def join(self, clause: str) -> None:
        if clause not in self.joins:
            self.joins.append(clause)

    def render(self) -> str:
        parts = [f"SELECT {', '.join(self.select)}", f"FROM {self.from_}", *self.joins]
        if self.where:
            parts.append("WHERE " + " AND ".join(self.where))
        if self.group_by:
            parts.append("GROUP BY " + ", ".join(self.group_by))
        if self.order_by:
            parts.append("ORDER BY " + ", ".join(self.order_by))
        if self.limit is not None:
            parts.append(f"LIMIT {int(self.limit)}")
        return " ".join(parts)


class PostgresCompiler:
    """Compiles validated plans. `now` comes from the service's injected clock."""

    def __init__(self, catalog: QueryCatalog, *, default_limit: int, max_limit: int) -> None:
        self._catalog = catalog
        self._default_limit = default_limit
        self._max_limit = max_limit

    def compile(self, plan: QueryPlan, now: dt.datetime) -> CompiledQuery:
        if plan.intent is Intent.ABSTAIN:
            raise CompileError("an abstaining plan has no SQL")
        if plan.target is None:  # pragma: no cover - QueryPlan guarantees it
            raise CompileError("plan has no target")
        if plan.visual_query is not None:
            raise CompileError("visual_query needs CLIP search, which is not available")
        return _Compilation(self, plan, now).run()

    # Helpers shared with _Compilation.
    def _cameras(self, plan: QueryPlan) -> list[CameraInfo]:
        f = plan.filters
        if f.cameras:
            names = f.cameras
        elif f.zones:
            ids = {self._zone(z).camera_id for z in f.zones}
            names = [c.name for c in self._catalog.cameras if c.camera_id in ids]
        else:
            names = [c.name for c in self._catalog.cameras]
        cameras = []
        for name in sorted(set(names)):
            camera = self._catalog.camera(name)
            if camera is None:
                raise CompileError(f"unknown camera {name!r}")
            cameras.append(camera)
        return cameras

    def _zone(self, name: str) -> ZoneInfoRow:
        zone = self._catalog.zone(name)
        if zone is None:
            raise CompileError(f"unknown zone {name!r}")
        return zone


class _Compilation:
    def __init__(self, compiler: PostgresCompiler, plan: QueryPlan, now: dt.datetime) -> None:
        assert plan.target is not None
        self.c = compiler
        self.plan = plan
        self.target = plan.target
        self.table = _TABLES[plan.target]
        self.now = now
        self.p = _Params()
        self.q = _Query()
        self.cameras = compiler._cameras(plan)
        self.zone_ids = sorted({compiler._zone(z).zone_id for z in plan.filters.zones})
        self.by_zone = GroupKey.ZONE in plan.group_by
        # Dwell in zones reads visits built from events instead of the tracks table.
        self.visits = plan.intent is Intent.DWELL and bool(self.zone_ids or self.by_zone)

    # --- scope -----------------------------------------------------------------------------

    def scope(self, alias: str, *, overlap: bool = False, timed: bool = True) -> str:
        """Camera and time condition on `alias`: one OR branch per timezone group."""
        tr = self.plan.filters.time_range
        ids = [cam.camera_id for cam in self.cameras]
        if tr is None or not timed:
            return self.p.in_list(f"{alias}.camera_id", ids)
        groups: dict[str, list[str]] = {}
        for cam in self.cameras:
            groups.setdefault(cam.timezone, []).append(cam.camera_id)
        if not groups:
            return "FALSE"
        branches = []
        for tz_name in sorted(groups):
            bounds = resolve(tr, ZoneInfo(tz_name), self.now)
            start, end = self.p.add(bounds.start), self.p.add(bounds.end)
            if overlap:
                time = f"{alias}.last_seen >= {start} AND {alias}.first_seen < {end}"
            else:
                time = f"{alias}.ts >= {start} AND {alias}.ts < {end}"
            cams = self.p.in_list(f"{alias}.camera_id", groups[tz_name])
            branches.append(f"({cams} AND {time})")
        return branches[0] if len(branches) == 1 else "(" + " OR ".join(branches) + ")"

    def zone_entries(self, columns: str, distinct: bool) -> str:
        """Subquery of zone entries in scope (for tracks filtered or grouped by zone)."""
        where = ["e.event_type = 'zone_enter'", self.scope("e")]
        if self.zone_ids:
            where.append(self.p.in_list("e.zone_id", self.zone_ids))
        keyword = "SELECT DISTINCT" if distinct else "SELECT"
        return f"{keyword} {columns} FROM events e WHERE {' AND '.join(where)}"

    # --- building blocks ---------------------------------------------------------------------

    def base(self) -> None:
        f, a = self.plan.filters, self.table.alias
        if self.visits:
            self.base_visits()
            return
        self.q.from_ = f"{self.table.name} {a}"
        if self.target is Target.TRACKS:
            if self.zone_ids or self.by_zone:
                self.q.where.append(self.scope("t", timed=False))
                if self.by_zone:
                    pairs = self.zone_entries("e.track_id, e.zone_id", distinct=True)
                    self.q.join(f"JOIN ({pairs}) tz ON tz.track_id = t.track_id")
                    self.q.join("JOIN zones z ON z.zone_id = tz.zone_id")
                else:
                    entries = self.zone_entries("e.track_id", distinct=False)
                    self.q.where.append(f"t.track_id IN ({entries})")
            else:
                self.q.where.append(self.scope("t", overlap=True))
            if f.classes:
                self.q.where.append(self.p.in_list("t.class_name", f.classes))
        elif self.target is Target.DETECTIONS:
            self.q.where.append(self.scope("d"))
            if f.classes:
                self.q.where.append(self.p.in_list("d.class_name", f.classes))
            if f.min_confidence is not None:
                self.q.where.append(f"d.confidence >= {self.p.add(f.min_confidence)}")
        elif self.target is Target.EVENTS:
            self.q.where.append(self.scope("e"))
            if self.zone_ids:
                self.q.where.append(self.p.in_list("e.zone_id", self.zone_ids))
            if f.event_types:
                self.q.where.append(self.p.in_list("e.event_type", [str(t) for t in f.event_types]))
            if f.classes or GroupKey.CLASS_NAME in self.plan.group_by:
                self.q.join("JOIN tracks t ON t.track_id = e.track_id")
            if f.classes:
                self.q.where.append(self.p.in_list("t.class_name", f.classes))
        else:  # frames
            self.q.where.append(self.scope("f"))
            if f.classes or f.min_confidence is not None:
                sub = [self.scope("d")]
                if f.classes:
                    sub.append(self.p.in_list("d.class_name", f.classes))
                if f.min_confidence is not None:
                    sub.append(f"d.confidence >= {self.p.add(f.min_confidence)}")
                detections = f"SELECT d.frame_id FROM detections d WHERE {' AND '.join(sub)}"
                self.q.where.append(f"f.frame_id IN ({detections})")

    def base_visits(self) -> None:
        f = self.plan.filters
        where = ["e.event_type IN ('zone_enter', 'zone_exit')", self.scope("e")]
        if self.zone_ids:
            where.append(self.p.in_list("e.zone_id", self.zone_ids))
        visits = (
            "SELECT e.track_id, e.camera_id, e.zone_id, "
            "min(e.ts) FILTER (WHERE e.event_type = 'zone_enter') AS entered, "
            "max(e.ts) FILTER (WHERE e.event_type = 'zone_exit') AS exited "
            f"FROM events e WHERE {' AND '.join(where)} "
            "GROUP BY e.track_id, e.camera_id, e.zone_id"
        )
        self.q.from_ = f"({visits}) v"
        self.q.join("JOIN tracks t ON t.track_id = v.track_id")
        self.q.where.append("v.exited > v.entered")
        if f.classes:
            self.q.where.append(self.p.in_list("t.class_name", f.classes))

    def camera_alias(self) -> str:
        return "v" if self.visits else self.table.alias

    def join_cameras(self) -> None:
        self.q.join(f"JOIN cameras c ON c.camera_id = {self.camera_alias()}.camera_id")

    def group_keys(self) -> list[str]:
        """Add the plan's group keys to SELECT and GROUP BY; returns their output names."""
        names = []
        for key in self.plan.group_by:
            if key is GroupKey.CAMERA:
                self.join_cameras()
                expr, name = "c.name", "camera"
            elif key is GroupKey.ZONE:
                if self.visits:
                    self.q.join("JOIN zones z ON z.zone_id = v.zone_id")
                elif self.target is Target.EVENTS:
                    self.q.join("JOIN zones z ON z.zone_id = e.zone_id")
                expr, name = "z.name", "zone"
            elif key is GroupKey.CLASS_NAME:
                alias = "d" if self.target is Target.DETECTIONS else "t"
                expr, name = f"{alias}.class_name", "class_name"
            else:
                expr, name = "e.event_type", "event_type"
            self.q.select.append(expr if expr.endswith(f".{name}") else f"{expr} AS {name}")
            self.q.group_by.append(expr)
            names.append(name)
        return names

    def measure_expr(self) -> str:
        measure = self.plan.measure
        assert measure is not None
        if measure.field is MeasureField.DWELL_SECONDS:
            return _DWELL_IN_ZONE if self.visits else _DWELL_ON_CAMERA
        if measure.field is MeasureField.CONFIDENCE:
            return "d.confidence"
        return "t.n_frames"

    def direction(self, default: Direction) -> str:
        order = self.plan.order
        return (order.direction if order is not None else default).value.upper()

    # --- intents -----------------------------------------------------------------------------

    def run(self) -> CompiledQuery:
        self.base()
        intent = self.plan.intent
        if intent in (Intent.COUNT, Intent.TOP_K):
            self.count()
        elif intent in (Intent.AGGREGATE, Intent.DWELL):
            self.aggregate()
        elif intent is Intent.TREND:
            self.trend()
        else:  # list, frames
            self.listing()
        limit = min(self.q.limit or self.c._max_limit, self.c._max_limit)
        self.q.limit = limit
        return CompiledQuery(
            sql=self.q.render(),
            params=self.p.values,
            result_kind=result_kind(self.plan),
            limit=limit,
        )

    def grouped_order(self, metric: str, keys: list[str]) -> None:
        self.q.order_by = [f"{metric} {self.direction(Direction.DESC)}", *keys]

    def count(self) -> None:
        keys = self.group_keys()
        self.q.select.append("count(*) AS n")
        if keys:
            self.grouped_order("n", keys)
            self.q.limit = self.plan.limit or self.c._max_limit
        else:
            self.q.limit = 1

    def aggregate(self) -> None:
        measure = self.plan.measure
        assert measure is not None
        keys = self.group_keys()
        self.q.select.append(f"{measure.op.value}({self.measure_expr()}) AS value")
        if keys:
            self.grouped_order("value", keys)
            self.q.limit = self.plan.limit or self.c._max_limit
        else:
            self.q.limit = 1

    def trend(self) -> None:
        assert self.plan.bucket is not None
        keys = self.group_keys()
        self.join_cameras()
        a, col = self.table.alias, self.table.order_col
        bucket = f"date_trunc('{self.plan.bucket.value}', {a}.{col} AT TIME ZONE c.timezone)"
        self.q.select.extend([f"{bucket} AS bucket", "count(*) AS n"])
        self.q.group_by.append(bucket)
        order = self.plan.order
        if order is not None and order.by is OrderBy.COUNT:
            self.q.order_by = [f"n {self.direction(Direction.DESC)}", "bucket", *keys]
        else:
            self.q.order_by = [f"bucket {self.direction(Direction.ASC)}", *keys]
        self.q.limit = self.plan.limit or self.c._max_limit

    def listing(self) -> None:
        self.join_cameras()
        if self.target is Target.EVENTS:
            self.q.join("JOIN zones z ON z.zone_id = e.zone_id")
        a, t = self.table.alias, self.table
        self.q.select.extend(_LIST_COLUMNS[self.target])
        direction = self.direction(Direction.DESC)
        self.q.order_by = [f"{a}.{t.order_col} {direction}", f"{a}.{t.id_col} {direction}"]
        if self.target is Target.DETECTIONS:
            # det_idx (the key) is not readable by the query role; these make ties unlikely.
            self.q.order_by.extend(f"d.{col} {direction}" for col in ("class_name", "x1", "y1"))
        self.q.limit = self.plan.limit or self.c._default_limit
