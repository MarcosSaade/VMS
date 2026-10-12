"""Semantic validation: checks of a schema-valid plan against the deployment and policy.

`QueryPlan` itself checks what is decidable from the plan alone (field combinations). Here the
plan meets live data: camera, zone and class names must exist, zones must belong to the plan's
cameras, the time range must be sane, and the limit must fit the row cap. Every problem is
reported at once, phrased so the planner can fix it in its single repair attempt (the messages
list the valid names).
"""

from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

from vms.query.catalog import QueryCatalog
from vms.query.plan import AbsoluteTimeRange, LastTimeRange, QueryPlan
from vms.query.settings import PlannerSettings
from vms.query.timerange import last_duration, resolve

MAX_LISTED_NAMES = 30


def _known(names: list[str] | tuple[str, ...]) -> str:
    shown = ", ".join(names[:MAX_LISTED_NAMES])
    return shown + (", ..." if len(names) > MAX_LISTED_NAMES else "")


def validate_plan(
    plan: QueryPlan,
    catalog: QueryCatalog,
    now: dt.datetime,
    settings: PlannerSettings,
    *,
    max_limit: int,
) -> list[str]:
    """Problems with `plan`, empty if it can be compiled and executed as asked."""
    errors: list[str] = []
    f = plan.filters

    camera_names = [c.name for c in catalog.cameras]
    unknown = [c for c in f.cameras if catalog.camera(c) is None]
    if unknown:
        errors.append(
            f"unknown camera(s) {', '.join(map(repr, unknown))}; known: {_known(camera_names)}"
        )

    zone_names = [z.name for z in catalog.zones]
    unknown = [z for z in f.zones if catalog.zone(z) is None]
    if unknown:
        errors.append(
            f"unknown zone(s) {', '.join(map(repr, unknown))}; known: {_known(zone_names)}"
        )
    elif f.cameras:
        camera_ids = {cam.camera_id for name in f.cameras if (cam := catalog.camera(name))}
        for name in f.zones:
            zone = catalog.zone(name)
            if zone is not None and zone.camera_id not in camera_ids:
                owner = catalog.camera_by_id(zone.camera_id)
                camera = owner.name if owner else zone.camera_id
                errors.append(
                    f"zone {name!r} is on camera {camera!r}, not on the cameras in filters.cameras"
                )

    unknown = [c for c in f.classes if c not in catalog.classes]
    if unknown:
        errors.append(
            f"unknown class(es) {', '.join(map(repr, unknown))}; known: {_known(catalog.classes)}"
        )

    max_range = dt.timedelta(days=settings.max_time_range_days)
    tr = f.time_range
    if isinstance(tr, LastTimeRange) and last_duration(tr) > max_range:
        errors.append(f"time range longer than {settings.max_time_range_days} days")
    elif isinstance(tr, AbsoluteTimeRange):
        if tr.end - tr.start > max_range:
            errors.append(f"time range longer than {settings.max_time_range_days} days")
        horizon = now + dt.timedelta(hours=settings.max_future_start_hours)
        # The earliest start over all camera timezones, so no camera's reading is rejected.
        zones = [c.tz for c in catalog.cameras] or [ZoneInfo("UTC")]
        if min(resolve(tr, tz, now).start for tz in zones) > horizon:
            errors.append(
                "time range starts in the future; a date without a year means the most "
                "recent such date"
            )

    if plan.limit is not None and plan.limit > max_limit:
        errors.append(f"limit {plan.limit} exceeds the maximum of {max_limit} rows")
    return errors
