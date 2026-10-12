"""`python -m vms.storage`: database maintenance commands. Connects as POSTGRES_USER.

bootstrap      (run by make migrate after `alembic upgrade head`) brings a migrated database to
               its configured state, idempotently:
               1. the query service's read-only login (POSTGRES_READONLY_USER, vms.storage.roles);
               2. cameras and zones from the catalog (configs/cameras.yaml, vms.storage.catalog).
wait-drained   (make demo) waits until no frame is pending or processing, then prints what the
               pipeline stored: frames by status, tracks and events per camera.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import psycopg

from vms.common.clock import Clock, SystemClock
from vms.common.logs import configure_logging, get_logger
from vms.storage.catalog import DEFAULT_CATALOG, load_catalog, sync_catalog
from vms.storage.config import PostgresSettings
from vms.storage.roles import ReadonlyUserSettings, ensure_query_login

log = get_logger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vms.storage", description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    commands = parser.add_subparsers(dest="command", required=True)
    bootstrap = commands.add_parser("bootstrap", help="read-only login and camera catalog")
    bootstrap.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    bootstrap.add_argument("--db", help="database name (default: POSTGRES_DB)")
    drained = commands.add_parser("wait-drained", help="wait for the worker backlog to empty")
    drained.add_argument("--timeout-s", type=float, default=900.0)
    drained.add_argument("--interval-s", type=float, default=2.0)
    args = parser.parse_args(argv)
    configure_logging("storage", args.log_level)
    if args.command == "wait-drained":
        return _wait_drained(args.timeout_s, args.interval_s)

    catalog = load_catalog(args.catalog)  # validate before touching the database
    pg = PostgresSettings()
    if args.db:
        pg = pg.model_copy(update={"db": args.db})
    readonly = ReadonlyUserSettings()
    with psycopg.connect(pg.conninfo(application_name="vms-bootstrap")) as conn:
        ensure_query_login(conn, readonly.user, readonly.password)
        report = sync_catalog(conn, catalog)
    log.info(
        "bootstrap done",
        extra={
            "db": pg.db,
            "readonly_user": readonly.user,
            "cameras": report.cameras,
            "zones": report.zones,
        },
    )
    if report.stale_cameras or report.stale_zones:
        log.warning(
            "rows not in the catalog were kept",
            extra={"cameras": list(report.stale_cameras), "zones": list(report.stale_zones)},
        )
    return 0


BACKLOG_SQL = "SELECT count(*) FROM frames WHERE status IN ('pending', 'processing')"
SUMMARY_SQL = {
    "frames": "SELECT status, count(*) FROM frames GROUP BY 1 ORDER BY 1",
    "tracks": "SELECT camera_id, class_name, count(*) FROM tracks GROUP BY 1, 2 ORDER BY 1, 3 DESC",
    "events": "SELECT camera_id, event_type, count(*) FROM events GROUP BY 1, 2 ORDER BY 1, 2",
}


def wait_until_drained(
    backlog: Callable[[], int],
    *,
    timeout_s: float,
    interval_s: float,
    clock: Clock,
    sleep: Callable[[float], None],
    report: Callable[[int], None],
) -> bool:
    """Poll `backlog` until it is 0 (True) or `timeout_s` passes (False)."""
    deadline = clock.monotonic() + timeout_s
    while True:
        left = backlog()
        report(left)
        if left == 0:
            return True
        if clock.monotonic() >= deadline:
            return False
        sleep(interval_s)


def _wait_drained(timeout_s: float, interval_s: float) -> int:
    pg = PostgresSettings()
    with psycopg.connect(pg.conninfo(application_name="vms-wait-drained"), autocommit=True) as conn:

        def backlog() -> int:
            row = conn.execute(BACKLOG_SQL).fetchone()
            return int(row[0]) if row else 0

        ok = wait_until_drained(
            backlog,
            timeout_s=timeout_s,
            interval_s=interval_s,
            clock=SystemClock(),
            sleep=time.sleep,
            report=lambda n: log.info("worker backlog", extra={"frames": n}),
        )
        summary = {k: conn.execute(q).fetchall() for k, q in SUMMARY_SQL.items()}
    sys.stdout.write(format_summary(summary))
    if not ok:
        log.error("backlog not drained in time", extra={"timeout_s": timeout_s})
    return 0 if ok else 1


def format_summary(summary: dict[str, list[tuple[Any, ...]]]) -> str:
    lines = ["Pipeline output:"]
    frames = ", ".join(f"{status} {n}" for status, n in summary["frames"]) or "none"
    lines.append(f"  frames: {frames}")
    per_camera: dict[str, list[str]] = {}
    for camera, cls, n in summary["tracks"]:
        per_camera.setdefault(camera, []).append(f"{n} {cls}")
    for camera, kinds in per_camera.items():
        lines.append(f"  tracks {camera}: {', '.join(kinds)}")
    events: dict[str, list[str]] = {}
    for camera, kind, n in summary["events"]:
        events.setdefault(camera, []).append(f"{n} {kind}")
    for camera, kinds in events.items():
        lines.append(f"  events {camera}: {', '.join(kinds)}")
    return "\n".join(lines) + "\n"
