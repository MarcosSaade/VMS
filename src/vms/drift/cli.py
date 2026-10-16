"""`python -m vms.drift`: the drift monitor and its reference management.

run                               (default; make drift) per-camera drift statistics as gauges
references                        list each camera's reference range
recalibrate CAMERA ... | --all    start a new calibration period now (the next reference_s of
                                  capture time), or set one with --start and --end (ISO 8601,
                                  a period known to be clean); the monitor picks it up on its
                                  next update
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from datetime import datetime, timedelta

from vms.common.clock import Clock, SystemClock
from vms.common.logs import configure_logging, get_logger
from vms.common.settings import load_settings
from vms.drift.settings import DriftSettings
from vms.drift.store import DriftStore, PostgresDriftStore, ReferenceRange
from vms.storage.config import PostgresSettings

log = get_logger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vms.drift",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("run", help="run the drift monitor (default)")
    commands.add_parser("references", help="list reference ranges")
    recal = commands.add_parser("recalibrate", help="replace cameras' reference ranges")
    recal.add_argument("cameras", nargs="*")
    recal.add_argument("--all", action="store_true", help="every camera with a reference")
    recal.add_argument("--start", type=datetime.fromisoformat)
    recal.add_argument("--end", type=datetime.fromisoformat)
    args = parser.parse_args(argv)

    if args.command in (None, "run"):
        from vms.drift.main import serve

        return serve()

    configure_logging("drift", "INFO")
    settings = load_settings(DriftSettings, "drift")
    if args.command == "recalibrate":
        if bool(args.cameras) == args.all:
            parser.error("name cameras or pass --all (not both)")
        if (args.start is None) != (args.end is None):
            parser.error("--start and --end go together")
        for ts in (args.start, args.end):
            if ts is not None and ts.tzinfo is None:
                parser.error("--start and --end need a UTC offset, e.g. 2026-10-16T09:00:00+00:00")
    return asyncio.run(_with_store(settings, args))


async def _with_store(settings: DriftSettings, args: argparse.Namespace) -> int:
    postgres = PostgresSettings()
    store = PostgresDriftStore(
        postgres.conninfo(
            connect_timeout_s=settings.db.connect_timeout_s, application_name="drift-cli"
        ),
        settings.db,
    )
    await store.open()
    try:
        if args.command == "references":
            for ref in sorted((await store.references()).values(), key=lambda r: r.camera_id):
                sys.stdout.write(
                    "\t".join(
                        (ref.camera_id, ref.start.isoformat(), ref.end.isoformat(), ref.reason)
                    )
                    + "\n"
                )
            return 0
        cameras = args.cameras or sorted(await store.references())
        for ref in await recalibrate(
            store, cameras, settings, SystemClock(), start=args.start, end=args.end
        ):
            log.info(
                "reference replaced",
                extra={
                    "camera_id": ref.camera_id,
                    "start": ref.start.isoformat(),
                    "end": ref.end.isoformat(),
                },
            )
        return 0
    finally:
        await store.aclose()


async def recalibrate(
    store: DriftStore,
    cameras: Sequence[str],
    settings: DriftSettings,
    clock: Clock,
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> list[ReferenceRange]:
    """Replace each camera's reference: [start, end) if given, else the next reference_s from
    now (frames from then on are the new calibration period)."""
    if start is None or end is None:
        start = clock.now()
        end = start + timedelta(seconds=settings.monitor.reference_s)
    if end <= start:
        raise ValueError("the reference must end after it starts")
    refs = [ReferenceRange(c, start, end, "recalibrate") for c in cameras]
    for ref in refs:
        await store.set_reference(ref)
    return refs
