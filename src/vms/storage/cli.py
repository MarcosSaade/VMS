"""`python -m vms.storage bootstrap` (run by make migrate after `alembic upgrade head`).

Brings a migrated database to its configured state, idempotently:

1. the query service's read-only login (POSTGRES_READONLY_USER, vms.storage.roles);
2. cameras and zones from the catalog (configs/cameras.yaml, vms.storage.catalog).

Connects as POSTGRES_USER, the schema owner.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import psycopg

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
    args = parser.parse_args(argv)
    configure_logging("storage", args.log_level)

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
