"""Training states (task 8.2): synthetic databases the generated pairs are executed on.

Each state is a snapshot (`vms.eval.snapshot`, the eval's generator) built from its own spec
and seed into `states_dir/<name>/`, then loaded like the eval snapshot into its own database
(`vms_train_<name>`: migrated, read-only login, catalog, COPY), on the stack's Postgres. The
states manifest in git pins each one's row counts and file hashes, so a rebuild that differs
fails instead of silently changing which pairs pass.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg

from vms.common.logs import get_logger
from vms.eval import snapshot as snap
from vms.eval.cli import create_eval_db
from vms.eval.gold import load_catalog_rows
from vms.query.catalog import QueryCatalog
from vms.storage.catalog import load_catalog, sync_catalog
from vms.storage.config import PostgresSettings
from vms.storage.roles import ReadonlyUserSettings, ensure_query_login
from vms.traindata.spec import State, TrainDataSpec

log = get_logger(__name__)


@dataclass(frozen=True)
class LoadedState:
    name: str
    db: str
    now: dt.datetime
    catalog: QueryCatalog


def build_states(spec: TrainDataSpec, root: Path) -> dict[str, Any]:
    """Generate every state's files and (re)load its database; returns the manifest."""
    manifest: dict[str, Any] = {}
    for state in spec.states:
        out = root / spec.states_dir / state.name
        built = snap.build(root / state.spec, out, root)
        target = create_eval_db(PostgresSettings(), state.db)
        readonly = ReadonlyUserSettings()
        with psycopg.connect(target.conninfo(application_name="vms-traindata")) as conn:
            ensure_query_login(conn, readonly.user, readonly.password)
            sync_catalog(conn, load_catalog(root / built.catalog))
            conn.commit()
            conn.autocommit = True
            counts = snap.load_tables(conn, out)
        if counts != built.rows:
            raise SystemExit(f"{state.name}: loaded {counts}, generated {built.rows}")
        manifest[state.name] = built.model_dump(mode="json")
        log.info("state loaded", extra={"state": state.name, "db": state.db, "rows": counts})
    return manifest


def check_states(spec: TrainDataSpec, root: Path) -> None:
    """Each state's database holds the rows its manifest pins (else: make traindata-states)."""
    pinned = json.loads((root / spec.states_manifest).read_text(encoding="utf-8"))
    for state in spec.states:
        with psycopg.connect(owner_conninfo(state)) as conn:
            counts = {
                t: int(conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0])  # type: ignore[index]
                for t in snap.TABLES
            }
        if counts != pinned[state.name]["rows"]:
            raise SystemExit(f"{state.db} holds {counts}; run make traindata-states")


def owner_conninfo(state: State) -> str:
    return (
        PostgresSettings()
        .model_copy(update={"db": state.db})
        .conninfo(application_name="vms-traindata")
    )


def reader_conninfo(state: State, statement_timeout_ms: int) -> str:
    readonly = ReadonlyUserSettings()
    pg = PostgresSettings().model_copy(
        update={"db": state.db, "user": readonly.user, "password": readonly.password}
    )
    return pg.conninfo(application_name="vms-traindata") + (
        f" options='-c statement_timeout={statement_timeout_ms}'"
    )


def load_state(state: State, root: Path, classes: list[str]) -> LoadedState:
    spec = snap.load_spec(root / state.spec)
    with psycopg.connect(owner_conninfo(state)) as conn:
        catalog = load_catalog_rows(conn, classes)
    return LoadedState(state.name, state.db, spec.now, catalog)
