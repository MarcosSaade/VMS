"""Few-shot gold SQL runs on the migrated schema as the query service's read-only login.

The unit tests prove the SQL passes the validator; this proves Postgres accepts it (syntax,
types, time zone arithmetic) on the real schema, and that the read-only grants cover every
column the gold SQL reads, so the allowlist cannot drift from the migrations unnoticed.
"""

from __future__ import annotations

import psycopg
import pytest

from tests.integration.conftest import REPO_ROOT, StackSettings
from vms.common.settings import load_settings
from vms.query.fewshots import load_fewshots
from vms.query.settings import QuerySettings
from vms.query.sql_validator import SqlValidator
from vms.storage.catalog import load_catalog, sync_catalog

pytestmark = pytest.mark.integration

# config type name -> information_schema data_type
TYPE_ALIASES = {"int": "integer", "timestamptz": "timestamp with time zone"}


def test_allowlist_matches_migrated_schema(stack: StackSettings, fresh_db: str) -> None:
    settings = load_settings(QuerySettings, "query", config_dir=REPO_ROOT / "configs")
    with psycopg.connect(f"{stack.postgres_dsn} dbname={fresh_db}") as conn:
        rows = conn.execute(
            "SELECT table_name, column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'public'"
        ).fetchall()
    actual: dict[str, dict[str, str]] = {}
    for table, column, data_type in rows:
        actual.setdefault(table, {})[column] = data_type
    for table, columns in settings.sql.tables.items():
        for column, sql_type in columns.items():
            assert actual[table].get(column) == TYPE_ALIASES.get(sql_type, sql_type), (
                f"{table}.{column}"
            )


def test_fewshot_sql_runs_as_reader(stack: StackSettings, fresh_db: str, reader_dsn: str) -> None:
    settings = load_settings(QuerySettings, "query", config_dir=REPO_ROOT / "configs")
    validator = SqlValidator(settings.sql)
    examples = [e for e in load_fewshots(REPO_ROOT / settings.assets.fewshots) if e.sql]
    assert examples
    with psycopg.connect(f"{stack.postgres_dsn} dbname={fresh_db}", autocommit=True) as conn:
        sync_catalog(conn, load_catalog(REPO_ROOT / "configs/cameras.yaml"))

    with psycopg.connect(reader_dsn, autocommit=True) as conn:
        for example in examples:
            assert example.sql is not None
            checked = validator.validate(example.sql).sql
            try:
                conn.execute(checked)  # type: ignore[arg-type]
            except psycopg.Error as exc:
                pytest.fail(f"{example.id}: {exc}\n{checked}")
