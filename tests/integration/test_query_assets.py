"""Few-shot gold SQL is valid PostgreSQL against the configured schema.

The unit tests prove the SQL passes the validator; this proves Postgres accepts it (syntax,
types, time zone arithmetic). The tables are created as temporary tables from the validator's
allowlist inside a transaction that is rolled back, so the database is left untouched. Once
migrations exist (task 1.4), this also guards the allowlist against drifting from them.
"""

from __future__ import annotations

from pathlib import Path

import psycopg
import pytest
from psycopg import sql

from tests.integration.conftest import StackSettings
from vms.common.settings import load_settings
from vms.query.fewshots import load_fewshots
from vms.query.settings import QuerySettings
from vms.query.sql_validator import SqlValidator

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_fewshot_sql_plans_on_postgres(stack: StackSettings) -> None:
    settings = load_settings(QuerySettings, "query", config_dir=REPO_ROOT / "configs")
    validator = SqlValidator(settings.sql)
    examples = [e for e in load_fewshots(REPO_ROOT / settings.assets.fewshots) if e.sql]

    with psycopg.connect(f"{stack.postgres_dsn} dbname={stack.postgres_db}") as conn:
        try:
            for table, columns in settings.sql.tables.items():
                cols = sql.SQL(", ").join(
                    sql.SQL("{} {}").format(sql.Identifier(name), sql.SQL(sql_type))
                    for name, sql_type in columns.items()
                )
                conn.execute(
                    sql.SQL("CREATE TEMP TABLE {} ({})").format(sql.Identifier(table), cols)
                )
            for example in examples:
                assert example.sql is not None
                checked = validator.validate(example.sql).sql
                try:
                    conn.execute(sql.SQL("EXPLAIN ") + sql.SQL(checked))  # type: ignore[arg-type]
                except psycopg.Error as exc:
                    pytest.fail(f"{example.id}: {exc}\n{checked}")
        finally:
            conn.rollback()
