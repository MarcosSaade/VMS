"""The query service's read-only login (ARCHITECTURE 11).

Migration 0003 creates the privileges as the NOLOGIN group role `vms_query_reader`. This module
creates or updates the login that holds them, POSTGRES_READONLY_USER, with its password from the
environment (`python -m vms.storage bootstrap`, run by `make migrate`). Kept out of migrations so
no password lands in version control, and so rotating it is a re-run, not a new revision.

The login is also read-only by default (`default_transaction_read_only`), a second wall behind
the grants: even a privilege granted by mistake cannot be used to write.
"""

from __future__ import annotations

from typing import Any

import psycopg
from psycopg import sql
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

QUERY_READER_ROLE = "vms_query_reader"  # created by migration 0003


class ReadonlyUserSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="POSTGRES_READONLY_", env_file=".env", extra="ignore"
    )

    user: str = "vms_query"
    password: SecretStr = SecretStr("change-me")


def ensure_query_login(conn: psycopg.Connection[Any], user: str, password: SecretStr) -> None:
    """Create or update `user` as a read-only member of the query reader role. Idempotent."""
    owner = conn.execute("SELECT current_user").fetchone()
    if owner is not None and owner[0] == user:
        raise ValueError(f"the read-only user must not be the connecting user {user!r}")
    role = sql.Identifier(user)
    with conn.transaction():
        exists = conn.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (user,)).fetchone()
        verb = sql.SQL("ALTER") if exists else sql.SQL("CREATE")
        conn.execute(
            sql.SQL("{} ROLE {} LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE PASSWORD {}").format(
                verb, role, sql.Literal(password.get_secret_value())
            )
        )
        conn.execute(sql.SQL("ALTER ROLE {} SET default_transaction_read_only = on").format(role))
        conn.execute(sql.SQL("GRANT {} TO {}").format(sql.Identifier(QUERY_READER_ROLE), role))
