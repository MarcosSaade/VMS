"""Database access for the query service: read-only execution and the query log.

Two connection pools with different identities, on purpose:

- `PostgresReader` logs in as POSTGRES_READONLY_USER (member of `vms_query_reader`, D19):
  SELECT on the allowlisted columns only, read-only by default. Every connection also sets
  `default_transaction_read_only` and a `statement_timeout` itself, so the guarantees hold even
  against a misconfigured login, and each query runs in its own read-only transaction. Results
  are fetched with a row cap. Only SQL that passed the validator, plus the fixed catalog and
  live-view queries in this module, run here.
- `PostgresQueryLog` logs in as the schema owner and runs exactly one statement, a
  parameterized INSERT into query_log. No SQL derived from a question ever runs on it. A failed
  log write is reported and counted, never turned into a failed answer.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Protocol

import psycopg
from psycopg import errors as pg_errors
from psycopg_pool import AsyncConnectionPool, PoolTimeout

from vms.query.catalog import CAMERAS_SQL, ZONES_SQL
from vms.storage.frames import PoolTuning


class ExecutionError(Exception):
    """The database failed to run a query."""


class ExecutionTimeoutError(ExecutionError):
    """The query hit the statement timeout and was cancelled."""


@dataclass(frozen=True)
class QueryRows:
    columns: list[str]
    rows: list[tuple[Any, ...]]
    truncated: bool  # more rows existed than the cap allowed


@dataclass(frozen=True)
class QueryLogEntry:
    query_id: str
    ts: dt.datetime
    question: str
    query_plan: dict[str, Any] | None
    compiled_sql: str | None
    latency_ms: float
    tokens: int | None
    success: bool
    error_type: str | None
    timings: dict[str, float]
    row_count: int | None
    llm: str | None
    mode: str = "structured"
    engine: str = "plan"


class Reader(Protocol):
    async def execute(self, sql: str, params: dict[str, Any], max_rows: int) -> QueryRows: ...

    async def fetch(self, sql: str, params: dict[str, Any] | None = None) -> QueryRows:
        """Fixed, trusted queries written in this codebase (catalog, live view)."""
        ...

    async def catalog_rows(self) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]: ...

    async def ready(self) -> bool: ...

    async def aclose(self) -> None: ...


class QueryLog(Protocol):
    async def write(self, entry: QueryLogEntry) -> None: ...

    async def aclose(self) -> None: ...


class PostgresReader:
    def __init__(self, conninfo: str, tuning: PoolTuning) -> None:
        self._tuning = tuning
        options = (
            f"-c statement_timeout={tuning.statement_timeout_ms} "
            "-c default_transaction_read_only=on "
            # Bound memory per sort or hash before spilling, so one query cannot hog the server.
            "-c work_mem=16MB"
        )
        self._pool = AsyncConnectionPool(
            conninfo,
            min_size=tuning.min_size,
            max_size=tuning.max_size,
            timeout=tuning.acquire_timeout_s,
            kwargs={"options": options},
            open=False,
        )

    async def open(self) -> None:
        await self._pool.open(wait=False)

    async def execute(self, sql: str, params: dict[str, Any], max_rows: int) -> QueryRows:
        try:
            async with self._pool.connection() as conn:
                await conn.set_read_only(True)
                async with conn.transaction(), conn.cursor() as cur:
                    await cur.execute(sql.encode(), params)  # bytes: psycopg's LiteralString
                    rows = await cur.fetchmany(max_rows + 1)
                    columns = [d.name for d in cur.description or []]
        except pg_errors.QueryCanceled as exc:
            raise ExecutionTimeoutError(
                f"query cancelled after {self._tuning.statement_timeout_ms} ms"
            ) from exc
        except PoolTimeout as exc:
            raise ExecutionError("database unavailable (no connection)") from exc
        except psycopg.Error as exc:
            raise ExecutionError(f"{type(exc).__name__}: {exc}".strip()) from exc
        return QueryRows(columns, rows[:max_rows], truncated=len(rows) > max_rows)

    async def fetch(self, sql: str, params: dict[str, Any] | None = None) -> QueryRows:
        return await self.execute(sql, params or {}, max_rows=10_000)

    async def catalog_rows(self) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
        cameras = await self.fetch(CAMERAS_SQL)
        zones = await self.fetch(ZONES_SQL)
        return cameras.rows, zones.rows

    async def ready(self) -> bool:
        try:
            async with self._pool.connection(timeout=self._tuning.acquire_timeout_s) as conn:
                await conn.execute("SELECT 1")
        except Exception:
            return False
        return True

    async def aclose(self) -> None:
        await self._pool.close()


INSERT_LOG_SQL = """
INSERT INTO query_log (query_id, ts, mode, engine, question, query_plan, compiled_sql,
                       latency_ms, tokens, success, error_type, timings, row_count, llm)
VALUES (%(query_id)s, %(ts)s, %(mode)s, %(engine)s, %(question)s, %(query_plan)s,
        %(compiled_sql)s, %(latency_ms)s, %(tokens)s, %(success)s, %(error_type)s,
        %(timings)s, %(row_count)s, %(llm)s)
"""


class PostgresQueryLog:
    def __init__(self, conninfo: str, tuning: PoolTuning) -> None:
        self._pool = AsyncConnectionPool(
            conninfo,
            min_size=tuning.min_size,
            max_size=tuning.max_size,
            timeout=tuning.acquire_timeout_s,
            kwargs={"options": f"-c statement_timeout={tuning.statement_timeout_ms}"},
            open=False,
        )

    async def open(self) -> None:
        await self._pool.open(wait=False)

    async def write(self, entry: QueryLogEntry) -> None:
        params = {
            "query_id": entry.query_id,
            "ts": entry.ts,
            "mode": entry.mode,
            "engine": entry.engine,
            "question": entry.question,
            "query_plan": None if entry.query_plan is None else json.dumps(entry.query_plan),
            "compiled_sql": entry.compiled_sql,
            "latency_ms": entry.latency_ms,
            "tokens": entry.tokens,
            "success": entry.success,
            "error_type": entry.error_type,
            "timings": json.dumps(entry.timings),
            "row_count": entry.row_count,
            "llm": entry.llm,
        }
        async with self._pool.connection() as conn:
            await conn.execute(INSERT_LOG_SQL, params)

    async def aclose(self) -> None:
        await self._pool.close()


def jsonable(value: Any) -> Any:
    """A database value as JSON: timestamps as ISO 8601, decimals as floats."""
    if isinstance(value, dt.datetime | dt.date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict | list | str | int | float | bool) or value is None:
        return value
    return str(value)
