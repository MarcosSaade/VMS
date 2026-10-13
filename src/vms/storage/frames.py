"""The `frames` table: one row per ingested frame, which is also the work queue (D3 default).

Ingest inserts rows with status `pending`; the worker (task 1.5) claims them. A frame is
identified by its idempotency key `(camera_id, run_id, seq)` (D17); `frame_id` is a ULID the
ingest assigns. Inserting the same key twice returns the first row's `frame_id` and
`created=False`, so a retried upload never creates a second row.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, ConfigDict, Field


class FrameStoreError(Exception):
    """The database could not complete a request."""


class PoolTuning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    min_size: int = Field(default=1, ge=0)
    max_size: int = Field(default=10, ge=1)
    connect_timeout_s: float = Field(default=5.0, gt=0)
    # Waiting for a free connection from the pool; past this, the request fails fast.
    acquire_timeout_s: float = Field(default=2.0, gt=0)
    statement_timeout_ms: int = Field(default=2000, ge=1)


@dataclass(frozen=True)
class NewFrame:
    frame_id: str
    camera_id: str
    run_id: str
    seq: int
    ts: datetime  # capture time, UTC
    object_key: str
    size_bytes: int
    # W3C traceparent of the ingest span; the worker continues this trace (D25).
    traceparent: str | None = None


@dataclass(frozen=True)
class InsertResult:
    frame_id: str  # the stored row's id: the new one, or the original on a duplicate
    object_key: str
    created: bool


class FrameRepository(Protocol):
    async def insert(self, frame: NewFrame) -> InsertResult: ...

    async def ready(self) -> bool: ...

    async def aclose(self) -> None: ...


INSERT_SQL = """
INSERT INTO frames (frame_id, camera_id, run_id, seq, ts, object_key, size_bytes, traceparent)
VALUES (%(frame_id)s, %(camera_id)s, %(run_id)s, %(seq)s, %(ts)s, %(object_key)s, %(size_bytes)s,
        %(traceparent)s)
ON CONFLICT (camera_id, run_id, seq) DO NOTHING
RETURNING frame_id, object_key
"""

SELECT_BY_KEY_SQL = """
SELECT frame_id, object_key FROM frames
WHERE camera_id = %(camera_id)s AND run_id = %(run_id)s AND seq = %(seq)s
"""


class PostgresFrameRepository:
    def __init__(self, conninfo: str, tuning: PoolTuning) -> None:
        self._tuning = tuning
        self._pool = AsyncConnectionPool(
            conninfo,
            min_size=tuning.min_size,
            max_size=tuning.max_size,
            timeout=tuning.acquire_timeout_s,
            kwargs={"options": f"-c statement_timeout={tuning.statement_timeout_ms}"},
            open=False,
        )

    async def open(self) -> None:
        # wait=False: the service starts (and reports not ready) even if Postgres is down.
        await self._pool.open(wait=False)

    async def insert(self, frame: NewFrame) -> InsertResult:
        params = {
            "frame_id": frame.frame_id,
            "camera_id": frame.camera_id,
            "run_id": frame.run_id,
            "seq": frame.seq,
            "ts": frame.ts,
            "object_key": frame.object_key,
            "size_bytes": frame.size_bytes,
            "traceparent": frame.traceparent,
        }
        try:
            async with self._pool.connection() as conn:
                row = await (await conn.execute(INSERT_SQL, params)).fetchone()
                if row is not None:
                    return InsertResult(row[0], row[1], created=True)
                # Conflict: the key exists. The row is committed (ON CONFLICT waits for a
                # concurrent insert of the same key to finish), so this read finds it.
                existing = await (await conn.execute(SELECT_BY_KEY_SQL, params)).fetchone()
        except Exception as exc:  # psycopg and pool errors: connection, timeout, SQL
            raise FrameStoreError(f"insert failed: {type(exc).__name__}: {exc}") from exc
        if existing is None:
            raise FrameStoreError("insert conflicted but the existing row was not found")
        return InsertResult(existing[0], existing[1], created=False)

    async def ready(self) -> bool:
        try:
            async with self._pool.connection(timeout=self._tuning.acquire_timeout_s) as conn:
                await conn.execute("SELECT 1")
        except Exception:
            return False
        return True

    async def aclose(self) -> None:
        await self._pool.close()
