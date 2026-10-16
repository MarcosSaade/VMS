"""What the drift monitor reads and writes in Postgres: done frames and their detections per
camera and capture-time range, and one reference range per camera (`drift_references`,
migration 0007).

The monitor connects as the schema owner (it writes `drift_references`), never as the query
reader. Its reads go through the `(camera_id, ts)` indexes of frames and detections; cameras
are found with a loose index scan over `frames_camera_ts` (one index probe per camera), so
neither query scans the whole table as it grows.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, Protocol

from psycopg_pool import AsyncConnectionPool

from vms.drift.signals import DetectionRow, FrameRow
from vms.storage.frames import FrameStoreError, PoolTuning

Reason = Literal["auto", "recalibrate"]


@dataclass(frozen=True)
class ReferenceRange:
    camera_id: str
    start: datetime  # capture time, inclusive
    end: datetime  # exclusive
    reason: Reason


class DriftStore(Protocol):
    async def cameras(self) -> list[str]:
        """Every camera with at least one frame."""
        ...

    async def references(self) -> dict[str, ReferenceRange]: ...

    async def create_reference(self, ref: ReferenceRange) -> ReferenceRange:
        """Insert `ref` unless the camera already has one; return the camera's reference."""
        ...

    async def set_reference(self, ref: ReferenceRange) -> None:
        """Insert or replace the camera's reference (recalibration)."""
        ...

    async def delete_reference(self, camera_id: str) -> bool: ...

    async def frames(self, cameras: list[str], start: datetime, end: datetime) -> list[FrameRow]:
        """Done frames of `cameras` captured in [start, end), in capture order."""
        ...

    async def detections(
        self, cameras: list[str], start: datetime, end: datetime, min_conf: float
    ) -> list[DetectionRow]: ...

    async def ready(self) -> bool: ...

    async def aclose(self) -> None: ...


CAMERAS_SQL = """
WITH RECURSIVE c AS (
    (SELECT camera_id FROM frames ORDER BY camera_id LIMIT 1)
    UNION ALL
    SELECT (SELECT f.camera_id FROM frames f WHERE f.camera_id > c.camera_id
            ORDER BY f.camera_id LIMIT 1)
    FROM c WHERE c.camera_id IS NOT NULL
)
SELECT camera_id FROM c WHERE camera_id IS NOT NULL
"""

REFERENCES_SQL = "SELECT camera_id, start_ts, end_ts, reason FROM drift_references"

CREATE_REFERENCE_SQL = """
WITH ins AS (
    INSERT INTO drift_references (camera_id, start_ts, end_ts, reason)
    VALUES (%(camera_id)s, %(start)s, %(end)s, %(reason)s)
    ON CONFLICT (camera_id) DO NOTHING
    RETURNING camera_id, start_ts, end_ts, reason
)
SELECT * FROM ins
UNION ALL
SELECT camera_id, start_ts, end_ts, reason FROM drift_references
WHERE camera_id = %(camera_id)s AND NOT EXISTS (SELECT 1 FROM ins)
"""

SET_REFERENCE_SQL = """
INSERT INTO drift_references (camera_id, start_ts, end_ts, reason)
VALUES (%(camera_id)s, %(start)s, %(end)s, %(reason)s)
ON CONFLICT (camera_id) DO UPDATE
SET start_ts = EXCLUDED.start_ts, end_ts = EXCLUDED.end_ts, reason = EXCLUDED.reason,
    created_at = now()
"""

DELETE_REFERENCE_SQL = "DELETE FROM drift_references WHERE camera_id = %(camera_id)s"

FRAMES_SQL = """
SELECT camera_id, frame_id, ts, brightness, contrast, blur_var, noise
FROM frames
WHERE camera_id = ANY(%(cameras)s) AND ts >= %(start)s AND ts < %(end)s AND status = 'done'
ORDER BY camera_id, ts, seq
"""

DETECTIONS_SQL = """
SELECT camera_id, frame_id, class_name, confidence
FROM detections
WHERE camera_id = ANY(%(cameras)s) AND ts >= %(start)s AND ts < %(end)s
  AND confidence >= %(min_conf)s
"""


class PostgresDriftStore:
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

    async def _rows(self, query: str, params: dict[str, Any] | None = None) -> list[Any]:
        try:
            async with self._pool.connection() as conn:
                cur = await conn.execute(query, params)
                return await cur.fetchall() if cur.description is not None else []
        except Exception as exc:  # psycopg and pool errors: connection, timeout, SQL
            raise FrameStoreError(f"{type(exc).__name__}: {exc}") from exc

    async def cameras(self) -> list[str]:
        return [r[0] for r in await self._rows(CAMERAS_SQL)]

    async def references(self) -> dict[str, ReferenceRange]:
        return {r[0]: ReferenceRange(*r) for r in await self._rows(REFERENCES_SQL)}

    async def create_reference(self, ref: ReferenceRange) -> ReferenceRange:
        rows = await self._rows(CREATE_REFERENCE_SQL, _params(ref))
        return ReferenceRange(*rows[0])

    async def set_reference(self, ref: ReferenceRange) -> None:
        await self._rows(SET_REFERENCE_SQL, _params(ref))

    async def delete_reference(self, camera_id: str) -> bool:
        try:
            async with self._pool.connection() as conn:
                cur = await conn.execute(DELETE_REFERENCE_SQL, {"camera_id": camera_id})
                return cur.rowcount > 0
        except Exception as exc:
            raise FrameStoreError(f"{type(exc).__name__}: {exc}") from exc

    async def frames(self, cameras: list[str], start: datetime, end: datetime) -> list[FrameRow]:
        rows = await self._rows(FRAMES_SQL, {"cameras": cameras, "start": start, "end": end})
        return [FrameRow(*r) for r in rows]

    async def detections(
        self, cameras: list[str], start: datetime, end: datetime, min_conf: float
    ) -> list[DetectionRow]:
        rows = await self._rows(
            DETECTIONS_SQL,
            {"cameras": cameras, "start": start, "end": end, "min_conf": min_conf},
        )
        return [DetectionRow(*r) for r in rows]

    async def ready(self) -> bool:
        try:
            await self._rows("SELECT 1")
        except FrameStoreError:
            return False
        return True

    async def aclose(self) -> None:
        await self._pool.close()


def _params(ref: ReferenceRange) -> dict[str, Any]:
    return {"camera_id": ref.camera_id, "start": ref.start, "end": ref.end, "reason": ref.reason}
