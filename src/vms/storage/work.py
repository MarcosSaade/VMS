"""The frame work queue on Postgres: camera leases, claims and fenced completion (D3, D20).

Life of a frame for the worker:

    pending --claim--> processing --complete--> done
                           |--fail (retryable, attempts left)--> pending
                           |--fail (permanent, or out of attempts)--> failed
                           |--worker lost (lease taken over) / release--> pending

Ordering. Tracking needs one camera's frames processed in capture order by one worker, since
the tracker's state lives in that worker's memory. So a worker first leases cameras
(`camera_leases`, renewed every loop, expiring after `ttl_s`), then claims only frames of
cameras it leases, oldest first. `SKIP LOCKED` keeps two claim queries from blocking on, or
both taking, the same row; the lease keeps the camera itself with one worker.

Fencing. A claim stamps `claimed_by`. `complete` and `fail` only act while the stamp is still
this worker's, inside the same transaction that writes detections. A worker that stalled past
its lease (its frames were reset and re-claimed by another) therefore writes nothing: its
`complete` returns False. Detections are keyed `(frame_id, det_idx)` with ON CONFLICT DO
NOTHING as a second guard against doubling.

Workers that acquire at the same instant pick the same oldest cameras; the conflict check
gives each camera to exactly one of them, so a loser can come away with fewer than it asked
for. It retries on its next idle step, so cameras spread out within a poll interval.

Time comes from the database clock (`now()`), the one clock every worker shares.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Protocol

from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

from vms.storage.frames import FrameStoreError, PoolTuning


@dataclass(frozen=True)
class ClaimedFrame:
    frame_id: str
    camera_id: str
    run_id: str
    seq: int
    ts: datetime  # capture time
    received_at: datetime
    object_key: str
    attempts: int  # including this one


@dataclass(frozen=True)
class DetectionRow:
    det_idx: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    track_id: str | None = None  # set by the tracker (task 1.6)


@dataclass(frozen=True)
class TrackUpdate:
    """One frame's sighting of a track: creates the track or extends it."""

    track_id: str
    class_name: str


@dataclass(frozen=True)
class EventRow:
    event_id: str
    ts: datetime
    track_id: str
    zone_id: str
    event_type: str  # zone_enter | zone_exit | loiter


@dataclass(frozen=True)
class ZoneRow:
    zone_id: str
    name: str
    polygon: tuple[tuple[float, float], ...]  # normalized frame coordinates


@dataclass(frozen=True)
class FrameResult:
    frame_id: str
    camera_id: str
    ts: datetime
    width: int
    height: int
    brightness: float
    contrast: float
    blur_var: float
    model_version: str
    detections: tuple[DetectionRow, ...]
    tracks: tuple[TrackUpdate, ...] = ()
    events: tuple[EventRow, ...] = ()


class FailOutcome(StrEnum):
    RETRY = "retry"  # back to pending
    FAILED = "failed"  # terminal
    LOST = "lost"  # the claim was no longer ours; nothing changed


class WorkQueue(Protocol):
    async def acquire_cameras(
        self, worker_id: str, limit: int, ttl_s: float, max_attempts: int
    ) -> list[str]:
        """Lease up to `limit` unleased (or expired) cameras that have pending frames or
        frames left in processing by a dead worker; reset the latter; return the cameras."""
        ...

    async def renew(self, worker_id: str, ttl_s: float) -> list[str]:
        """Extend this worker's live leases; return the cameras it still holds."""
        ...

    async def claim(self, worker_id: str, limit: int) -> list[ClaimedFrame]:
        """Claim up to `limit` pending frames of leased cameras, in capture order."""
        ...

    async def complete(self, worker_id: str, result: FrameResult) -> bool:
        """Store detections and mark the frame done; False if the claim was lost."""
        ...

    async def fail(
        self, worker_id: str, frame_id: str, error: str, *, retry: bool, max_attempts: int
    ) -> FailOutcome: ...

    async def release(self, worker_id: str) -> None:
        """Give back every lease and in-flight frame (graceful shutdown)."""
        ...

    async def release_frames(self, worker_id: str) -> None:
        """Put this worker's in-flight frames back to pending, keeping their attempt counts
        (recovery after a failure could not be recorded); leases are kept."""
        ...

    async def backlog(self) -> int:
        """Pending frames across all cameras."""
        ...

    async def zones(self, camera_id: str) -> list[ZoneRow]:
        """The camera's zones from the catalog (empty for uncatalogued cameras)."""
        ...

    async def ready(self) -> bool: ...

    async def aclose(self) -> None: ...


ACQUIRE_SQL = """
WITH candidates AS (
    SELECT f.camera_id, min(f.ts) AS oldest
    FROM frames f
    LEFT JOIN camera_leases l ON l.camera_id = f.camera_id
    -- processing too: a dead worker's last frames must be recovered even if nothing new
    -- arrives for that camera.
    WHERE f.status IN ('pending', 'processing')
      AND (l.camera_id IS NULL OR l.expires_at < now())
    GROUP BY f.camera_id
    ORDER BY oldest
    LIMIT %(limit)s
)
INSERT INTO camera_leases (camera_id, worker_id, expires_at)
SELECT camera_id, %(worker_id)s, now() + make_interval(secs => %(ttl_s)s) FROM candidates
ON CONFLICT (camera_id) DO UPDATE
    SET worker_id = EXCLUDED.worker_id, acquired_at = now(), expires_at = EXCLUDED.expires_at
    -- Re-checked against the latest row version: a lease another worker just took is skipped.
    WHERE camera_leases.expires_at < now()
RETURNING camera_id
"""

# Frames a previous holder left in processing: it died or stalled, so it will never finish
# them (and cannot: completion is fenced on claimed_by). Out of attempts means a frame that
# keeps killing workers; stop retrying it.
RESET_ORPHANED_SQL = """
UPDATE frames
SET status = CASE WHEN attempts >= %(max_attempts)s THEN 'failed' ELSE 'pending' END,
    error = 'worker lost while processing',
    processed_at = CASE WHEN attempts >= %(max_attempts)s THEN now() END,
    claimed_by = NULL, claimed_at = NULL
WHERE camera_id = ANY(%(cameras)s) AND status = 'processing'
"""

RENEW_SQL = """
UPDATE camera_leases SET expires_at = now() + make_interval(secs => %(ttl_s)s)
WHERE worker_id = %(worker_id)s AND expires_at >= now()
RETURNING camera_id
"""

CLAIM_SQL = """
UPDATE frames f
SET status = 'processing', claimed_by = %(worker_id)s, claimed_at = now(),
    attempts = f.attempts + 1
WHERE f.frame_id IN (
    SELECT p.frame_id FROM frames p
    JOIN camera_leases l ON l.camera_id = p.camera_id
    WHERE p.status = 'pending' AND l.worker_id = %(worker_id)s AND l.expires_at >= now()
    ORDER BY p.ts, p.seq
    LIMIT %(limit)s
    FOR UPDATE OF p SKIP LOCKED
)
RETURNING f.frame_id, f.camera_id, f.run_id, f.seq, f.ts, f.received_at, f.object_key,
          f.attempts
"""

COMPLETE_FRAME_SQL = """
UPDATE frames
SET status = 'done', processed_at = now(), claimed_by = NULL, error = NULL,
    width = %(width)s, height = %(height)s, brightness = %(brightness)s,
    contrast = %(contrast)s, blur_var = %(blur_var)s, model_version = %(model_version)s
WHERE frame_id = %(frame_id)s AND claimed_by = %(worker_id)s AND status = 'processing'
RETURNING frame_id
"""

INSERT_DETECTION_SQL = """
INSERT INTO detections (frame_id, det_idx, ts, camera_id, track_id, class_name, confidence,
                        x1, y1, x2, y2, model_version)
VALUES (%(frame_id)s, %(det_idx)s, %(ts)s, %(camera_id)s, %(track_id)s, %(class_name)s,
        %(confidence)s, %(x1)s, %(y1)s, %(x2)s, %(y2)s, %(model_version)s)
ON CONFLICT (frame_id, det_idx) DO NOTHING
"""

# Tracks are the one updated table: a sighting extends last_seen and counts the frame. Safe
# against doubling because it runs only inside a winning, fenced completion.
UPSERT_TRACK_SQL = """
INSERT INTO tracks (track_id, camera_id, class_name, first_seen, last_seen, n_frames)
VALUES (%(track_id)s, %(camera_id)s, %(class_name)s, %(ts)s, %(ts)s, 1)
ON CONFLICT (track_id) DO UPDATE
SET first_seen = least(tracks.first_seen, EXCLUDED.first_seen),
    last_seen = greatest(tracks.last_seen, EXCLUDED.last_seen),
    n_frames = tracks.n_frames + 1
"""

INSERT_EVENT_SQL = """
INSERT INTO events (event_id, ts, camera_id, track_id, zone_id, event_type)
VALUES (%(event_id)s, %(ts)s, %(camera_id)s, %(track_id)s, %(zone_id)s, %(event_type)s)
ON CONFLICT (event_id) DO NOTHING
"""

ZONES_SQL = "SELECT zone_id, name, polygon FROM zones WHERE camera_id = %(camera_id)s ORDER BY 1"

FAIL_SQL = """
UPDATE frames
SET status = CASE WHEN %(retry)s AND attempts < %(max_attempts)s THEN 'pending'
                  ELSE 'failed' END,
    processed_at = CASE WHEN %(retry)s AND attempts < %(max_attempts)s THEN NULL
                        ELSE now() END,
    error = %(error)s, claimed_by = NULL, claimed_at = NULL
WHERE frame_id = %(frame_id)s AND claimed_by = %(worker_id)s AND status = 'processing'
RETURNING status
"""

# A graceful release did not really attempt the frames it hands back.
RELEASE_FRAMES_SQL = """
UPDATE frames
SET status = 'pending', claimed_by = NULL, claimed_at = NULL,
    attempts = greatest(attempts - 1, 0)
WHERE claimed_by = %(worker_id)s AND status = 'processing'
"""

REQUEUE_FRAMES_SQL = """
UPDATE frames SET status = 'pending', claimed_by = NULL, claimed_at = NULL
WHERE claimed_by = %(worker_id)s AND status = 'processing'
"""

RELEASE_LEASES_SQL = "DELETE FROM camera_leases WHERE worker_id = %(worker_id)s"

BACKLOG_SQL = "SELECT count(*) FROM frames WHERE status = 'pending'"

MAX_ERROR_CHARS = 500


class PostgresWorkQueue:
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
        await self._pool.open(wait=False)

    async def _rows(self, query: str, params: dict[str, Any]) -> list[tuple[Any, ...]]:
        try:
            async with self._pool.connection() as conn:
                return await (await conn.execute(query, params)).fetchall()
        except Exception as exc:  # psycopg and pool errors: connection, timeout, SQL
            raise FrameStoreError(f"{type(exc).__name__}: {exc}") from exc

    async def _rows_none(self, query: str, params: dict[str, Any]) -> None:
        try:
            async with self._pool.connection() as conn:
                await conn.execute(query, params)
        except Exception as exc:
            raise FrameStoreError(f"{type(exc).__name__}: {exc}") from exc

    async def acquire_cameras(
        self, worker_id: str, limit: int, ttl_s: float, max_attempts: int
    ) -> list[str]:
        if limit <= 0:
            return []
        try:
            async with self._pool.connection() as conn, conn.transaction():
                rows = await (
                    await conn.execute(
                        ACQUIRE_SQL, {"worker_id": worker_id, "limit": limit, "ttl_s": ttl_s}
                    )
                ).fetchall()
                cameras = [r[0] for r in rows]
                if cameras:
                    await conn.execute(
                        RESET_ORPHANED_SQL, {"cameras": cameras, "max_attempts": max_attempts}
                    )
        except Exception as exc:
            raise FrameStoreError(f"{type(exc).__name__}: {exc}") from exc
        return cameras

    async def renew(self, worker_id: str, ttl_s: float) -> list[str]:
        rows = await self._rows(RENEW_SQL, {"worker_id": worker_id, "ttl_s": ttl_s})
        return [r[0] for r in rows]

    async def claim(self, worker_id: str, limit: int) -> list[ClaimedFrame]:
        rows = await self._rows(CLAIM_SQL, {"worker_id": worker_id, "limit": limit})
        frames = [ClaimedFrame(*r) for r in rows]
        # RETURNING has no order; the claim took the oldest.
        return sorted(frames, key=lambda f: (f.ts, f.seq))

    async def complete(self, worker_id: str, result: FrameResult) -> bool:
        frame = {
            "frame_id": result.frame_id,
            "worker_id": worker_id,
            "width": result.width,
            "height": result.height,
            "brightness": result.brightness,
            "contrast": result.contrast,
            "blur_var": result.blur_var,
            "model_version": result.model_version,
        }
        try:
            async with self._pool.connection() as conn, conn.transaction():
                won = await (await conn.execute(COMPLETE_FRAME_SQL, frame)).fetchone()
                if won is None:
                    return False  # lost the claim; the transaction writes nothing
                await self._insert_detections(conn, result)
                await self._write_tracks_and_events(conn, result)
        except Exception as exc:
            raise FrameStoreError(f"{type(exc).__name__}: {exc}") from exc
        return True

    @staticmethod
    async def _insert_detections(conn: AsyncConnection[Any], result: FrameResult) -> None:
        if not result.detections:
            return
        rows = [
            {
                "frame_id": result.frame_id,
                "ts": result.ts,
                "camera_id": result.camera_id,
                "model_version": result.model_version,
                "det_idx": d.det_idx,
                "track_id": d.track_id,
                "class_name": d.class_name,
                "confidence": d.confidence,
                "x1": d.x1,
                "y1": d.y1,
                "x2": d.x2,
                "y2": d.y2,
            }
            for d in result.detections
        ]
        async with conn.cursor() as cur:
            await cur.executemany(INSERT_DETECTION_SQL, rows)

    @staticmethod
    async def _write_tracks_and_events(conn: AsyncConnection[Any], result: FrameResult) -> None:
        async with conn.cursor() as cur:
            if result.tracks:
                await cur.executemany(
                    UPSERT_TRACK_SQL,
                    [
                        {
                            "track_id": t.track_id,
                            "camera_id": result.camera_id,
                            "class_name": t.class_name,
                            "ts": result.ts,
                        }
                        # A track appears once per frame; dedupe so n_frames counts frames.
                        for t in {t.track_id: t for t in result.tracks}.values()
                    ],
                )
            if result.events:
                await cur.executemany(
                    INSERT_EVENT_SQL,
                    [
                        {
                            "event_id": e.event_id,
                            "ts": e.ts,
                            "camera_id": result.camera_id,
                            "track_id": e.track_id,
                            "zone_id": e.zone_id,
                            "event_type": e.event_type,
                        }
                        for e in result.events
                    ],
                )

    async def zones(self, camera_id: str) -> list[ZoneRow]:
        rows = await self._rows(ZONES_SQL, {"camera_id": camera_id})
        return [ZoneRow(r[0], r[1], tuple((float(x), float(y)) for x, y in r[2])) for r in rows]

    async def fail(
        self, worker_id: str, frame_id: str, error: str, *, retry: bool, max_attempts: int
    ) -> FailOutcome:
        rows = await self._rows(
            FAIL_SQL,
            {
                "worker_id": worker_id,
                "frame_id": frame_id,
                "error": error[:MAX_ERROR_CHARS],
                "retry": retry,
                "max_attempts": max_attempts,
            },
        )
        if not rows:
            return FailOutcome.LOST
        return FailOutcome.RETRY if rows[0][0] == "pending" else FailOutcome.FAILED

    async def release(self, worker_id: str) -> None:
        try:
            async with self._pool.connection() as conn, conn.transaction():
                await conn.execute(RELEASE_FRAMES_SQL, {"worker_id": worker_id})
                await conn.execute(RELEASE_LEASES_SQL, {"worker_id": worker_id})
        except Exception as exc:
            raise FrameStoreError(f"{type(exc).__name__}: {exc}") from exc

    async def release_frames(self, worker_id: str) -> None:
        await self._rows_none(REQUEUE_FRAMES_SQL, {"worker_id": worker_id})

    async def backlog(self) -> int:
        rows = await self._rows(BACKLOG_SQL, {})
        return int(rows[0][0])

    async def ready(self) -> bool:
        try:
            async with self._pool.connection(timeout=self._tuning.acquire_timeout_s) as conn:
                await conn.execute("SELECT 1")
        except Exception:
            return False
        return True

    async def aclose(self) -> None:
        await self._pool.close()
