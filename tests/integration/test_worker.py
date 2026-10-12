"""Worker against the compose stack: the Postgres queue's lease, claim and fencing SQL, and a
full loadgen -> ingest -> worker run with the tiny YOLO11n model on CPU.

Each test gets a throwaway database (`fresh_db`); objects are written under a unique prefix
and deleted afterwards.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import cv2
import psycopg
import pytest
from ultralytics.utils import ASSETS

from tests.integration.conftest import StackSettings, poll, service
from vms.common.clock import SystemClock
from vms.common.metrics import Metrics
from vms.loadgen.frames import LoadedClip
from vms.loadgen.recorder import Recorder
from vms.loadgen.runner import Runner
from vms.loadgen.schedule import plan_cameras
from vms.loadgen.settings import LoadgenSettings
from vms.loadgen.transport import HttpTransport
from vms.storage.frames import PoolTuning
from vms.storage.work import DetectionRow, FailOutcome, FrameResult, PostgresWorkQueue
from vms.worker.weights import ensure_weights, load_models_config

pytestmark = pytest.mark.integration

T0 = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)
TTL_S = 30.0


def dsn(stack: StackSettings, db: str) -> str:
    return f"{stack.postgres_dsn} dbname={db}"


def insert_frames(stack: StackSettings, db: str, cameras: list[str], n: int) -> None:
    with psycopg.connect(dsn(stack, db)) as conn:
        for camera in cameras:
            for seq in range(n):
                conn.execute(
                    "INSERT INTO frames (frame_id, camera_id, run_id, seq, ts, object_key, "
                    "size_bytes) VALUES (%s, %s, 'run', %s, %s, 'k', 1)",
                    (f"{camera}-{seq}", camera, seq, T0 + timedelta(seconds=seq / 5)),
                )


def fetch(stack: StackSettings, db: str, query: str, *args: Any) -> list[tuple[Any, ...]]:
    with psycopg.connect(dsn(stack, db)) as conn:
        return conn.execute(query, args).fetchall()  # type: ignore[arg-type]


def result(frame_id: str, camera: str, n_detections: int = 2) -> FrameResult:
    return FrameResult(
        frame_id=frame_id,
        camera_id=camera,
        ts=T0,
        width=64,
        height=48,
        brightness=1.0,
        contrast=1.0,
        blur_var=1.0,
        model_version="test@0",
        detections=tuple(
            DetectionRow(i, "person", 0.9, 0.0, 0.0, 1.0, 1.0) for i in range(n_detections)
        ),
    )


def with_queues(
    stack: StackSettings, db: str, n: int, body: Callable[..., Awaitable[None]]
) -> None:
    async def main() -> None:
        queues = [
            PostgresWorkQueue(dsn(stack, db), PoolTuning(min_size=0, max_size=4)) for _ in range(n)
        ]
        for q in queues:
            await q.open()
        try:
            await body(*queues)
        finally:
            for q in queues:
                await q.aclose()

    asyncio.run(main())


def test_workers_lease_disjoint_cameras_and_claim_in_capture_order(
    stack: StackSettings, fresh_db: str
) -> None:
    cameras = ["cam-a", "cam-b", "cam-c", "cam-d"]
    insert_frames(stack, fresh_db, cameras, 5)

    async def body(q1: PostgresWorkQueue, q2: PostgresWorkQueue) -> None:
        got1, got2 = await asyncio.gather(
            q1.acquire_cameras("w1", 2, TTL_S, 3), q2.acquire_cameras("w2", 2, TTL_S, 3)
        )
        # Racing workers pick the same oldest cameras; the database gives each to one of
        # them, so the loser may get fewer. It retries on its next idle step (as here).
        assert not set(got1) & set(got2)
        got1 += await q1.acquire_cameras("w1", 2 - len(got1), TTL_S, 3)
        got2 += await q2.acquire_cameras("w2", 2 - len(got2), TTL_S, 3)
        assert not set(got1) & set(got2)
        assert sorted(got1 + got2) == cameras

        seen: dict[str, list[int]] = {c: [] for c in cameras}
        owner: dict[str, str] = {}
        while True:
            claims = await asyncio.gather(q1.claim("w1", 3), q2.claim("w2", 3))
            if not any(claims):
                break
            for worker, frames in zip(["w1", "w2"], claims, strict=True):
                for f in frames:
                    assert owner.setdefault(f.camera_id, worker) == worker
                    seen[f.camera_id].append(f.seq)
                    assert f.attempts == 1
                    assert await (q1 if worker == "w1" else q2).complete(
                        worker, result(f.frame_id, f.camera_id)
                    )
        assert seen == {c: [0, 1, 2, 3, 4] for c in cameras}
        assert await q1.backlog() == 0

    with_queues(stack, fresh_db, 2, body)
    assert fetch(stack, fresh_db, "SELECT count(*) FROM detections") == [(40,)]
    assert fetch(stack, fresh_db, "SELECT DISTINCT status FROM frames") == [("done",)]


def test_concurrent_lease_attempts_have_one_winner(stack: StackSettings, fresh_db: str) -> None:
    insert_frames(stack, fresh_db, ["cam-a"], 1)

    async def body(*queues: PostgresWorkQueue) -> None:
        won = await asyncio.gather(
            *(q.acquire_cameras(f"w{i}", 1, TTL_S, 3) for i, q in enumerate(queues))
        )
        assert sorted(len(w) for w in won) == [0] * (len(queues) - 1) + [1]

    with_queues(stack, fresh_db, 8, body)


def test_takeover_after_lease_expiry_resets_frames_and_fences_the_old_worker(
    stack: StackSettings, fresh_db: str
) -> None:
    insert_frames(stack, fresh_db, ["cam-a"], 3)

    async def body(q1: PostgresWorkQueue, q2: PostgresWorkQueue) -> None:
        assert await q1.acquire_cameras("w1", 1, TTL_S, 3) == ["cam-a"]
        stalled = await q1.claim("w1", 2)
        assert [f.seq for f in stalled] == [0, 1]
        # w2 cannot take a live lease ...
        assert await q2.acquire_cameras("w2", 1, TTL_S, 3) == []
        # ... until w1 stops renewing it.
        with psycopg.connect(dsn(stack, fresh_db), autocommit=True) as conn:
            conn.execute("UPDATE camera_leases SET expires_at = now() - interval '1 second'")
        assert await q2.acquire_cameras("w2", 1, TTL_S, 3) == ["cam-a"]
        assert await q1.renew("w1", TTL_S) == []

        # w1 wakes up and tries to finish: fenced, nothing written.
        assert not await q1.complete("w1", result(stalled[0].frame_id, "cam-a"))
        assert await q1.fail("w1", stalled[1].frame_id, "x", retry=True, max_attempts=3) is (
            FailOutcome.LOST
        )
        assert await q1.claim("w1", 10) == []

        # w2 picks up all three in order; the reset ones count a second attempt.
        frames = await q2.claim("w2", 10)
        assert [(f.seq, f.attempts) for f in frames] == [(0, 2), (1, 2), (2, 1)]
        for f in frames:
            assert await q2.complete("w2", result(f.frame_id, "cam-a"))
        # Completing again (a redelivery) changes nothing.
        assert not await q2.complete("w2", result(frames[0].frame_id, "cam-a"))

    with_queues(stack, fresh_db, 2, body)
    assert fetch(stack, fresh_db, "SELECT count(*) FROM detections") == [(6,)]


def test_failures_retry_then_fail_and_poison_frames_stop_after_max_attempts(
    stack: StackSettings, fresh_db: str
) -> None:
    insert_frames(stack, fresh_db, ["cam-a"], 2)

    async def body(q1: PostgresWorkQueue, q2: PostgresWorkQueue) -> None:
        await q1.acquire_cameras("w1", 1, TTL_S, 2)
        (f0,) = await q1.claim("w1", 1)
        assert await q1.fail("w1", f0.frame_id, "fetch", retry=True, max_attempts=2) is (
            FailOutcome.RETRY
        )
        (again,) = await q1.claim("w1", 1)
        assert (again.frame_id, again.attempts) == (f0.frame_id, 2)
        assert await q1.fail("w1", f0.frame_id, "fetch", retry=True, max_attempts=2) is (
            FailOutcome.FAILED
        )

        # Frame 1 "kills" its worker twice: each takeover resets it, the last one fails it.
        for worker, q in [("w1", q1), ("w2", q2)]:
            if worker == "w2":
                assert await q.acquire_cameras(worker, 1, TTL_S, 2) == ["cam-a"]
            (f1,) = await q.claim(worker, 1)
            assert f1.seq == 1
            with psycopg.connect(dsn(stack, fresh_db), autocommit=True) as conn:
                conn.execute("UPDATE camera_leases SET expires_at = now() - interval '1 s'")
        # Nothing is pending, but the camera's dead lease holds a frame in processing: it is
        # still taken over, and the frame, out of attempts, is failed instead of retried.
        assert await q1.acquire_cameras("w3", 1, TTL_S, 2) == ["cam-a"]
        assert await q1.claim("w3", 1) == []

    with_queues(stack, fresh_db, 2, body)
    rows = fetch(stack, fresh_db, "SELECT seq, status, attempts, error FROM frames ORDER BY seq")
    assert rows[0][:3] == (0, "failed", 2)
    assert rows[0][3] == "fetch"
    assert rows[1][:3] == (1, "failed", 2)
    assert rows[1][3] == "worker lost while processing"


def test_release_hands_back_frames_and_leases(stack: StackSettings, fresh_db: str) -> None:
    insert_frames(stack, fresh_db, ["cam-a", "cam-b"], 2)

    async def body(q1: PostgresWorkQueue) -> None:
        await q1.acquire_cameras("w1", 2, TTL_S, 3)
        assert len(await q1.claim("w1", 3)) == 3
        assert await q1.backlog() == 1
        await q1.release("w1")
        assert await q1.backlog() == 4

    with_queues(stack, fresh_db, 1, body)
    assert fetch(stack, fresh_db, "SELECT count(*) FROM camera_leases") == [(0,)]
    assert fetch(stack, fresh_db, "SELECT DISTINCT attempts FROM frames") == [(0,)]


# --- End to end ---------------------------------------------------------------------------


def bus_clip() -> LoadedClip:
    image = cv2.imread(str(ASSETS / "bus.jpg"))
    assert image is not None
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    height, width = image.shape[:2]
    return LoadedClip("bus", 30.0, width, height, (bytes(buf),) * 4)


def test_loadgen_to_detections_with_yolo11n(
    stack: StackSettings, fresh_db: str, ingest_url: str, prefix: str
) -> None:
    ensure_weights(load_models_config(), "yolo11n")  # fetched once, then cached
    clip = bus_clip()
    plans = plan_cameras(2, ["bus"], {"bus": 4}, 5.0, seed=0, prefix=prefix)
    settings = LoadgenSettings(cameras=2, fps=5.0, duration_s=0.8, warmup_s=0.0)

    async def send() -> AsyncIterator[None]:  # type: ignore[misc]
        transport = HttpTransport(
            ingest_url, request_timeout_s=5.0, connect_timeout_s=2.0, max_connections=8
        )
        recorder = Recorder(Metrics("loadgen"), warmup_s=0.0)
        await Runner(
            settings, plans, {"bus": clip}, transport, recorder, clock=SystemClock(), run_id="r"
        ).run()
        await transport.aclose()

    asyncio.run(send())  # type: ignore[arg-type]
    sent = fetch(stack, fresh_db, "SELECT count(*) FROM frames")[0][0]
    assert sent == 8

    worker_env = {"VMS_DETECTOR__WEIGHTS": "yolo11n", "VMS_QUEUE__POLL_MAX_S": "0.2"}
    with service("vms.worker", stack, fresh_db, worker_env) as url:
        poll(
            lambda: fetch(
                stack, fresh_db, "SELECT count(*) = 0 FROM frames WHERE status <> 'done'"
            )[0][0],
            timeout_s=120,
        )
        metrics = poll(lambda: _get(f"{url}/metrics"), timeout_s=5)
        assert 'vms_worker_frames_total{outcome="done"} 8.0' in metrics

    frames = fetch(
        stack,
        fresh_db,
        "SELECT width, height, brightness, contrast, blur_var, model_version, attempts FROM frames",
    )
    assert {f[:2] for f in frames} == {(clip.width, clip.height)}
    assert all(f[2] > 0 and f[3] > 0 and f[4] > 0 for f in frames)
    assert {f[5] for f in frames} == {"yolo11n@0ebbc80d4a76"}
    assert {f[6] for f in frames} == {1}

    by_class = dict(
        fetch(
            stack,
            fresh_db,
            "SELECT class_name, count(*) FROM detections GROUP BY 1",
        )
    )
    # bus.jpg: one bus and four people per frame with yolo11n at conf 0.25 (upstream example).
    assert by_class.get("bus", 0) == 8
    assert by_class.get("person", 0) >= 3 * 8
    # Same image, deterministic model: every frame has identical detections.
    per_frame = fetch(
        stack,
        fresh_db,
        "SELECT count(DISTINCT n) FROM (SELECT frame_id, count(*) n FROM detections GROUP BY 1) t",
    )
    assert per_frame == [(1,)]
    # Graceful shutdown handed back the leases.
    assert fetch(stack, fresh_db, "SELECT count(*) FROM camera_leases") == [(0,)]


def _get(url: str) -> str:
    import urllib.request

    with urllib.request.urlopen(url, timeout=5) as response:
        return str(response.read().decode())
