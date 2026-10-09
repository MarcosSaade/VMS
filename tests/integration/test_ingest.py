"""Ingest end to end: the real service (subprocess) against the compose stack, fed by loadgen.

Each test gets a throwaway database (`fresh_db`) and a unique camera prefix, and deletes the
objects it wrote, so the dev database and bucket are left as they were.
"""

from __future__ import annotations

import asyncio
import os
import signal
import socket
import subprocess
import sys
import urllib.request
from collections.abc import Iterator
from typing import Any

import cv2
import numpy as np
import psycopg
import pytest

from tests.integration.conftest import REPO_ROOT, StackSettings, poll
from vms.common.clock import SystemClock
from vms.common.ids import new_ulid
from vms.common.metrics import Metrics
from vms.loadgen.frames import LoadedClip
from vms.loadgen.recorder import Recorder
from vms.loadgen.runner import Runner
from vms.loadgen.schedule import plan_cameras
from vms.loadgen.settings import LoadgenSettings
from vms.loadgen.transport import HttpTransport, Outcome, OutgoingFrame

pytestmark = pytest.mark.integration


def _jpeg(value: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", np.full((48, 64, 3), value, dtype=np.uint8))
    assert ok
    return bytes(buf)


CLIP = LoadedClip("c", 30.0, 64, 48, tuple(_jpeg(v) for v in range(0, 240, 8)))


@pytest.fixture
def ingest_url(stack: StackSettings, fresh_db: str) -> Iterator[str]:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {
        **os.environ,
        "POSTGRES_HOST": stack.postgres_host,
        "POSTGRES_PORT": str(stack.postgres_port),
        "POSTGRES_DB": fresh_db,
        "VMS_HTTP__PORT": str(port),
        "VMS_TELEMETRY__ENABLED": "false",
    }
    proc = subprocess.Popen(
        [sys.executable, "-m", "vms.ingest"], cwd=REPO_ROOT, env=env, stderr=subprocess.PIPE
    )
    url = f"http://127.0.0.1:{port}"
    try:
        poll(lambda: urllib.request.urlopen(f"{url}/readyz", timeout=1), timeout_s=30)
        yield url
    finally:
        proc.terminate()
        _, stderr = proc.communicate(timeout=15)
        # uvicorn re-raises SIGTERM after a graceful shutdown, so the exit status is -SIGTERM;
        # what matters is that the lifespan shutdown ran (pool and clients closed).
        log = stderr.decode()
        assert proc.returncode in (0, -signal.SIGTERM), log[-2000:]
        assert "Application shutdown complete" in log, log[-2000:]


@pytest.fixture
def prefix(s3: Any, stack: StackSettings) -> Iterator[str]:
    """A unique camera-id prefix; every object under it is deleted afterwards."""
    value = f"it{new_ulid().lower()}-"
    yield value
    keys = [
        o["Key"]
        for page in s3.get_paginator("list_objects_v2").paginate(
            Bucket=stack.s3_bucket, Prefix=f"frames/{value}"
        )
        for o in page.get("Contents", [])
    ]
    for key in keys:
        s3.delete_object(Bucket=stack.s3_bucket, Key=key)


def rows(stack: StackSettings, db: str) -> list[tuple[Any, ...]]:
    with psycopg.connect(f"{stack.postgres_dsn} dbname={db}") as conn:
        return conn.execute(
            "SELECT camera_id, run_id, seq, object_key, size_bytes, status FROM frames "
            "ORDER BY camera_id, seq"
        ).fetchall()


def object_keys(s3: Any, stack: StackSettings, prefix: str) -> set[str]:
    response = s3.list_objects_v2(Bucket=stack.s3_bucket, Prefix=f"frames/{prefix}")
    return {o["Key"] for o in response.get("Contents", [])}


def transport(url: str) -> HttpTransport:
    return HttpTransport(url, request_timeout_s=5.0, connect_timeout_s=2.0, max_connections=16)


def test_loadgen_frames_land_as_objects_and_rows(
    stack: StackSettings, fresh_db: str, ingest_url: str, prefix: str, s3: Any
) -> None:
    settings = LoadgenSettings(
        cameras=3, fps=5.0, duration_s=1.0, warmup_s=0.0, camera_prefix=prefix
    )
    plans = plan_cameras(3, ["c"], {"c": len(CLIP.frames)}, 5.0, seed=0, prefix=prefix)
    recorder = Recorder(Metrics("loadgen"), warmup_s=0.0)

    async def run() -> None:
        t = transport(ingest_url)
        await Runner(
            settings, plans, {"c": CLIP}, t, recorder, clock=SystemClock(), run_id="run-a"
        ).run()
        await t.aclose()

    asyncio.run(run())
    assert recorder.summary(1.0, 15.0)["frames"]["accepted"] == 15

    stored = rows(stack, fresh_db)
    assert len(stored) == 15
    assert {(r[0], r[2]) for r in stored} == {(p.camera_id, k) for p in plans for k in range(5)}
    assert {r[5] for r in stored} == {"pending"}
    assert object_keys(s3, stack, prefix) == {r[3] for r in stored}
    # Bytes in the store are exactly what was sent.
    for camera_id, _, _, key, size, _ in stored:
        body = s3.get_object(Bucket=stack.s3_bucket, Key=key)["Body"].read()
        assert len(body) == size
        assert body in CLIP.frames
        assert key.startswith(f"frames/{camera_id}/")


def test_resend_and_concurrent_duplicates_store_one_frame(
    stack: StackSettings, fresh_db: str, ingest_url: str, prefix: str, s3: Any
) -> None:
    frame = OutgoingFrame(f"{prefix}cam-001", "run-b", 7, 1_760_000_000_000, CLIP.frames[0])

    async def send(frame: OutgoingFrame, n: int) -> list[Any]:
        t = transport(ingest_url)
        try:
            return list(await asyncio.gather(*(t.send(frame) for _ in range(n))))
        finally:
            await t.aclose()

    first = asyncio.run(send(frame, 1))[0]
    assert (first.outcome, first.status) == (Outcome.ACCEPTED, 201)
    again = asyncio.run(send(frame, 1))[0]
    assert (again.outcome, again.status) == (Outcome.ACCEPTED, 200)

    # Ten concurrent copies of a new frame: one wins, nine get the winner's frame_id.
    racing = OutgoingFrame(f"{prefix}cam-001", "run-b", 8, 1_760_000_000_200, CLIP.frames[1])
    results = asyncio.run(send(racing, 10))
    assert all(r.outcome is Outcome.ACCEPTED for r in results)
    assert sorted(r.status for r in results) == [200] * 9 + [201]

    stored = rows(stack, fresh_db)
    assert [r[2] for r in stored] == [7, 8]
    # Losers deleted the objects they wrote: exactly one object per row, no orphans.
    assert object_keys(s3, stack, prefix) == {r[3] for r in stored}
