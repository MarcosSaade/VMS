"""Loadgen's HTTP transport against the stub ingest server over loopback.

Unit tests cover pacing and accounting with fake transports; this checks the real wire format:
the multipart body the transport builds is what the frame contract (vms.common.ingest_api)
expects, and status codes, timeouts and refused connections map to the right outcomes. Needs
no compose stack, only a free loopback port.
"""

from __future__ import annotations

import asyncio
import os
import random
import signal
import socket
import subprocess
import sys
import urllib.request
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from typing import Any

import aiohttp
import pytest
from aiohttp import web

from tests.integration.conftest import poll
from vms.common.clock import SystemClock
from vms.common.ingest_api import FRAMES_PATH
from vms.common.metrics import Metrics
from vms.loadgen.frames import LoadedClip
from vms.loadgen.recorder import Recorder
from vms.loadgen.runner import Runner
from vms.loadgen.schedule import plan_cameras
from vms.loadgen.settings import LoadgenSettings, StubSettings
from vms.loadgen.stub import build_app
from vms.loadgen.transport import HttpTransport, Outcome, OutgoingFrame

pytestmark = pytest.mark.integration

FRAME = OutgoingFrame("cam-001", "run-1", 0, 1_760_000_000_000, b"\xff\xd8 not really a jpeg")


@asynccontextmanager
async def stub(**settings: Any) -> AsyncIterator[tuple[str, Metrics]]:
    metrics = Metrics("loadgen_stub")
    app = build_app(StubSettings(**settings), metrics, rng=random.Random(0))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = runner.addresses[0][1]
    try:
        yield f"http://127.0.0.1:{port}", metrics
    finally:
        await runner.cleanup()


def transport(url: str, timeout_s: float = 5.0) -> HttpTransport:
    return HttpTransport(url, request_timeout_s=timeout_s, connect_timeout_s=1.0, max_connections=8)


def received(metrics: Metrics, outcome: str) -> float:
    value = metrics.registry.get_sample_value(
        "vms_loadgen_stub_received_frames_total", {"outcome": outcome}
    )
    return value or 0.0


def run(main: Callable[[], Coroutine[Any, Any, None]]) -> None:
    asyncio.run(main())


def test_frame_is_accepted_and_a_resend_is_a_duplicate() -> None:
    async def main() -> None:
        async with stub() as (url, metrics):
            t = transport(url)
            first = await t.send(FRAME)
            second = await t.send(FRAME)
            await t.aclose()
        assert (first.outcome, first.status) == (Outcome.ACCEPTED, 202)
        assert second.outcome is Outcome.ACCEPTED
        assert received(metrics, "accepted") == 1
        assert received(metrics, "duplicate") == 1

    run(main)


def test_shed_frames_are_rejected() -> None:
    async def main() -> None:
        async with stub(reject_ratio=1.0) as (url, _):
            t = transport(url)
            result = await t.send(FRAME)
            await t.aclose()
        assert (result.outcome, result.status) == (Outcome.REJECTED, 429)

    run(main)


def test_invalid_metadata_is_a_client_error() -> None:
    async def main() -> None:
        async with stub() as (url, metrics):
            form = aiohttp.FormData()
            form.add_field("camera_id", "cam-001")
            form.add_field("run_id", "r")
            form.add_field("seq", "-1")
            form.add_field("ts_unix_ms", "0")
            form.add_field("jpeg", b"x", filename="x.jpg", content_type="image/jpeg")
            async with aiohttp.ClientSession() as session:
                async with session.post(url + FRAMES_PATH, data=form) as bad_meta:
                    assert bad_meta.status == 422
                no_file = {"camera_id": "c"}
                async with session.post(url + FRAMES_PATH, data=no_file) as missing:
                    assert missing.status == 400
        assert received(metrics, "invalid") == 2

    run(main)


def test_timeouts_and_refused_connections_are_failures() -> None:
    async def main() -> None:
        async with stub(latency_ms=2000) as (url, _):
            t = transport(url, timeout_s=0.2)
            slow = await t.send(FRAME)
            await t.aclose()
        assert (slow.outcome, slow.reason) == (Outcome.FAILED, "timeout")

        with socket.socket() as s:  # a port nothing listens on
            s.bind(("127.0.0.1", 0))
            closed_port = s.getsockname()[1]
        t = transport(f"http://127.0.0.1:{closed_port}")
        refused = await t.send(FRAME)
        await t.aclose()
        assert (refused.outcome, refused.reason) == (Outcome.FAILED, "connection")

    run(main)


def test_paced_run_against_the_stub() -> None:
    clip = LoadedClip("c", 30.0, 64, 48, tuple(bytes([0xFF, 0xD8, i]) for i in range(30)))
    settings = LoadgenSettings(cameras=3, fps=10.0, duration_s=1.0, warmup_s=0.0)

    async def main() -> None:
        async with stub() as (url, metrics):
            plans = plan_cameras(3, ["c"], {"c": 30}, settings.fps, seed=0)
            recorder = Recorder(Metrics("loadgen"), warmup_s=0.0)
            t = transport(url)
            await Runner(
                settings, plans, {"c": clip}, t, recorder, clock=SystemClock(), run_id="r"
            ).run()
            await t.aclose()
        frames = recorder.summary(1.0, 30.0)["frames"]
        assert frames["accepted"] == frames["sent"] == 30
        assert received(metrics, "accepted") == 30

    run(main)


def test_sigterm_stops_every_stub_worker() -> None:
    """Regression: SIGTERM to a multi-worker stub used to orphan its workers on the port."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    env = {**os.environ, "VMS_TELEMETRY__ENABLED": "false"}
    cmd = [sys.executable, "-m", "vms.loadgen", "stub", "--workers", "2", "--port", str(port)]
    proc = subprocess.Popen(cmd, env=env, stderr=subprocess.DEVNULL)
    try:
        poll(
            lambda: urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1),
            timeout_s=20,
        )
    finally:
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=10) is not None
    with socket.socket() as s:
        # SO_REUSEADDR tolerates the health check's TIME_WAIT connection, but on Linux bind
        # still fails with EADDRINUSE while any socket is listening on the port.
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", port))
        s.listen()
