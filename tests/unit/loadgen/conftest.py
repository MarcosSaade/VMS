"""Fakes for loadgen tests: a virtual-time event loop, scripted transports and tiny clips."""

from __future__ import annotations

import asyncio
import selectors
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from vms.common.metrics import Metrics
from vms.datasets.clips import FRAMES_DIR, MANIFEST_NAME, ClipManifest, clip_dir
from vms.datasets.spec import Scene
from vms.loadgen.frames import LoadedClip
from vms.loadgen.recorder import Recorder
from vms.loadgen.transport import Outcome, OutgoingFrame, SendResult


class _VirtualSelector(selectors.DefaultSelector):  # type: ignore[misc,valid-type]
    """Never blocks: when the loop would wait for its next timer, jump virtual time instead."""

    loop: VirtualTimeLoop

    def select(self, timeout: float | None = None) -> list[Any]:
        events = super().select(0)
        if events or timeout == 0:
            return events
        if timeout is None:
            raise RuntimeError("deadlock: no timers pending and no I/O ready")
        self.loop.advance(timeout)
        return []


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    """An event loop whose clock (`loop.time()`, used by asyncio.sleep and wait_for) is virtual.

    Coroutines run exactly as on a real loop, but time only passes when every task is waiting
    on a timer, and then it jumps straight to the next timer. Pacing tests are exact and
    instant, with no real sleeps.
    """

    def __init__(self) -> None:
        selector = _VirtualSelector()
        super().__init__(selector)
        selector.loop = self
        self._now = 0.0

    def time(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += seconds


@pytest.fixture
def vrun() -> Callable[[Coroutine[Any, Any, Any]], Any]:
    def run(coro: Coroutine[Any, Any, Any]) -> Any:
        loop = VirtualTimeLoop()
        try:
            return loop.run_until_complete(coro)
        finally:
            loop.close()

    return run


class ScriptedTransport:
    """Takes `latency_s` of virtual time per send; answers from `script` (then accepts)."""

    def __init__(
        self,
        latency_s: float = 0.0,
        script: list[SendResult] | None = None,
        respond: Callable[[OutgoingFrame], SendResult] | None = None,
    ) -> None:
        self.latency_s = latency_s
        self.script = list(script or [])
        self.respond = respond
        self.frames: list[OutgoingFrame] = []
        self.send_times: list[float] = []
        self.closed = False

    async def send(self, frame: OutgoingFrame) -> SendResult:
        self.frames.append(frame)
        self.send_times.append(asyncio.get_running_loop().time())
        if self.latency_s:
            await asyncio.sleep(self.latency_s)
        else:
            await asyncio.sleep(0)
        if self.script:
            return self.script.pop(0)
        if self.respond is not None:
            return self.respond(frame)
        return SendResult(Outcome.ACCEPTED, 202, "202")

    async def aclose(self) -> None:
        self.closed = True


def fake_clip(clip_id: str = "clip-a", n: int = 12, fps: float = 30.0) -> LoadedClip:
    """Frame i is the bytes b"<clip_id>:<i>", so tests can tell which frame was sent."""
    return LoadedClip(
        id=clip_id,
        fps=fps,
        width=64,
        height=48,
        frames=tuple(f"{clip_id}:{i}".encode() for i in range(n)),
    )


@pytest.fixture
def recorder() -> Recorder:
    return Recorder(Metrics("loadgen"), warmup_s=0.0)


def write_clip(data_dir: Path, clip_id: str, n: int, width: int, height: int) -> list[bytes]:
    """A fetched clip on disk (frames + clip.json) as vms.datasets would leave it."""
    frames_dir = clip_dir(data_dir, clip_id) / FRAMES_DIR
    frames_dir.mkdir(parents=True)
    blobs = []
    for i in range(n):
        image = np.full((height, width, 3), (i * 30) % 256, dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        assert ok
        blobs.append(bytes(buf))
        (frames_dir / f"{i + 1:06d}.jpg").write_bytes(bytes(buf))
    manifest = ClipManifest(
        id=clip_id,
        title="test",
        license="test",
        attribution="test",
        page_url="https://example.test",
        source_url="https://example.test/x",
        scene=Scene(
            description="test",
            location="outdoor",
            lighting="day",
            viewpoint="eye_level",
            density="sparse",
            objects=["person"],
        ),
        fps=30.0,
        frames=n,
        duration_s=n / 30.0,
        width=width,
        height=height,
        mean_luma=100.0,
        spec_hash="sha256:0",
        content_sha256="sha256:0",
    )
    (clip_dir(data_dir, clip_id) / MANIFEST_NAME).write_text(manifest.model_dump_json())
    return blobs
