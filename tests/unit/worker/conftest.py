"""In-memory fakes for worker tests: a queue with the Postgres queue's claim and fencing
semantics, an object store, a detector, and an executor that runs inline."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from concurrent.futures import Executor, Future
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from numpy.typing import NDArray
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from vms.common.clock import FakeClock
from vms.common.ids import UlidGenerator
from vms.common.metrics import Metrics
from vms.common.settings import TelemetrySettings
from vms.common.telemetry import init_telemetry
from vms.storage.frames import FrameStoreError
from vms.storage.objects import ObjectStoreError
from vms.storage.work import Backlog, ClaimedFrame, FailOutcome, FrameResult, ZoneRow
from vms.worker.detector import Detection
from vms.worker.service import Worker
from vms.worker.settings import QueueSettings
from vms.worker.tracking import TrackerSettings
from vms.worker.zones import ZoneSettings

T0 = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)


def jpeg(value: int = 128, width: int = 64, height: int = 48) -> bytes:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    image[10:20, 10:30] = 255 - value  # an edge, so the Laplacian is not zero
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return bytes(buf)


@dataclass
class Row:
    frame: ClaimedFrame
    status: str = "pending"
    claimed_by: str | None = None
    attempts: int = 0
    error: str | None = None


@dataclass
class FakeQueue:
    """Single-process stand-in for PostgresWorkQueue (whose SQL is integration-tested)."""

    rows: dict[str, Row] = field(default_factory=dict)
    leases: dict[str, str] = field(default_factory=dict)  # camera -> worker
    results: dict[str, FrameResult] = field(default_factory=dict)
    fail_ops: set[str] = field(default_factory=set)  # operations that raise FrameStoreError
    zone_rows: dict[str, list[ZoneRow]] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)
    clock: FakeClock = field(default_factory=lambda: FakeClock(T0))  # the database clock

    def add(
        self,
        camera_id: str,
        seq: int,
        ts_s: float,
        key: str | None = None,
        traceparent: str | None = None,
    ) -> str:
        """A pending frame captured (and received) at T0 + ts_s."""
        frame_id = f"{camera_id}-{seq}"
        ts = T0 + timedelta(seconds=ts_s)
        self.rows[frame_id] = Row(
            ClaimedFrame(frame_id, camera_id, "run", seq, ts, ts, key or frame_id, 0, traceparent)
        )
        return frame_id

    def _op(self, name: str) -> None:
        self.calls.append(name)
        if name in self.fail_ops:
            raise FrameStoreError(f"{name} failed")

    def status(self, frame_id: str) -> str:
        return self.rows[frame_id].status

    async def acquire_cameras(
        self, worker_id: str, limit: int, ttl_s: float, max_attempts: int
    ) -> list[str]:
        self._op("acquire")
        pending = sorted(
            {r.frame.camera_id for r in self.rows.values() if r.status == "pending"}
            - set(self.leases)
        )[:limit]
        for camera in pending:
            self.leases[camera] = worker_id
        return pending

    async def renew(self, worker_id: str, ttl_s: float) -> list[str]:
        self._op("renew")
        return sorted(c for c, w in self.leases.items() if w == worker_id)

    async def release_idle(self, worker_id: str, cameras: list[str]) -> list[str]:
        self._op("release_idle")
        busy = {r.frame.camera_id for r in self.rows.values() if r.status == "pending"}
        released = [c for c in cameras if self.leases.get(c) == worker_id and c not in busy]
        for camera in released:
            del self.leases[camera]
        return released

    async def claim(self, worker_id: str, limit: int) -> list[ClaimedFrame]:
        self._op("claim")
        mine = {c for c, w in self.leases.items() if w == worker_id}
        rows = sorted(
            (r for r in self.rows.values() if r.status == "pending" and r.frame.camera_id in mine),
            key=lambda r: (r.frame.ts, r.frame.seq),
        )[:limit]
        for r in rows:
            r.status, r.claimed_by = "processing", worker_id
            r.attempts += 1
        return [replace(r.frame, attempts=r.attempts) for r in rows]

    async def complete(self, worker_id: str, result: FrameResult) -> bool:
        self._op("complete")
        row = self.rows[result.frame_id]
        if row.claimed_by != worker_id or row.status != "processing":
            return False
        row.status, row.claimed_by = "done", None
        self.results[result.frame_id] = result
        return True

    async def fail(
        self, worker_id: str, frame_id: str, error: str, *, retry: bool, max_attempts: int
    ) -> FailOutcome:
        self._op("fail")
        row = self.rows[frame_id]
        if row.claimed_by != worker_id or row.status != "processing":
            return FailOutcome.LOST
        row.claimed_by, row.error = None, error
        if retry and row.attempts < max_attempts:
            row.status = "pending"
            return FailOutcome.RETRY
        row.status = "failed"
        return FailOutcome.FAILED

    async def release(self, worker_id: str) -> None:
        self._op("release")
        await self.release_frames(worker_id, graceful=True)
        self.leases = {c: w for c, w in self.leases.items() if w != worker_id}

    async def release_frames(self, worker_id: str, graceful: bool = False) -> None:
        if not graceful:
            self._op("release_frames")
        for r in self.rows.values():
            if r.claimed_by == worker_id and r.status == "processing":
                r.status, r.claimed_by = "pending", None
                if graceful:
                    r.attempts = max(r.attempts - 1, 0)

    async def backlog(self) -> Backlog:
        self._op("backlog")
        pending = [r for r in self.rows.values() if r.status == "pending"]
        oldest = min((r.frame.received_at for r in pending), default=None)
        age = 0.0 if oldest is None else (self.clock.now() - oldest).total_seconds()
        return Backlog(len(pending), age)

    async def zones(self, camera_id: str) -> list[ZoneRow]:
        self._op("zones")
        return list(self.zone_rows.get(camera_id, []))

    async def ready(self) -> bool:
        return True

    async def aclose(self) -> None:
        return None


@dataclass
class FakeFetcher:
    objects: dict[str, bytes] = field(default_factory=dict)
    fail_keys: set[str] = field(default_factory=set)

    async def get(self, key: str) -> bytes:
        if key in self.fail_keys:
            raise ObjectStoreError(f"get {key} failed")
        return self.objects.get(key, jpeg())


@dataclass
class FakeDetector:
    detections: list[Detection] = field(
        default_factory=lambda: [
            Detection("person", 0.9, 1.0, 2.0, 11.0, 22.0),
            Detection("car", 0.5, 5.0, 5.0, 25.0, 15.0),
        ]
    )
    error: Exception | None = None
    seen: list[tuple[int, ...]] = field(default_factory=list)
    on_detect: Callable[[], None] | None = None
    script: list[list[Detection]] = field(default_factory=list)  # per call, then `detections`
    batches: list[int] = field(default_factory=list)  # images per detect_batch call

    @property
    def model_version(self) -> str:
        return "fake@000000000000"

    def detect(self, image: NDArray[np.uint8]) -> list[Detection]:
        self.seen.append(image.shape)
        if self.on_detect is not None:
            self.on_detect()
        if self.error is not None:
            raise self.error
        if self.script:
            return list(self.script.pop(0))
        return list(self.detections)

    def detect_batch(self, images: list[NDArray[np.uint8]]) -> list[list[Detection]]:
        self.batches.append(len(images))
        return [self.detect(image) for image in images]


class InlineExecutor(Executor):
    """Runs submitted work immediately on the calling thread: deterministic tests."""

    def submit(self, fn: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Future[Any]:
        future: Future[Any] = Future()
        try:
            future.set_result(fn(*args, **kwargs))
        except BaseException as exc:
            future.set_exception(exc)
        return future


@dataclass
class Harness:
    worker: Worker
    queue: FakeQueue
    fetcher: FakeFetcher
    detector: FakeDetector
    clock: FakeClock
    metrics: Metrics
    spans: InMemorySpanExporter

    def sample(self, name: str, **labels: str) -> float:
        value = self.metrics.registry.get_sample_value(name, labels or None)
        return 0.0 if value is None else value


@pytest.fixture
def harness() -> Harness:
    return make_harness(QueueSettings(max_cameras=2, claim_batch=2, max_attempts=2))


def make_harness(settings: QueueSettings) -> Harness:
    clock = FakeClock(T0 + timedelta(seconds=60))
    queue, fetcher, detector = FakeQueue(clock=clock), FakeFetcher(), FakeDetector()
    metrics = Metrics("worker")
    spans = InMemorySpanExporter()
    tracer = init_telemetry(
        "worker", TelemetrySettings(), exporter=spans, set_global=False
    ).get_tracer("test")
    counter = iter(range(1, 1_000_000))
    ids = UlidGenerator(clock, randbytes=lambda n: next(counter).to_bytes(n, "big"))
    worker = Worker(
        queue,
        fetcher,
        detector,
        settings,
        metrics,
        worker_id="w1",
        clock=clock,
        ids=ids,
        tracker_settings=TrackerSettings(fps=5.0, buffer_s=1.0),
        zone_settings=ZoneSettings(min_frames=2, loiter_s=2.0, refresh_s=60.0),
        executor=InlineExecutor(),
        tracer=tracer,
    )
    return Harness(worker, queue, fetcher, detector, clock, metrics, spans)


@pytest.fixture
def h(harness: Harness) -> Harness:
    """Short alias: tests read `h.queue`, `h.worker`, ..."""
    return harness


@pytest.fixture(scope="session")
def random_yolo(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, str]:
    """A randomly initialized YOLO11n (seed 0) saved as weights and exported to ONNX."""
    import importlib

    import torch

    from vms.worker.onnx_detector import export_onnx

    torch.manual_seed(0)
    yolo: Any = importlib.import_module("ultralytics").YOLO
    root = tmp_path_factory.mktemp("yolo")
    weights = root / "rand.pt"
    yolo("yolo11n.yaml", task="detect").save(str(weights))
    sha = hashlib.sha256(weights.read_bytes()).hexdigest()
    model = export_onnx(weights, sha, "rand", root / "onnx", 640)
    return weights, model, sha
