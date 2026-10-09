"""In-memory fakes for the object store and the frames table, with injectable failures."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from vms.common.clock import FakeClock
from vms.common.ids import UlidGenerator
from vms.common.metrics import Metrics
from vms.ingest.service import FrameIngestor
from vms.ingest.settings import FrameLimits
from vms.storage.frames import FrameStoreError, InsertResult, NewFrame
from vms.storage.objects import ObjectStoreError

NOW = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 100


class FakeObjectStore:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.fail_put = False
        self.fail_delete = False
        self.is_ready = True

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        if self.fail_put:
            raise ObjectStoreError("put refused")
        self.objects[key] = data

    async def delete(self, key: str) -> None:
        if self.fail_delete:
            raise ObjectStoreError("delete refused")
        self.objects.pop(key, None)

    async def ready(self) -> bool:
        return self.is_ready

    async def aclose(self) -> None:
        return None


class FakeFrameRepository:
    """Mirrors the table's UNIQUE (camera_id, run_id, seq) and ON CONFLICT DO NOTHING."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str, int], NewFrame] = {}
        self.fail_insert = False
        self.is_ready = True

    async def insert(self, frame: NewFrame) -> InsertResult:
        if self.fail_insert:
            raise FrameStoreError("insert refused")
        key = (frame.camera_id, frame.run_id, frame.seq)
        existing = self.rows.get(key)
        if existing is not None:
            return InsertResult(existing.frame_id, existing.object_key, created=False)
        self.rows[key] = frame
        return InsertResult(frame.frame_id, frame.object_key, created=True)

    async def ready(self) -> bool:
        return self.is_ready

    async def aclose(self) -> None:
        return None


@pytest.fixture
def store() -> FakeObjectStore:
    return FakeObjectStore()


@pytest.fixture
def repo() -> FakeFrameRepository:
    return FakeFrameRepository()


@pytest.fixture
def metrics() -> Metrics:
    return Metrics("ingest")


@pytest.fixture
def ingestor(store: FakeObjectStore, repo: FakeFrameRepository, metrics: Metrics) -> FrameIngestor:
    clock = FakeClock(NOW)
    counter = iter(range(1, 1_000_000))
    ids = UlidGenerator(clock, randbytes=lambda n: next(counter).to_bytes(n, "big"))
    return FrameIngestor(
        store, repo, FrameLimits(max_jpeg_bytes=1000), metrics, clock=clock, ids=ids
    )


def count(metrics: Metrics, outcome: str) -> float:
    value = metrics.registry.get_sample_value("vms_ingest_frames_total", {"outcome": outcome})
    return value or 0.0
