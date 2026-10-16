"""Drift monitor store against Postgres: camera discovery, window reads and reference ranges;
then one monitor update over stored rows."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest

from tests.integration.conftest import StackSettings
from vms.common.clock import FakeClock
from vms.common.metrics import Metrics
from vms.drift.monitor import DriftMonitor
from vms.drift.settings import MonitorSettings
from vms.drift.signals import SignalSettings
from vms.drift.store import PostgresDriftStore, ReferenceRange
from vms.storage.frames import PoolTuning

pytestmark = pytest.mark.integration

T0 = datetime(2026, 10, 16, 9, 0, tzinfo=UTC)


def dsn(stack: StackSettings, db: str) -> str:
    return f"{stack.postgres_dsn} dbname={db}"


def seed(stack: StackSettings, db: str) -> None:
    """cam-a: 100 s of done frames at 5 FPS, two detections each (one below 0.25), brightness
    120 then 60 from 80 s; cam-b: 10 pending frames."""
    with psycopg.connect(dsn(stack, db)) as conn:
        for seq in range(500):
            ts = T0 + timedelta(seconds=seq / 5)
            conn.execute(
                "INSERT INTO frames (frame_id, camera_id, run_id, seq, ts, object_key, size_bytes,"
                " status, brightness, contrast, blur_var, noise) "
                "VALUES (%s, 'cam-a', 'r', %s, %s, 'k', 1, 'done', %s, 40, 300, 2)",
                (f"a-{seq}", seq, ts, 120.0 + (seq % 3) if seq < 400 else 60.0 + (seq % 3)),
            )
            for idx, (cls, conf) in enumerate((("person", 0.8), ("car", 0.1))):
                conn.execute(
                    "INSERT INTO detections (frame_id, det_idx, ts, camera_id, class_name,"
                    " confidence, x1, y1, x2, y2, model_version) "
                    "VALUES (%s, %s, %s, 'cam-a', %s, %s, 0, 0, 1, 1, 'm')",
                    (f"a-{seq}", idx, ts, cls, conf),
                )
        for seq in range(10):
            conn.execute(
                "INSERT INTO frames (frame_id, camera_id, run_id, seq, ts, object_key, size_bytes)"
                " VALUES (%s, 'cam-b', 'r', %s, %s, 'k', 1)",
                (f"b-{seq}", seq, T0 + timedelta(seconds=seq)),
            )


def with_store(
    stack: StackSettings, db: str, fn: Callable[[PostgresDriftStore], Awaitable[Any]]
) -> Any:
    async def go() -> Any:
        store = PostgresDriftStore(dsn(stack, db), PoolTuning(min_size=0, max_size=2))
        await store.open()
        try:
            return await fn(store)
        finally:
            await store.aclose()

    return asyncio.run(go())


def test_reads_and_references(stack: StackSettings, fresh_db: str) -> None:
    seed(stack, fresh_db)

    async def check(store: PostgresDriftStore) -> None:
        assert await store.ready()
        assert await store.cameras() == ["cam-a", "cam-b"]
        frames = await store.frames(["cam-a", "cam-b"], T0, T0 + timedelta(seconds=10))
        assert len(frames) == 50  # cam-b's frames are pending
        assert [f.ts for f in frames] == sorted(f.ts for f in frames)
        assert frames[0].noise == 2.0
        dets = await store.detections(["cam-a"], T0, T0 + timedelta(seconds=10), 0.25)
        assert len(dets) == 50
        assert {d.class_name for d in dets} == {"person"}

        first = ReferenceRange("cam-a", T0, T0 + timedelta(seconds=60), "auto")
        assert await store.create_reference(first) == first
        other = ReferenceRange("cam-a", T0, T0 + timedelta(seconds=5), "auto")
        assert await store.create_reference(other) == first  # the first one stays
        again = ReferenceRange("cam-a", T0, T0 + timedelta(seconds=30), "recalibrate")
        await store.set_reference(again)
        assert await store.references() == {"cam-a": again}
        assert await store.delete_reference("cam-a")
        assert not await store.delete_reference("cam-a")

    with_store(stack, fresh_db, check)


def test_monitor_update_over_stored_rows(stack: StackSettings, fresh_db: str) -> None:
    seed(stack, fresh_db)
    clock = FakeClock(T0 + timedelta(seconds=20))
    metrics = Metrics("drift")
    settings = MonitorSettings(window_s=15, reference_s=60, update_every_s=5, settle_s=3)

    async def run(store: PostgresDriftStore) -> None:
        monitor = DriftMonitor(store, settings, SignalSettings(), metrics, clock=clock)
        await monitor.update()  # starts cam-a's reference at its first frame in the window
        clock.set(T0 + timedelta(seconds=98))  # window [80 s, 95 s): darkened
        await monitor.update()

    with_store(stack, fresh_db, run)

    def sample(name: str, **labels: str) -> float | None:
        return metrics.registry.get_sample_value(f"vms_drift_{name}", labels)

    assert sample("reference_frames", camera_id="cam-a") == 300  # [2 s, 62 s) at 5 FPS
    assert sample("window_frames", camera_id="cam-a") == 75
    shift = sample("shift_value", camera_id="cam-a", signal="brightness")
    assert shift is not None
    assert shift < -50  # 60 levels darker; reference std about 0.8
