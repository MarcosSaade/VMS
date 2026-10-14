"""Wire the worker from settings: weights, detector, queue, object store, loop, HTTP probes."""

from __future__ import annotations

import asyncio
import contextlib
import os
import socket
import time

import uvicorn
from fastapi import FastAPI

from vms.common.clock import SystemClock
from vms.common.ids import UlidGenerator, new_ulid
from vms.common.observability import Observability, init_observability
from vms.common.settings import load_settings
from vms.storage.config import ObjectStoreSettings, PostgresSettings
from vms.storage.objects import S3ObjectStore
from vms.storage.work import PostgresWorkQueue
from vms.worker.app import create_app
from vms.worker.detector import Detector, YoloDetector, build_detector
from vms.worker.onnx_detector import OnnxDetector
from vms.worker.service import Worker
from vms.worker.settings import WorkerSettings

SERVICE = "worker"


def worker_id() -> str:
    """Unique per process, readable in `frames.claimed_by` and `camera_leases`."""
    return f"{socket.gethostname()}-{os.getpid()}-{new_ulid()[-6:].lower()}"


def load_detector(settings: WorkerSettings, obs: Observability) -> YoloDetector | OnnxDetector:
    t0 = time.monotonic()
    detector = build_detector(settings.detector)
    detector.warmup()
    obs.logger.info(
        "model loaded",
        extra={
            "model_version": detector.model_version,
            "backend": settings.detector.backend,
            "device": settings.detector.device,
            "load_and_warmup_s": round(time.monotonic() - t0, 3),
        },
    )
    return detector


def build_app(settings: WorkerSettings, obs: Observability, detector: Detector) -> FastAPI:
    postgres = PostgresSettings()
    queue = PostgresWorkQueue(
        postgres.conninfo(
            connect_timeout_s=settings.db.connect_timeout_s, application_name=SERVICE
        ),
        settings.db,
    )
    store = S3ObjectStore(ObjectStoreSettings(), settings.objects)
    worker = Worker(
        queue,
        store,
        detector,
        settings.queue,
        obs.metrics,
        worker_id=worker_id(),
        clock=SystemClock(),
        ids=UlidGenerator(),
        tracker_settings=settings.tracker,
        zone_settings=settings.zones,
        tracer=obs.tracer,
    )
    stop = asyncio.Event()
    task: asyncio.Task[None] | None = None

    async def startup() -> None:
        nonlocal task
        await queue.open()
        task = asyncio.create_task(worker.run(stop), name="worker-loop")
        obs.logger.info(
            "worker started",
            extra={"worker_id": worker.worker_id, "db": postgres.db, "port": settings.http.port},
        )

    async def shutdown() -> None:
        stop.set()
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=settings.queue.shutdown_grace_s)
            except TimeoutError:
                # The frame in flight takes too long: leases expire and its frames are
                # reclaimed by another worker after lease_ttl_s.
                obs.logger.warning("worker loop did not stop in time; cancelling")
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        await queue.aclose()
        await store.aclose()
        obs.shutdown()

    async def model_ready() -> bool:
        return True  # loaded and warmed up before the app starts

    return create_app(
        obs.metrics,
        ready_checks={"postgres": queue.ready, "object_store": store.ready, "model": model_ready},
        on_startup=startup,
        on_shutdown=shutdown,
    )


def serve() -> int:
    settings = load_settings(WorkerSettings, SERVICE)
    obs = init_observability(settings)
    detector = load_detector(settings, obs)
    app = build_app(settings, obs, detector)
    uvicorn.run(
        app,
        host=settings.http.host,
        port=settings.http.port,
        timeout_graceful_shutdown=int(settings.queue.shutdown_grace_s),
        log_config=None,
        access_log=False,
    )
    return 0
