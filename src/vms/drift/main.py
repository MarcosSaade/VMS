"""Wire the drift monitor from settings: store, monitor loop, HTTP probes and metrics."""

from __future__ import annotations

import asyncio
import contextlib

import uvicorn
from fastapi import FastAPI

from vms.common.clock import SystemClock
from vms.common.observability import Observability, init_observability
from vms.common.probes import create_probe_app
from vms.common.settings import load_settings
from vms.drift.monitor import DriftMonitor
from vms.drift.settings import DriftSettings
from vms.drift.store import PostgresDriftStore
from vms.storage.config import PostgresSettings

SERVICE = "drift"


def build_app(settings: DriftSettings, obs: Observability) -> FastAPI:
    postgres = PostgresSettings()
    store = PostgresDriftStore(
        postgres.conninfo(
            connect_timeout_s=settings.db.connect_timeout_s, application_name=SERVICE
        ),
        settings.db,
    )
    monitor = DriftMonitor(
        store, settings.monitor, settings.signals, obs.metrics, clock=SystemClock()
    )
    stop = asyncio.Event()
    task: asyncio.Task[None] | None = None

    async def startup() -> None:
        nonlocal task
        await store.open()
        task = asyncio.create_task(monitor.run(stop), name="drift-monitor")
        obs.logger.info(
            "drift monitor started",
            extra={
                "db": postgres.db,
                "port": settings.http.port,
                "window_s": settings.monitor.window_s,
                "reference_s": settings.monitor.reference_s,
            },
        )

    async def shutdown() -> None:
        stop.set()
        if task is not None:
            try:
                await asyncio.wait_for(task, timeout=settings.monitor.shutdown_grace_s)
            except TimeoutError:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        await store.aclose()
        obs.shutdown()

    return create_probe_app(
        "vms drift",
        obs.metrics,
        ready_checks={"postgres": store.ready},
        on_startup=startup,
        on_shutdown=shutdown,
    )


def serve() -> int:
    settings = load_settings(DriftSettings, SERVICE)
    obs = init_observability(settings)
    uvicorn.run(
        build_app(settings, obs),
        host=settings.http.host,
        port=settings.http.port,
        timeout_graceful_shutdown=int(settings.monitor.shutdown_grace_s),
        log_config=None,
        access_log=False,
    )
    return 0
