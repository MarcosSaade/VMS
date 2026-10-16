"""HTTP surface of a background service (FastAPI): probes and metrics only; the work runs in
a background task started by the lifespan. Used by the worker and the drift monitor.

    GET /healthz   the process is up
    GET /readyz    every ready check passes within READY_TIMEOUT_S (503 otherwise; the worker:
                   Postgres, the object store, the model; the drift monitor: Postgres)
    GET /metrics   Prometheus
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

from vms.common.metrics import Metrics
from vms.common.telemetry import http_telemetry

READY_TIMEOUT_S = 2.0

ReadyCheck = Callable[[], Awaitable[bool]]


def create_probe_app(
    title: str,
    metrics: Metrics,
    *,
    ready_checks: dict[str, ReadyCheck],
    on_startup: Callable[[], Awaitable[None]],
    on_shutdown: Callable[[], Awaitable[None]],
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await on_startup()
        try:
            yield
        finally:
            await on_shutdown()

    app = FastAPI(
        title=title,
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        telemetry=http_telemetry(),
    )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> Response:
        async def check(fn: ReadyCheck) -> bool:
            try:
                return await asyncio.wait_for(fn(), timeout=READY_TIMEOUT_S)
            except TimeoutError:
                return False

        names = list(ready_checks)
        results = await asyncio.gather(*(check(ready_checks[n]) for n in names))
        body: dict[str, Any] = {
            n: ("ok" if r else "unavailable") for n, r in zip(names, results, strict=True)
        }
        return JSONResponse(body, status_code=200 if all(results) else 503)

    @app.get("/metrics")
    async def metrics_endpoint() -> Response:
        return Response(metrics.exposition(), media_type="text/plain; version=0.0.4")

    return app
