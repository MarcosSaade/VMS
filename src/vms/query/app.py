"""HTTP layer of the query service (FastAPI). The pipeline is in vms.query.engine.

    POST /v1/query                  {"question": "..."} -> QueryResponse (JSON; no SSE in v0)
    GET  /v1/catalog                cameras, zones and classes the planner knows
    GET  /v1/live                   per camera: latest processed frame, detections, zones
    GET  /v1/examples               example questions (the canned planner answers these)
    GET  /v1/frames/{id}/image      a stored frame's JPEG, read from the object store
    GET  /healthz                   the process is up
    GET  /readyz                    Postgres (read-only login) and the object store answer
    GET  /metrics                   Prometheus

With a built UI (`frontend/dist`, `make ui-build` or the app image), it is served at `/`, so
the browser talks to one origin and needs no CORS or proxy.

Every query response has the same shape whatever happened, so a client renders answers,
abstentions and errors from one type; the HTTP status says which (vms.query.engine).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path as FsPath
from typing import Any, Protocol

from fastapi import FastAPI, Path, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from vms.common.metrics import Metrics
from vms.common.telemetry import http_telemetry, remote_parent
from vms.query.engine import CatalogCache, StructuredQueryEngine
from vms.query.live import live_view
from vms.query.store import ExecutionError, Reader

READY_TIMEOUT_S = 2.0
# An edge at the 1.5 s query SLO (ARCHITECTURE 8), so the share within it is read, not guessed.
REQUEST_BUCKETS_S = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0, 30.0)
ULID_PATTERN = r"^[0-9A-HJKMNP-TV-Z]{26}$"
FRAME_KEY_SQL = "SELECT object_key FROM frames WHERE frame_id = %(frame_id)s"

ReadyCheck = Callable[[], Awaitable[bool]]


class FrameSource(Protocol):
    async def get(self, key: str) -> bytes: ...


class QueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=1000)

    @field_validator("question")
    @classmethod
    def _strip(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("question is empty")
        return value


def create_app(
    engine: StructuredQueryEngine,
    catalog: CatalogCache,
    reader: Reader,
    frames: FrameSource,
    metrics: Metrics,
    *,
    ready_checks: dict[str, ReadyCheck],
    examples: list[str] | None = None,
    ui_dir: FsPath | None = None,
    on_startup: Callable[[], Awaitable[None]] | None = None,
    on_shutdown: Callable[[], Awaitable[None]] | None = None,
) -> FastAPI:
    duration = metrics.histogram(
        "request_duration",
        "seconds",
        "HTTP request latency by route and status",
        ["route", "status"],
        REQUEST_BUCKETS_S,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        if on_startup is not None:
            await on_startup()
        try:
            yield
        finally:
            if on_shutdown is not None:
                await on_shutdown()

    app = FastAPI(
        title="vms query",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        telemetry=http_telemetry(),
    )

    @app.middleware("http")
    async def observe(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        with remote_parent(request.headers):
            response = await call_next(request)
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        duration.labels(route=path, status=str(response.status_code)).observe(loop.time() - t0)
        return response

    @app.post("/v1/query")
    async def query(body: QueryRequest) -> Response:
        result = await engine.answer(body.question)
        return JSONResponse(result.model_dump(mode="json"), status_code=result.http_status)

    @app.get("/v1/catalog")
    async def get_catalog() -> Response:
        try:
            cat = await catalog.get()
        except ExecutionError as exc:
            return JSONResponse({"error": "database_unavailable", "detail": str(exc)}, 503)
        return JSONResponse(
            {
                "cameras": [
                    {
                        "camera_id": c.camera_id,
                        "name": c.name,
                        "timezone": c.timezone,
                        "zones": [z.name for z in cat.zones_of(c.camera_id)],
                    }
                    for c in cat.cameras
                ],
                "classes": list(cat.classes),
            }
        )

    @app.get("/v1/frames/{frame_id}/image")
    async def frame_image(frame_id: str = Path(pattern=ULID_PATTERN)) -> Response:
        try:
            found = await reader.fetch(FRAME_KEY_SQL, {"frame_id": frame_id})
        except ExecutionError as exc:
            return JSONResponse({"error": "database_unavailable", "detail": str(exc)}, 503)
        if not found.rows:
            return JSONResponse({"error": "not_found", "detail": frame_id}, 404)
        try:
            jpeg = await frames.get(str(found.rows[0][0]))
        except Exception as exc:  # object store errors, missing object
            return JSONResponse({"error": "object_store", "detail": str(exc)}, 502)
        # Frames never change once written.
        headers = {"Cache-Control": "public, max-age=86400, immutable"}
        return Response(jpeg, media_type="image/jpeg", headers=headers)

    @app.get("/v1/live")
    async def live() -> Response:
        try:
            cameras = await live_view(reader, await catalog.get())
        except ExecutionError as exc:
            return JSONResponse({"error": "database_unavailable", "detail": str(exc)}, 503)
        return JSONResponse({"cameras": [c.model_dump(mode="json") for c in cameras]})

    @app.get("/v1/examples")
    async def get_examples() -> dict[str, list[str]]:
        return {"questions": list(examples or [])}

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

    if ui_dir is not None and (ui_dir / "index.html").is_file():
        # Last, so API routes win; html=True serves index.html at "/".
        app.mount("/", StaticFiles(directory=ui_dir, html=True), name="ui")
    return app
