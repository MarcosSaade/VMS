"""HTTP layer of the ingest service (FastAPI). Parsing and status codes only; the logic is in
vms.ingest.service.

    POST /v1/frames    the frame contract (vms.common.ingest_api)
    GET  /healthz      the process is up
    GET  /readyz       Postgres and the object store answer (503 otherwise)
    GET  /metrics      Prometheus
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from starlette.datastructures import UploadFile

from vms.common.ingest_api import FRAMES_PATH, JPEG_FIELD, FrameMeta
from vms.common.metrics import Metrics
from vms.common.telemetry import http_telemetry, remote_parent
from vms.ingest.service import FrameIngestor, IngestError, Outcome

META_FIELDS = ("camera_id", "run_id", "seq", "ts_unix_ms")
# Room for the form fields and multipart boundaries on top of the JPEG itself.
MULTIPART_OVERHEAD_BYTES = 16 * 1024
MAX_FIELD_BYTES = 1024  # each non-file form field
READY_TIMEOUT_S = 2.0
REQUEST_BUCKETS_S = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)

ReadyCheck = Callable[[], Awaitable[bool]]


def _error(err: IngestError) -> JSONResponse:
    return JSONResponse({"error": err.code, "detail": err.detail}, status_code=err.status)


def create_app(
    ingestor: FrameIngestor,
    metrics: Metrics,
    *,
    max_jpeg_bytes: int,
    ready_checks: dict[str, ReadyCheck],
    on_startup: Callable[[], Awaitable[None]] | None = None,
    on_shutdown: Callable[[], Awaitable[None]] | None = None,
) -> FastAPI:
    max_body = max_jpeg_bytes + MULTIPART_OVERHEAD_BYTES
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
        title="vms ingest",
        lifespan=lifespan,
        docs_url=None,
        redoc_url=None,
        telemetry=http_telemetry(),
    )

    @app.post(FRAMES_PATH)
    async def post_frame(request: Request) -> Response:
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        # Continue the sender's trace (loadgen injects `traceparent`) if there is one.
        with remote_parent(request.headers):
            response = await _post_frame(request)
        duration.labels(route=FRAMES_PATH, status=str(response.status_code)).observe(
            loop.time() - t0
        )
        return response

    async def _post_frame(request: Request) -> Response:
        # Starlette limits form fields but not file parts, so the body size is enforced here,
        # before parsing. Requiring Content-Length closes the chunked-upload way around it.
        length = request.headers.get("content-length", "")
        if not length.isdigit():
            return _error(ingestor.invalid(411, "length_required", "Content-Length is required"))
        if int(length) > max_body:
            return _error(ingestor.invalid(413, "body_too_large", f"body over {max_body} bytes"))
        try:
            # Reading and parsing the multipart body: ~1.7 ms of CPU per 200 KB frame (E-010).
            with ingestor.timed("parse"):
                form = await request.form(
                    max_files=1, max_fields=len(META_FIELDS), max_part_size=MAX_FIELD_BYTES
                )
        except Exception as exc:  # malformed multipart, too many parts, wrong content type
            return _error(ingestor.invalid(400, "bad_multipart", str(exc)))
        try:
            upload = form.get(JPEG_FIELD)
            if not isinstance(upload, UploadFile):
                return _error(ingestor.invalid(422, "missing_jpeg", f"no file part {JPEG_FIELD!r}"))
            try:
                meta = FrameMeta.model_validate({k: form.get(k) for k in META_FIELDS})
            except ValidationError as exc:
                fields = sorted({str(e["loc"][0]) for e in exc.errors() if e["loc"]})
                return _error(ingestor.invalid(422, "invalid_fields", f"invalid: {fields}"))
            jpeg = await upload.read()
        finally:
            await form.close()  # releases spooled upload files
        try:
            result = await ingestor.ingest(meta, jpeg)
        except IngestError as err:
            return _error(err)
        status = 201 if result.outcome is Outcome.CREATED else 200
        return JSONResponse(result.accepted.model_dump(), status_code=status)

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
