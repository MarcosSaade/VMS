"""A stub ingest server: implements the frame contract (vms.common.ingest_api) without storing.

Used to test and measure loadgen alone (task 1.2) before the real ingest exists, and as the
"server is free" end of a client-cost measurement. It parses and validates every upload like a
real server would, can add latency and reject a seeded fraction with 429, and remembers
`(camera_id, run_id, seq)` so a retried frame is answered as a duplicate.

Run it in its own process (`python -m vms.loadgen stub`) so its CPU does not count against the
generator's. One process parses about 600 multipart frames/s (E-010); `workers` > 1 starts that
many processes sharing the port (SO_REUSEPORT, Linux), each with its own duplicate memory and
metrics, so a max-rate measurement is limited by the client and not by the stub.
"""

from __future__ import annotations

import asyncio
import multiprocessing
import random
from collections.abc import Awaitable, Callable
from typing import Any

from aiohttp import web
from pydantic import ValidationError

from vms.common.ids import UlidGenerator
from vms.common.ingest_api import FRAMES_PATH, JPEG_FIELD, FrameAccepted, FrameMeta
from vms.common.metrics import Metrics
from vms.common.observability import init_observability
from vms.loadgen.settings import LoadgenSettings, StubSettings

# A 1280x720 JPEG is ~100 to 250 KB; aiohttp's 1 MiB default would reject large 1080p frames.
MAX_UPLOAD_BYTES = 8 * 1024 * 1024


def build_app(
    settings: StubSettings,
    metrics: Metrics,
    *,
    rng: random.Random,
    ids: UlidGenerator | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> web.Application:
    ids = ids or UlidGenerator()
    seen: dict[tuple[str, str, int], str] = {}
    received = metrics.counter("received_frames", "total", "Frames by response", ["outcome"])
    received_bytes = metrics.counter("received_jpeg_bytes", "total", "JPEG bytes received")

    async def post_frame(request: web.Request) -> web.Response:
        form = await request.post()
        upload = form.get(JPEG_FIELD)
        if not isinstance(upload, web.FileField):
            received.labels(outcome="invalid").inc()
            return web.json_response({"error": f"missing file field {JPEG_FIELD!r}"}, status=400)
        try:
            meta = FrameMeta.model_validate(
                {k: form.get(k) for k in ("camera_id", "run_id", "seq", "ts_unix_ms")}
            )
        except ValidationError as exc:
            received.labels(outcome="invalid").inc()
            return web.json_response({"error": exc.errors(include_url=False)}, status=422)
        jpeg = upload.file.read()
        received_bytes.inc(len(jpeg))
        if settings.latency_ms > 0:
            await sleep(settings.latency_ms / 1000.0)
        if settings.reject_ratio > 0 and rng.random() < settings.reject_ratio:
            received.labels(outcome="rejected").inc()
            return web.json_response({"error": "overloaded"}, status=429)
        key = (meta.camera_id, meta.run_id, meta.seq)
        duplicate = key in seen
        if not duplicate:
            seen[key] = ids.new()
        received.labels(outcome="duplicate" if duplicate else "accepted").inc()
        body = FrameAccepted(frame_id=seen[key], duplicate=duplicate)
        return web.json_response(body.model_dump(), status=202)

    async def ok(_: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def metrics_endpoint(_: web.Request) -> web.Response:
        return web.Response(body=metrics.exposition(), content_type="text/plain")

    app = web.Application(client_max_size=MAX_UPLOAD_BYTES)
    app.router.add_post(FRAMES_PATH, post_frame)
    app.router.add_get("/healthz", ok)
    app.router.add_get("/readyz", ok)
    app.router.add_get("/metrics", metrics_endpoint)
    return app


def _serve_one(config: dict[str, Any], worker: int) -> None:
    # A plain dict crosses the process boundary; load_settings' bound subclass cannot be pickled.
    settings = LoadgenSettings.model_validate(config)
    stub = settings.stub
    obs = init_observability(settings)
    app = build_app(stub, Metrics("loadgen_stub"), rng=random.Random(settings.seed + worker))
    obs.logger.info(
        "stub ingest listening", extra={"host": stub.host, "port": stub.port, "worker": worker}
    )
    web.run_app(
        app,
        host=stub.host,
        port=stub.port,
        reuse_port=stub.workers > 1,
        access_log=None,
        print=None,
    )


def serve(settings: LoadgenSettings) -> None:
    """Serve until SIGINT/SIGTERM, in `settings.stub.workers` processes."""
    if settings.stub.workers == 1:
        _serve_one(settings.model_dump(), 0)
        return
    ctx = multiprocessing.get_context("spawn")
    procs = [
        ctx.Process(target=_serve_one, args=(settings.model_dump(), w), daemon=True)
        for w in range(settings.stub.workers)
    ]
    for proc in procs:
        proc.start()
    try:
        for proc in procs:
            proc.join()
    except KeyboardInterrupt:
        pass
    finally:
        for proc in procs:
            proc.terminate()
            proc.join()
