"""Wire the ingest service from settings: storage clients, ingestor, HTTP app, uvicorn."""

from __future__ import annotations

import uvicorn
from fastapi import FastAPI

from vms.common.clock import SystemClock
from vms.common.ids import UlidGenerator
from vms.common.observability import Observability, init_observability
from vms.common.settings import load_settings
from vms.ingest.app import create_app
from vms.ingest.service import FrameIngestor
from vms.ingest.settings import IngestSettings
from vms.storage.config import ObjectStoreSettings, PostgresSettings
from vms.storage.frames import PostgresFrameRepository
from vms.storage.objects import S3ObjectStore

SERVICE = "ingest"


def build_app(settings: IngestSettings, obs: Observability) -> FastAPI:
    postgres = PostgresSettings()
    repo = PostgresFrameRepository(
        postgres.conninfo(
            connect_timeout_s=settings.db.connect_timeout_s, application_name=SERVICE
        ),
        settings.db,
    )
    store = S3ObjectStore(ObjectStoreSettings(), settings.objects)
    ingestor = FrameIngestor(
        store,
        repo,
        settings.limits,
        obs.metrics,
        clock=SystemClock(),
        ids=UlidGenerator(),
        tracer=obs.tracer,
    )

    async def startup() -> None:
        await repo.open()
        obs.logger.info(
            "ingest listening",
            extra={"host": settings.http.host, "port": settings.http.port, "db": postgres.db},
        )

    async def shutdown() -> None:
        await repo.aclose()
        await store.aclose()
        obs.shutdown()

    return create_app(
        ingestor,
        obs.metrics,
        max_jpeg_bytes=settings.limits.max_jpeg_bytes,
        ready_checks={"postgres": repo.ready, "object_store": store.ready},
        on_startup=startup,
        on_shutdown=shutdown,
    )


def main() -> int:
    settings = load_settings(IngestSettings, SERVICE)
    obs = init_observability(settings)
    app = build_app(settings, obs)
    uvicorn.run(
        app,
        host=settings.http.host,
        port=settings.http.port,
        timeout_graceful_shutdown=int(settings.http.graceful_shutdown_s),
        log_config=None,  # keep the JSON logging configured by init_observability
        access_log=False,  # per-request logs would dominate at hundreds of frames/s; see metrics
    )
    return 0
