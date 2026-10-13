"""Frame ingestion: validate, store the JPEG, record the frame. Framework-free, so the HTTP layer
(vms.ingest.app) stays thin and this logic is tested with fakes.

Write order and what each failure leaves behind:

1. Put the JPEG at `frames/{camera_id}/{yyyy-mm-dd}/{frame_id}.jpg` (capture date, UTC).
   Fails -> nothing stored; 500, the client may retry.
2. Insert the `frames` row (status `pending`) with ON CONFLICT on `(camera_id, run_id, seq)`.
   Fails -> the object from step 1 is deleted (best effort) and the request fails with 500.
   If that delete also fails, the object is orphaned: counted in
   `vms_ingest_orphaned_objects_total`, harmless (no row points at it), and reclaimable by a
   sweep that deletes objects without a row. A row never points at a missing object.
3. Conflict (a retry or a concurrent duplicate): the existing row wins. The object written in
   step 1 is deleted and the response carries the original `frame_id` with `duplicate: true`.

Object first, row second, because the row is what makes a frame visible to the worker: the
worker must never claim a frame whose image is not there yet.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from opentelemetry import trace

from vms.common.clock import Clock
from vms.common.ids import UlidGenerator
from vms.common.ingest_api import JPEG_CONTENT_TYPE, FrameAccepted, FrameMeta
from vms.common.logs import get_logger
from vms.common.metrics import Metrics
from vms.common.telemetry import current_traceparent
from vms.ingest.settings import FrameLimits
from vms.storage.frames import FrameRepository, FrameStoreError, NewFrame
from vms.storage.objects import ObjectStore, ObjectStoreError

log = get_logger(__name__)

JPEG_MAGIC = b"\xff\xd8\xff"
STAGE_BUCKETS_S = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
LAG_BUCKETS_S = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


class Outcome(StrEnum):
    CREATED = "created"
    DUPLICATE = "duplicate"
    INVALID = "invalid"
    ERROR = "error"


class IngestError(Exception):
    """A request that cannot be ingested. `status` is the HTTP status to answer with."""

    def __init__(self, status: int, code: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class IngestResult:
    accepted: FrameAccepted
    outcome: Outcome


def object_key(camera_id: str, ts: datetime, frame_id: str) -> str:
    return f"frames/{camera_id}/{ts.astimezone(UTC):%Y-%m-%d}/{frame_id}.jpg"


class FrameIngestor:
    def __init__(
        self,
        store: ObjectStore,
        repo: FrameRepository,
        limits: FrameLimits,
        metrics: Metrics,
        *,
        clock: Clock,
        ids: UlidGenerator,
        tracer: trace.Tracer | None = None,
    ) -> None:
        self._store = store
        self._repo = repo
        self._limits = limits
        self._clock = clock
        self._ids = ids
        self._tracer = tracer or trace.get_tracer(__name__)
        self._frames = metrics.counter("frames", "total", "Frames by outcome", ["outcome"])
        self._bytes = metrics.counter("received_jpeg_bytes", "total", "JPEG bytes stored")
        self._orphans = metrics.counter(
            "orphaned_objects", "total", "Objects left without a row (cleanup failed)"
        )
        self._stage = metrics.histogram(
            "stage_duration", "seconds", "Time per ingest stage", ["stage"], STAGE_BUCKETS_S
        )
        self._lag = metrics.histogram(
            "capture_to_receive",
            "seconds",
            "Server receive time minus frame capture time (network + client queueing + skew)",
            buckets=LAG_BUCKETS_S,
        )

    def invalid(self, status: int, code: str, detail: str) -> IngestError:
        """Count and build a validation error (also used by the HTTP layer)."""
        self._frames.labels(outcome=Outcome.INVALID).inc()
        return IngestError(status, code, detail)

    def _validate(self, meta: FrameMeta, jpeg: bytes) -> datetime:
        if not jpeg:
            raise self.invalid(422, "empty_jpeg", "the jpeg part is empty")
        if len(jpeg) > self._limits.max_jpeg_bytes:
            raise self.invalid(
                413, "jpeg_too_large", f"jpeg is larger than {self._limits.max_jpeg_bytes} bytes"
            )
        if not jpeg.startswith(JPEG_MAGIC):
            raise self.invalid(415, "not_jpeg", "the jpeg part does not start with a JPEG marker")
        ts = datetime.fromtimestamp(meta.ts_unix_ms / 1000, tz=UTC)
        now = self._clock.now()
        if (ts - now).total_seconds() > self._limits.max_future_skew_s:
            raise self.invalid(422, "ts_in_future", "ts_unix_ms is too far in the future")
        self._lag.observe(max(0.0, (now - ts).total_seconds()))
        return ts

    async def ingest(self, meta: FrameMeta, jpeg: bytes) -> IngestResult:
        """Store one frame. The `ingest.frame` span continues the sender's trace (if the HTTP
        layer attached one) and its traceparent is stored on the row for the worker (D25)."""
        with self._tracer.start_as_current_span(
            "ingest.frame",
            attributes={"camera_id": meta.camera_id, "seq": meta.seq},
        ) as span:
            try:
                result = await self._ingest(meta, jpeg)
            except IngestError as err:
                span.set_attribute("outcome", err.code)
                raise
            span.set_attribute("frame_id", result.accepted.frame_id)
            span.set_attribute("outcome", result.outcome.value)
            return result

    async def _ingest(self, meta: FrameMeta, jpeg: bytes) -> IngestResult:
        ts = self._validate(meta, jpeg)
        frame_id = self._ids.new()
        key = object_key(meta.camera_id, ts, frame_id)

        try:
            with self.timed("object_put"):
                await self._store.put(key, jpeg, JPEG_CONTENT_TYPE)
        except ObjectStoreError as exc:
            self._frames.labels(outcome=Outcome.ERROR).inc()
            log.warning("object put failed", extra={"key": key, "error": str(exc)})
            raise IngestError(500, "object_store_unavailable", "could not store the frame") from exc

        frame = NewFrame(
            frame_id,
            meta.camera_id,
            meta.run_id,
            meta.seq,
            ts,
            key,
            len(jpeg),
            traceparent=current_traceparent(),  # the ingest.frame span
        )
        try:
            with self.timed("db_insert"):
                stored = await self._repo.insert(frame)
        except FrameStoreError as exc:
            self._frames.labels(outcome=Outcome.ERROR).inc()
            log.warning("frame insert failed", extra={"frame_id": frame_id, "error": str(exc)})
            await self._delete_quietly(key)
            raise IngestError(500, "database_unavailable", "could not record the frame") from exc

        if not stored.created:
            await self._delete_quietly(key)
            self._frames.labels(outcome=Outcome.DUPLICATE).inc()
            return IngestResult(
                FrameAccepted(frame_id=stored.frame_id, duplicate=True), Outcome.DUPLICATE
            )

        self._frames.labels(outcome=Outcome.CREATED).inc()
        self._bytes.inc(len(jpeg))
        return IngestResult(FrameAccepted(frame_id=frame_id), Outcome.CREATED)

    async def _delete_quietly(self, key: str) -> None:
        try:
            with self.timed("cleanup"):
                await self._store.delete(key)
        except ObjectStoreError as exc:
            self._orphans.inc()
            log.error("orphaned object", extra={"key": key, "error": str(exc)})

    @contextmanager
    def timed(self, stage: str) -> Iterator[None]:
        """A span plus a stage-latency observation around one step."""
        with self._tracer.start_as_current_span(f"ingest.{stage}"):
            t0 = self._clock.monotonic()
            try:
                yield
            finally:
                self._stage.labels(stage=stage).observe(self._clock.monotonic() - t0)
