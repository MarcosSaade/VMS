"""How loadgen delivers a frame: an injectable `Transport`.

- `HttpTransport`: multipart POST to the ingest REST endpoint (vms.common.ingest_api).
- `NullTransport`: accepts every frame without I/O; the self-test uses it to measure the
  generator's own maximum rate.
- `RetryingTransport`: bounded retries with exponential backoff and full jitter, wrapping
  either of the above.

A transport never raises for delivery problems: every send ends in a `SendResult`.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

import aiohttp

from vms.common.ingest_api import FRAMES_PATH, JPEG_CONTENT_TYPE, JPEG_FIELD, REJECT_STATUSES


class Outcome(StrEnum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"  # the server shed the frame (backpressure)
    FAILED = "failed"


@dataclass(frozen=True)
class OutgoingFrame:
    camera_id: str
    run_id: str
    seq: int
    ts_unix_ms: int
    jpeg: bytes
    headers: Mapping[str, str] = field(default_factory=dict)  # e.g. trace context


@dataclass(frozen=True)
class SendResult:
    outcome: Outcome
    status: int | None  # HTTP status, None when no response arrived
    reason: str  # the status code as text, or timeout / connection / client_error
    attempts: int = 1

    @property
    def retryable(self) -> bool:
        """Transient failures. 429/503 are rejections, not failures, and are never retried."""
        if self.outcome is not Outcome.FAILED:
            return False
        return self.status is None or self.status >= 500


def classify_status(status: int) -> Outcome:
    if 200 <= status < 300:
        return Outcome.ACCEPTED
    if status in REJECT_STATUSES:
        return Outcome.REJECTED
    return Outcome.FAILED


def from_status(status: int) -> SendResult:
    return SendResult(classify_status(status), status, str(status))


class Transport(Protocol):
    async def send(self, frame: OutgoingFrame) -> SendResult: ...

    async def aclose(self) -> None: ...


class NullTransport:
    """Accepts instantly. Yields to the event loop once, like a real send would."""

    def __init__(self) -> None:
        self.sent = 0

    async def send(self, frame: OutgoingFrame) -> SendResult:
        await asyncio.sleep(0)
        self.sent += 1
        return SendResult(Outcome.ACCEPTED, 202, "202")

    async def aclose(self) -> None:
        return None


class HttpTransport:
    """Multipart POST per frame over a pooled aiohttp session (keep-alive connections)."""

    def __init__(
        self,
        base_url: str,
        *,
        request_timeout_s: float,
        connect_timeout_s: float,
        max_connections: int,
    ) -> None:
        self._url = base_url.rstrip("/") + FRAMES_PATH
        self._timeout = aiohttp.ClientTimeout(total=request_timeout_s, connect=connect_timeout_s)
        self._max_connections = max_connections
        self._session: aiohttp.ClientSession | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        # Created lazily: a ClientSession must be created inside the running event loop.
        if self._session is None:
            connector = aiohttp.TCPConnector(limit=self._max_connections)
            self._session = aiohttp.ClientSession(connector=connector, timeout=self._timeout)
        return self._session

    async def send(self, frame: OutgoingFrame) -> SendResult:
        form = aiohttp.FormData()
        form.add_field("camera_id", frame.camera_id)
        form.add_field("run_id", frame.run_id)
        form.add_field("seq", str(frame.seq))
        form.add_field("ts_unix_ms", str(frame.ts_unix_ms))
        form.add_field(
            JPEG_FIELD, frame.jpeg, filename=f"{frame.seq}.jpg", content_type=JPEG_CONTENT_TYPE
        )
        try:
            async with self._get_session().post(
                self._url, data=form, headers=dict(frame.headers)
            ) as response:
                await response.read()  # drain so the connection returns to the pool
                return from_status(response.status)
        except TimeoutError:
            return SendResult(Outcome.FAILED, None, "timeout")
        except aiohttp.ClientConnectionError:
            return SendResult(Outcome.FAILED, None, "connection")
        except aiohttp.ClientError:
            return SendResult(Outcome.FAILED, None, "client_error")

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


class RetryingTransport:
    """Retries transient failures up to `max_retries` times.

    Backoff before retry n (1-based) is uniform in [0, min(max_s, base_s * 2**(n-1))]: "full
    jitter", so many cameras failing together do not retry in lockstep.
    """

    def __init__(
        self,
        inner: Transport,
        *,
        max_retries: int,
        base_s: float,
        max_s: float,
        rng: random.Random,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._inner = inner
        self._max_retries = max_retries
        self._base_s = base_s
        self._max_s = max_s
        self._rng = rng
        self._sleep = sleep

    def backoff(self, retry: int) -> float:
        return self._rng.uniform(0.0, min(self._max_s, self._base_s * 2 ** (retry - 1)))

    async def send(self, frame: OutgoingFrame) -> SendResult:
        result = await self._inner.send(frame)
        retry = 0
        while result.retryable and retry < self._max_retries:
            retry += 1
            await self._sleep(self.backoff(retry))
            result = await self._inner.send(frame)
        return SendResult(result.outcome, result.status, result.reason, attempts=retry + 1)

    async def aclose(self) -> None:
        await self._inner.aclose()
