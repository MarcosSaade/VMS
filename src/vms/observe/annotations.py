"""Grafana annotations: markers on the dashboards' time axis (task 9.4).

The load generator posts one annotation per injected drift event (vms.loadgen.drift) when a
repetition starts, so the Drift dashboard shows the known onset next to the alarm that
answers it. An abrupt drift is a point; a gradual one is a region over its ramp. The
dashboards query them by tag (`drift-onset`).

Annotations are best effort: Grafana being down or the credentials being wrong is logged and
counted, never fails the run. Posting needs an editor, so the client authenticates as the
Grafana admin from `.env` (`GRAFANA_ADMIN_USER`, `GRAFANA_ADMIN_PASSWORD`), as the compose
stack does; the anonymous viewer can only read.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

import aiohttp
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from vms.common.logs import get_logger

log = get_logger(__name__)


class GrafanaSettings(BaseSettings):
    """The local Grafana, from the same `GRAFANA_*` variables the compose stack reads."""

    model_config = SettingsConfigDict(env_prefix="GRAFANA_", env_file=".env", extra="ignore")

    host: str = "127.0.0.1"
    port: int = Field(default=3000, ge=1, le=65535)
    admin_user: str = "admin"
    admin_password: SecretStr = SecretStr("change-me-grafana")

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class Annotation:
    time: datetime
    text: str
    tags: tuple[str, ...]
    time_end: datetime | None = None  # a region; None is a point

    def payload(self) -> dict[str, Any]:
        """The body of POST /api/annotations (times in Unix milliseconds)."""
        body: dict[str, Any] = {
            "time": round(self.time.timestamp() * 1000),
            "tags": list(self.tags),
            "text": self.text,
        }
        if self.time_end is not None:
            body["timeEnd"] = round(self.time_end.timestamp() * 1000)
        return body


class AnnotationError(Exception):
    def __init__(self, reason: str, *, retryable: bool) -> None:
        super().__init__(reason)
        self.retryable = retryable


class AnnotationSink(Protocol):
    async def post(self, annotation: Annotation) -> None:
        """Store one annotation; raises AnnotationError."""

    async def aclose(self) -> None: ...


class GrafanaAnnotations:
    """POST /api/annotations with basic auth, one pooled session."""

    def __init__(self, settings: GrafanaSettings, *, timeout_s: float) -> None:
        self._url = f"{settings.url}/api/annotations"
        self._auth = aiohttp.BasicAuth(
            settings.admin_user, settings.admin_password.get_secret_value()
        )
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._session: aiohttp.ClientSession | None = None

    async def post(self, annotation: Annotation) -> None:
        if self._session is None:
            self._session = aiohttp.ClientSession(timeout=self._timeout, auth=self._auth)
        try:
            async with self._session.post(self._url, json=annotation.payload()) as resp:
                if resp.status >= 300:
                    # 5xx may pass; 401/403/400 will not (credentials, a bad body).
                    raise AnnotationError(f"http {resp.status}", retryable=resp.status >= 500)
        except TimeoutError as exc:
            raise AnnotationError("timeout", retryable=True) from exc
        except aiohttp.ClientError as exc:
            raise AnnotationError(f"connection: {exc}", retryable=True) from exc

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


@dataclass
class PostResult:
    posted: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {"posted": self.posted, "failed": self.failed, "errors": sorted(set(self.errors))}


async def post_all(
    sink: AnnotationSink,
    annotations: Sequence[Annotation],
    *,
    max_retries: int,
    base_s: float,
    max_s: float,
    rng: random.Random,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> PostResult:
    """Post each annotation, retrying transient failures with capped, fully jittered
    exponential backoff. Never raises AnnotationError: failures are counted and logged once."""
    result = PostResult()
    for annotation in annotations:
        retry = 0
        while True:
            try:
                await sink.post(annotation)
            except AnnotationError as exc:
                if exc.retryable and retry < max_retries:
                    retry += 1
                    await sleep(rng.uniform(0.0, min(max_s, base_s * 2 ** (retry - 1))))
                    continue
                result.failed += 1
                result.errors.append(str(exc))
            else:
                result.posted += 1
            break
    if result.failed:
        log.warning(
            "grafana annotations not posted; the dashboards will lack these markers",
            extra=result.summary(),
        )
    return result
