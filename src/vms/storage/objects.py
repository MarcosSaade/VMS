"""Object store access behind a small async interface.

`S3ObjectStore` talks to any S3 API (SeaweedFS locally, AWS S3 in the cloud) with boto3. boto3
is blocking, so calls run in a bounded thread pool and the event loop never waits on a socket;
the pool size caps concurrent requests to the store. boto3's "standard" retry mode retries
throttling and transient errors with exponential backoff and jitter, bounded by `max_attempts`,
and every call has connect and read timeouts.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Protocol

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError
from pydantic import BaseModel, ConfigDict, Field

from vms.storage.config import ObjectStoreSettings


class ObjectStoreError(Exception):
    """The store could not complete a request (after its own retries)."""


class ObjectStoreTuning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_concurrency: int = Field(default=16, ge=1)
    connect_timeout_s: float = Field(default=2.0, gt=0)
    read_timeout_s: float = Field(default=5.0, gt=0)
    max_attempts: int = Field(default=3, ge=1, le=10)  # first try included


class ObjectStore(Protocol):
    async def put(self, key: str, data: bytes, content_type: str) -> None: ...

    async def delete(self, key: str) -> None: ...

    async def ready(self) -> bool:
        """Whether the store answers and the bucket exists (readiness probe)."""
        ...

    async def aclose(self) -> None: ...


class S3ObjectStore:
    def __init__(self, settings: ObjectStoreSettings, tuning: ObjectStoreTuning) -> None:
        self._bucket = settings.bucket
        self._pool = ThreadPoolExecutor(max_workers=tuning.max_concurrency, thread_name_prefix="s3")

        def secret(value: Any) -> str | None:
            return None if value is None else str(value.get_secret_value())

        self._client: Any = boto3.client(
            "s3",
            endpoint_url=settings.endpoint_url,
            region_name=settings.region,
            aws_access_key_id=secret(settings.access_key_id),
            aws_secret_access_key=secret(settings.secret_access_key),
            config=Config(
                connect_timeout=tuning.connect_timeout_s,
                read_timeout=tuning.read_timeout_s,
                retries={"max_attempts": tuning.max_attempts, "mode": "standard"},
                max_pool_connections=tuning.max_concurrency,
                # Path-style URLs: SeaweedFS has no per-bucket DNS names.
                s3={"addressing_style": "path"},
            ),
        )

    async def _call(self, method: str, **kwargs: Any) -> Any:
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(
                self._pool, lambda: getattr(self._client, method)(**kwargs)
            )
        except (BotoCoreError, ClientError) as exc:
            raise ObjectStoreError(f"{method} failed: {exc}") from exc

    async def put(self, key: str, data: bytes, content_type: str) -> None:
        await self._call(
            "put_object", Bucket=self._bucket, Key=key, Body=data, ContentType=content_type
        )

    async def get(self, key: str) -> bytes:
        response = await self._call("get_object", Bucket=self._bucket, Key=key)
        body: bytes = await asyncio.get_running_loop().run_in_executor(
            self._pool, response["Body"].read
        )
        return body

    async def delete(self, key: str) -> None:
        await self._call("delete_object", Bucket=self._bucket, Key=key)

    async def ready(self) -> bool:
        try:
            await self._call("head_bucket", Bucket=self._bucket)
        except ObjectStoreError:
            return False
        return True

    async def aclose(self) -> None:
        self._pool.shutdown(wait=True)
        self._client.close()
