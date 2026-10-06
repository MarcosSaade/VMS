"""HTTP access for dataset downloads: bounded retries, range reads and checksummed files.

The transport sits behind the small `Transport` protocol, so tests run with an in-memory fake
and no network.
"""

from __future__ import annotations

import hashlib
import http.client
import io
import os
import random
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, cast

from vms.common.logs import get_logger

log = get_logger(__name__)

USER_AGENT = "vms-datasets/0.1 (+https://github.com/MarcosSaade)"
CHUNK_BYTES = 1 << 20


class Transport(Protocol):
    def size(self, url: str) -> int:
        """Content length of `url` in bytes (HEAD)."""
        ...

    def stream(self, url: str, start: int = 0, end: int | None = None) -> Iterator[bytes]:
        """Yield the bytes of `url` in [start, end] (inclusive; end None = to EOF)."""
        ...


class DownloadError(RuntimeError):
    pass


class ChecksumError(DownloadError):
    pass


@dataclass
class RetryPolicy:
    """Exponential backoff with full jitter, bounded in attempts."""

    attempts: int = 5
    base_s: float = 1.0
    max_s: float = 30.0
    sleep: Callable[[float], None] = time.sleep
    rng: random.Random = field(default_factory=lambda: random.Random(0))

    def run[T](self, what: str, fn: Callable[[], T]) -> T:
        for attempt in range(1, self.attempts + 1):
            try:
                return fn()
            except (OSError, urllib.error.URLError, TimeoutError) as exc:
                if isinstance(exc, urllib.error.HTTPError) and exc.code < 500 and exc.code != 429:
                    raise DownloadError(f"{what}: HTTP {exc.code}") from exc
                if attempt == self.attempts:
                    raise DownloadError(f"{what}: failed after {attempt} attempts: {exc}") from exc
                delay = self.rng.uniform(0, min(self.max_s, self.base_s * 2 ** (attempt - 1)))
                log.warning(
                    "retrying download",
                    extra={
                        "what": what,
                        "attempt": attempt,
                        "delay_s": round(delay, 2),
                        "error": str(exc),
                    },
                )
                self.sleep(delay)
        raise AssertionError("unreachable")


class UrllibTransport:
    def __init__(self, timeout_s: float = 60.0) -> None:
        self._timeout_s = timeout_s

    def _open(
        self, url: str, method: str = "GET", headers: dict[str, str] | None = None
    ) -> http.client.HTTPResponse:
        request = urllib.request.Request(
            url, method=method, headers={"User-Agent": USER_AGENT, **(headers or {})}
        )
        return cast(
            http.client.HTTPResponse, urllib.request.urlopen(request, timeout=self._timeout_s)
        )

    def size(self, url: str) -> int:
        with self._open(url, method="HEAD") as response:
            return int(response.headers["Content-Length"])

    def stream(self, url: str, start: int = 0, end: int | None = None) -> Iterator[bytes]:
        headers = {}
        if start or end is not None:
            headers["Range"] = f"bytes={start}-{'' if end is None else end}"
        with self._open(url, headers=headers) as response:
            if headers and response.status != 206:
                raise DownloadError(f"{url}: server ignored the Range header")
            while chunk := response.read(CHUNK_BYTES):
                yield chunk


def read_range(transport: Transport, retry: RetryPolicy, url: str, start: int, end: int) -> bytes:
    """Bytes [start, end] inclusive, retried as a whole."""

    def fetch() -> bytes:
        data = b"".join(transport.stream(url, start, end))
        if len(data) != end - start + 1:
            raise OSError(f"short read: got {len(data)} of {end - start + 1} bytes")
        return data

    return retry.run(f"{url} [{start}-{end}]", fetch)


class RemoteFile(io.RawIOBase):
    """Seekable read-only view of a remote file over HTTP range requests.

    `prefetch(start, end)` pulls one contiguous span in a single request into a local spool
    file; reads inside the span are then served locally. This turns "read 1,000 zip members"
    into one large request instead of 1,000 small ones.
    """

    def __init__(self, transport: Transport, retry: RetryPolicy, url: str, size: int) -> None:
        super().__init__()
        self._t, self._retry, self._url, self._size = transport, retry, url, size
        self._pos = 0
        self._span: tuple[int, int] | None = None
        self._spool: io.BufferedRandom | None = None

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self._pos, io.SEEK_END: self._size}[whence]
        self._pos = max(0, base + offset)
        return self._pos

    def prefetch(self, start: int, end: int, spool_dir: Path) -> None:
        end = min(end, self._size - 1)
        spool = open(  # noqa: SIM115 - lives as long as this object; closed in close()
            spool_dir / f".spool-{os.getpid()}-{id(self)}", "w+b"
        )
        os.unlink(spool.name)  # anonymous once open

        def fetch() -> None:
            spool.seek(0)
            spool.truncate()
            written = 0
            for chunk in self._t.stream(self._url, start, end):
                spool.write(chunk)
                written += len(chunk)
            if written != end - start + 1:
                raise OSError(f"short read: got {written} of {end - start + 1} bytes")

        self._retry.run(f"{self._url} [{start}-{end}]", fetch)
        if self._spool is not None:
            self._spool.close()
        self._spool, self._span = spool, (start, end)

    def readinto(self, buffer: bytearray | memoryview) -> int:  # type: ignore[override]
        if self._pos >= self._size:
            return 0
        n = min(len(buffer), self._size - self._pos)
        if self._span and self._spool and self._span[0] <= self._pos <= self._span[1]:
            n = min(n, self._span[1] - self._pos + 1)
            self._spool.seek(self._pos - self._span[0])
            data = self._spool.read(n)
        else:
            data = read_range(self._t, self._retry, self._url, self._pos, self._pos + n - 1)
        buffer[: len(data)] = data
        self._pos += len(data)
        return len(data)

    def close(self) -> None:
        if self._spool is not None:
            self._spool.close()
            self._spool = None
        super().close()


def verify_checksum(path: Path, checksum: str) -> None:
    algo, expected = checksum.split(":", 1)
    digest = hashlib.new(algo)
    with path.open("rb") as fh:
        while block := fh.read(CHUNK_BYTES):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ChecksumError(f"{path.name}: {algo} {digest.hexdigest()} != expected {expected}")


def download_file(
    transport: Transport,
    retry: RetryPolicy,
    url: str,
    dest: Path,
    checksum: str,
    expected_bytes: int,
) -> Path:
    """Download `url` to `dest` unless a verified copy is already there (idempotent).

    Writes to a temporary file and renames only after the size and checksum match, so an
    interrupted or corrupt download never looks complete.
    """
    if dest.exists():
        try:
            verify_checksum(dest, checksum)
            return dest
        except ChecksumError:
            log.warning("cached file failed checksum; downloading again", extra={"path": str(dest)})
            dest.unlink()
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".partial")

    def fetch() -> None:
        written = 0
        with partial.open("wb") as fh:
            for chunk in transport.stream(url):
                fh.write(chunk)
                written += len(chunk)
        if written != expected_bytes:
            raise OSError(f"short read: got {written} of {expected_bytes} bytes")

    log.info("downloading", extra={"url": url, "bytes": expected_bytes})
    try:
        retry.run(url, fetch)
        verify_checksum(partial, checksum)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(dest)
    return dest
