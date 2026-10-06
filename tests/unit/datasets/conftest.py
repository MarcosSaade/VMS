"""In-memory fakes for dataset tests: no network, deterministic."""

from __future__ import annotations

import hashlib
import io
import zipfile
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
import pytest

from vms.datasets.http import RetryPolicy


class FakeTransport:
    """Serves byte blobs by URL with HEAD and Range support; counts requests; can fail."""

    def __init__(self, files: dict[str, bytes]) -> None:
        self.files = files
        self.requests: list[tuple[str, int, int | None]] = []
        self.fail_next = 0  # raise OSError on the next n stream() calls
        self.truncate_next = 0  # return a short body on the next n stream() calls

    def size(self, url: str) -> int:
        return len(self.files[url])

    def stream(self, url: str, start: int = 0, end: int | None = None) -> Iterator[bytes]:
        self.requests.append((url, start, end))
        if self.fail_next:
            self.fail_next -= 1
            raise OSError("connection reset")
        data = self.files[url][start : None if end is None else end + 1]
        if self.truncate_next:
            self.truncate_next -= 1
            data = data[: len(data) // 2]
        for i in range(0, len(data), 4096):
            yield data[i : i + 4096]


@pytest.fixture
def retry() -> RetryPolicy:
    sleeps: list[float] = []
    return RetryPolicy(attempts=3, base_s=0.01, sleep=sleeps.append)


def jpeg(width: int, height: int, value: int) -> bytes:
    image = np.full((height, width, 3), value, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return bytes(buf)


def make_zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buf.getvalue()


def sequence_members(prefix: str, n: int, width: int = 64, height: int = 48) -> dict[str, bytes]:
    return {f"{prefix}{i:06d}.jpg": jpeg(width, height, (i * 20) % 256) for i in range(1, n + 1)}


def write_video(path: Path, frames: int, fps: float, width: int = 64, height: int = 48) -> bytes:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height))
    assert writer.isOpened()
    for i in range(frames):
        writer.write(np.full((height, width, 3), (i * 7) % 256, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def sha(algo: str, data: bytes) -> str:
    return f"{algo}:{hashlib.new(algo, data).hexdigest()}"
