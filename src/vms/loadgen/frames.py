"""Load clips into memory as ready-to-send JPEG bytes.

All decoding, resizing and encoding happens once at startup, so sending a frame costs only the
request itself and the generator stays far below the rates it drives (measured by the
self-test). Clips are shared by every camera that plays them.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2

from vms.datasets.clips import FRAMES_DIR, clip_dir, decode_jpeg, encode_jpeg, read_manifest
from vms.loadgen.settings import FrameSettings

# Used when a clip must be resized but frame.jpeg_quality is null (OpenCV's own default).
FALLBACK_JPEG_QUALITY = 95


class ClipNotFoundError(FileNotFoundError):
    pass


@dataclass(frozen=True)
class LoadedClip:
    id: str
    fps: float
    width: int
    height: int
    frames: tuple[bytes, ...]

    @property
    def total_bytes(self) -> int:
        return sum(len(f) for f in self.frames)


def prepare_frame(data: bytes, size: tuple[int, int] | None, quality: int | None) -> bytes:
    """Resize to `size` (width, height) and/or re-encode at `quality`; pass through otherwise."""
    if size is None and quality is None:
        return data
    image = decode_jpeg(data)
    if size is not None and (image.shape[1], image.shape[0]) != size:
        image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return encode_jpeg(image, quality if quality is not None else FALLBACK_JPEG_QUALITY)


def load_clip(
    data_dir: Path,
    clip_id: str,
    settings: FrameSettings,
    max_frames: int | None = None,
    workers: int | None = None,
) -> LoadedClip:
    manifest = read_manifest(data_dir, clip_id)
    if manifest is None:
        raise ClipNotFoundError(
            f"clip {clip_id!r} not found under {data_dir}/clips; run `make data-mini` or "
            "`make data` first"
        )
    paths = sorted((clip_dir(data_dir, clip_id) / FRAMES_DIR).glob("*.jpg"))[:max_frames]
    if not paths:
        raise ClipNotFoundError(f"clip {clip_id!r} has no frames")

    size: tuple[int, int] | None = None
    width, height = manifest.width, manifest.height
    if settings.width is not None and settings.height is not None:
        width, height = settings.width, settings.height
        if (width, height) != (manifest.width, manifest.height):
            size = (width, height)

    def prepare(path: Path) -> bytes:
        return prepare_frame(path.read_bytes(), size, settings.jpeg_quality)

    # OpenCV releases the GIL while decoding and encoding, so threads give a real speedup.
    with ThreadPoolExecutor(max_workers=workers or os.cpu_count() or 1) as pool:
        frames = tuple(pool.map(prepare, paths))
    return LoadedClip(id=clip_id, fps=manifest.fps, width=width, height=height, frames=frames)
