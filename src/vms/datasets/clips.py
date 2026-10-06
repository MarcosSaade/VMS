"""Turn a source into a normalized clip: JPEG frames plus a `clip.json` manifest entry.

Output layout, under the data directory:

    clips/<clip_id>/frames/000001.jpg ...   normalized frames, consecutive from 1
    clips/<clip_id>/clip.json               ClipManifest; written last, so it marks completion
    raw/<clip_id>/<file>                    cached source video files (video sources only)

Normalization: frames are downscaled to at most `max_height` (aspect kept, never upscaled),
sources above `max_fps` drop frames evenly, and frames are re-encoded as JPEG at
`jpeg_quality`. This keeps every virtual camera close to the workload's 720p JPEGs
(ARCHITECTURE 2) and makes clips from different datasets uniform.
"""

from __future__ import annotations

import hashlib
import io
import shutil
import zipfile
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, ConfigDict

from vms.common.logs import get_logger
from vms.datasets.http import (
    ChecksumError,
    RemoteFile,
    RetryPolicy,
    Transport,
    download_file,
)
from vms.datasets.spec import (
    Clip,
    Processing,
    Scene,
    VideoFileSource,
    ZipSequenceSource,
    clip_spec_hash,
)

log = get_logger(__name__)

MANIFEST_NAME = "clip.json"
FRAMES_DIR = "frames"
# A span may include unrelated members between the ones we need (a trimmed sequence is
# interleaved with its other frames); above this much overhead,
# members are fetched one by one instead of as one span.
MAX_SPAN_OVERHEAD = 4.0


class ClipManifest(BaseModel):
    """What a fetched clip contains. Loadgen reads this; the committed manifest collects it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    title: str
    license: str
    attribution: str
    page_url: str
    source_url: str
    scene: Scene
    fps: float
    frames: int
    duration_s: float
    width: int
    height: int
    mean_luma: float  # 0-255, over all frames; a reference point for brightness drift
    spec_hash: str  # clip_spec_hash: source + processing
    content_sha256: str  # over frame names and bytes, in order


def clip_dir(data_dir: Path, clip_id: str) -> Path:
    return data_dir / "clips" / clip_id


def read_manifest(data_dir: Path, clip_id: str) -> ClipManifest | None:
    path = clip_dir(data_dir, clip_id) / MANIFEST_NAME
    if not path.is_file():
        return None
    return ClipManifest.model_validate_json(path.read_text(encoding="utf-8"))


def is_current(data_dir: Path, clip: Clip, processing: Processing) -> bool:
    manifest = read_manifest(data_dir, clip.id)
    return manifest is not None and manifest.spec_hash == clip_spec_hash(clip, processing)


# --- Frame normalization ---------------------------------------------------------------------


def normalize(image: np.ndarray, processing: Processing) -> np.ndarray:
    height, width = image.shape[:2]
    if height <= processing.max_height:
        return image
    scale = processing.max_height / height
    size = (round(width * scale), processing.max_height)
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def encode_jpeg(image: np.ndarray, quality: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return bytes(buf)


def decode_jpeg(data: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("not a decodable image")
    return image


def frame_step(source_fps: float, max_fps: float) -> int:
    """Keep every n-th frame so the output rate is at most max_fps (integer step, no blending)."""
    step = 1
    while source_fps / step > max_fps + 1e-9:
        step += 1
    return step


# --- Sources ---------------------------------------------------------------------------------


def members_digest(members: list[tuple[str, bytes]]) -> str:
    """sha256 over (name, bytes) pairs in name order: pins a zip sequence's source bytes."""
    digest = hashlib.sha256()
    for name, data in sorted(members):
        digest.update(name.encode() + b"\0" + len(data).to_bytes(8, "big") + data)
    return "sha256:" + digest.hexdigest()


def _frame_number(name: str, prefix: str) -> int | None:
    stem = name[len(prefix) :].rsplit(".", 1)[0]
    return int(stem) if stem.isdigit() else None


def read_zip_members(
    source: ZipSequenceSource, transport: Transport, retry: RetryPolicy, spool_dir: Path
) -> list[tuple[str, bytes]]:
    """The selected members of a zip sequence, in frame order, NOT yet checked against the
    pinned digest (zipfile does check each member's CRC32 as it reads)."""
    url = str(source.url)
    size = transport.size(url)
    if size != source.archive_bytes:
        raise ChecksumError(f"{url}: archive is {size} bytes, spec pins {source.archive_bytes}")
    remote = RemoteFile(transport, retry, url, size)
    with zipfile.ZipFile(io.BufferedReader(remote, buffer_size=1 << 16)) as archive:
        selected: list[zipfile.ZipInfo] = []
        for info in archive.infolist():
            if info.is_dir() or not info.filename.startswith(source.frames_prefix):
                continue
            number = _frame_number(info.filename, source.frames_prefix)
            if number is None or number < source.first_frame:
                continue
            if source.last_frame is not None and number > source.last_frame:
                continue
            selected.append(info)
        if not selected:
            raise ChecksumError(f"{url}: no members under {source.frames_prefix}")
        selected.sort(key=lambda i: i.header_offset)
        start = selected[0].header_offset
        # Local headers are 30 bytes plus name and extra field; leave room for both.
        end = selected[-1].header_offset + 30 + 1024 + selected[-1].compress_size
        needed = sum(i.compress_size for i in selected)
        if end - start <= needed * MAX_SPAN_OVERHEAD:
            remote.prefetch(start, end, spool_dir)
        members = [(i.filename, archive.read(i)) for i in selected]
    remote.close()
    members.sort(key=lambda m: _frame_number(m[0], source.frames_prefix) or 0)
    return members


def read_zip_sequence(
    source: ZipSequenceSource, transport: Transport, retry: RetryPolicy, spool_dir: Path
) -> list[tuple[str, bytes]]:
    """The selected frames of a zip sequence, verified against the pinned digest."""
    members = read_zip_members(source, transport, retry, spool_dir)
    digest = members_digest(members)
    if digest != source.checksum:
        raise ChecksumError(
            f"{source.url} {source.frames_prefix}: {digest} != expected {source.checksum}"
        )
    return members


def iter_zip_frames(
    members: list[tuple[str, bytes]], source_fps: float, processing: Processing
) -> Iterator[np.ndarray]:
    step = frame_step(source_fps, processing.max_fps)
    for index, (_, data) in enumerate(members):
        if index % step == 0:
            yield normalize(decode_jpeg(data), processing)


# Probed frame rates outside this range are treated as container errors.
PLAUSIBLE_FPS = (1.0, 120.0)


def iter_video_frames(
    path: Path, source: VideoFileSource, processing: Processing, fps: float
) -> Iterator[np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"{path}: cannot open video")
    try:
        first = round(source.start_s * fps)
        last = None if source.duration_s is None else first + round(source.duration_s * fps) - 1
        step = frame_step(fps, processing.max_fps)
        index = 0
        # Decode sequentially rather than seeking: seeks land on keyframes and are not exact
        # or reproducible across decoders.
        while True:
            ok, image = capture.read()
            if not ok or (last is not None and index > last):
                break
            if index >= first and (index - first) % step == 0:
                yield normalize(image, processing)
            index += 1
    finally:
        capture.release()


def video_fps(path: Path) -> float:
    capture = cv2.VideoCapture(str(path))
    try:
        return float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()


# --- Writing ---------------------------------------------------------------------------------


def write_clip(
    data_dir: Path,
    clip: Clip,
    processing: Processing,
    frames: Iterator[np.ndarray],
    fps: float,
) -> ClipManifest:
    """Write frames to a temporary directory, then swap it in, so a partial clip never
    appears complete."""
    final = clip_dir(data_dir, clip.id)
    staging = final.with_name(final.name + ".staging")
    shutil.rmtree(staging, ignore_errors=True)
    (staging / FRAMES_DIR).mkdir(parents=True)

    digest = hashlib.sha256()
    count, luma_sum, size = 0, 0.0, (0, 0)
    for image in frames:
        count += 1
        name = f"{count:06d}.jpg"
        data = encode_jpeg(image, processing.jpeg_quality)
        (staging / FRAMES_DIR / name).write_bytes(data)
        digest.update(name.encode() + b"\0" + len(data).to_bytes(8, "big") + data)
        luma_sum += float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())
        size = (image.shape[1], image.shape[0])
    if count == 0:
        shutil.rmtree(staging)
        raise ValueError(f"clip {clip.id}: no frames produced")

    manifest = ClipManifest(
        id=clip.id,
        title=clip.title,
        license=clip.license,
        attribution=clip.attribution,
        page_url=str(clip.page_url),
        source_url=str(clip.source.url),
        scene=clip.scene,
        fps=round(fps, 3),
        frames=count,
        duration_s=round(count / fps, 3),
        width=size[0],
        height=size[1],
        mean_luma=round(luma_sum / count, 1),
        spec_hash=clip_spec_hash(clip, processing),
        content_sha256="sha256:" + digest.hexdigest(),
    )
    (staging / MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2) + "\n", "utf-8")
    shutil.rmtree(final, ignore_errors=True)
    staging.rename(final)
    return manifest


def fetch_clip(
    data_dir: Path,
    clip: Clip,
    processing: Processing,
    transport: Transport,
    retry: RetryPolicy,
) -> ClipManifest:
    """Fetch, verify and normalize one clip. Callers skip clips that are already current."""
    source = clip.source
    if isinstance(source, ZipSequenceSource):
        spool = data_dir / "raw"
        spool.mkdir(parents=True, exist_ok=True)
        members = read_zip_sequence(source, transport, retry, spool)
        step = frame_step(source.fps, processing.max_fps)
        frames = iter_zip_frames(members, source.fps, processing)
        return write_clip(data_dir, clip, processing, frames, source.fps / step)

    filename = str(source.url).rsplit("/", 1)[-1].split("?")[0] or "video"
    raw = download_file(
        transport,
        retry,
        str(source.url),
        data_dir / "raw" / clip.id / filename,
        source.checksum,
        source.file_bytes,
    )
    fps = source.fps
    if fps is None:
        fps = video_fps(raw)
        low, high = PLAUSIBLE_FPS
        if not low <= fps <= high:
            raise ValueError(
                f"clip {clip.id}: implausible frame rate {fps} from the container; "
                "set source.fps in the spec"
            )
    step = frame_step(fps, processing.max_fps)
    return write_clip(
        data_dir, clip, processing, iter_video_frames(raw, source, processing, fps), fps / step
    )
