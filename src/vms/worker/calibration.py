"""INT8 calibration set (task 5.2): frames the quantizer runs through the FP32 model to record
each activation's range. No labels are needed, only images like the ones the model will see.

The set must be disjoint from the detection eval subset (D31), or the INT8 eval would be scored
on the images its ranges were fitted to: COCO train2017 (the eval uses val2017) and the MOT17
sequences that are not in the eval (05, 10, 11, 13; the eval uses 02, 04, 09). Both halves
matter: COCO covers the 80 classes the COCO mAP scores, the MOT frames cover the camera
scenes. MOT frames go through loadgen's resize and re-encode at use, as in the eval.

`configs/calibration.v1.yaml` is the selection rule; `build` resolves it once into the frozen
list `configs/calibration.v1.lock.jsonl` (archive member and sha256 per image, in git), and
`fetch` downloads the listed images on a fresh clone. A different selection is a new version.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from vms.common.logs import get_logger
from vms.datasets.http import ChecksumError, RetryPolicy, Transport
from vms.datasets.zipmembers import fetch_members, remote_zip_index
from vms.eval.detection.subset import mot_root, parse_seqinfo
from vms.loadgen.frames import prepare_frame
from vms.worker.service import decode_jpeg

log = get_logger(__name__)


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CocoCalibration(_Model):
    images_url: HttpUrl
    images_bytes: int = Field(gt=0)
    member_prefix: str  # e.g. "train2017/"
    images: int = Field(ge=0)


class MotCalibration(_Model):
    url: HttpUrl
    archive_bytes: int = Field(gt=0)
    sequences: list[str] = Field(min_length=1)
    frames_per_sequence: int = Field(ge=1)  # evenly spaced over the sequence


class FrameSize(_Model):
    width: int = Field(ge=16)
    height: int = Field(ge=16)
    jpeg_quality: int = Field(ge=1, le=100)


class CalibrationSpec(_Model):
    version: int = Field(ge=1)
    seed: int
    licenses: dict[str, dict[str, str]] = Field(default_factory=dict)
    coco: CocoCalibration
    mot17: MotCalibration
    mot_frame: FrameSize  # MOT frames as loadgen sends them (configs/loadgen.yaml)


class CalibImage(_Model):
    id: str
    source: str  # "coco" | "mot17"
    member: str
    sha256: str

    def path(self, images_dir: Path) -> Path:
        return images_dir / self.source / f"{self.id}{Path(self.member).suffix}"


def load_spec(path: Path) -> CalibrationSpec:
    return CalibrationSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def write_lock(path: Path, images: list[CalibImage]) -> str:
    """Write the frozen list (one JSON object per line, in build order); returns its sha256."""
    text = "".join(img.model_dump_json() + "\n" for img in images)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode()).hexdigest()


def read_lock(path: Path) -> list[CalibImage]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [CalibImage.model_validate(json.loads(line)) for line in lines if line]


def lock_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_members(names: list[str], n: int, seed: int) -> list[str]:
    """`n` names sampled uniformly without replacement, sorted."""
    if n > len(names):
        raise ValueError(f"asked for {n} images, only {len(names)} available")
    ordered = sorted(names)
    picks = np.random.default_rng(seed).choice(len(ordered), size=n, replace=False)
    return sorted(ordered[int(i)] for i in picks)


def spaced_frames(length: int, n: int) -> list[int]:
    """`n` distinct 1-based frame numbers spread evenly from the first to the last frame."""
    if n > length:
        raise ValueError(f"asked for {n} frames of a {length}-frame sequence")
    return sorted({round(f) for f in np.linspace(1, length, n)})


def _store(images_dir: Path, image: CalibImage, data: bytes) -> None:
    digest = hashlib.sha256(data).hexdigest()
    if image.sha256 and digest != image.sha256:
        raise ChecksumError(f"{image.member}: sha256 {digest} != expected {image.sha256}")
    path = image.path(images_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(data)
    partial.replace(path)


def image_ok(images_dir: Path, image: CalibImage) -> bool:
    path = image.path(images_dir)
    return path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == image.sha256


def build(
    spec: CalibrationSpec,
    transport: Transport,
    retry: RetryPolicy,
    images_dir: Path,
    workers: int = 8,
) -> list[CalibImage]:
    """Resolve the spec against the archives and download the images; returns the list."""
    out: list[CalibImage] = []
    if spec.coco.images:
        c = spec.coco
        url = str(c.images_url)
        index = remote_zip_index(transport, retry, url, c.images_bytes)
        names = [n for n in index if n.startswith(c.member_prefix) and n.lower().endswith(".jpg")]
        members = select_members(names, c.images, spec.seed)
        log.info("fetching COCO calibration images", extra={"images": len(members)})
        blobs = fetch_members(transport, retry, url, c.images_bytes, [index[m] for m in members])
        for m in members:
            image = CalibImage(
                id=f"coco-{Path(m).stem}",
                source="coco",
                member=m,
                sha256=hashlib.sha256(blobs[m]).hexdigest(),
            )
            _store(images_dir, image, blobs[m])
            out.append(image)

    m17 = spec.mot17
    url = str(m17.url)
    index = remote_zip_index(transport, retry, url, m17.archive_bytes)
    for sequence in m17.sequences:
        root = mot_root(index, sequence)
        meta = fetch_members(
            transport, retry, url, m17.archive_bytes, [index[root + "seqinfo.ini"]]
        )
        info = parse_seqinfo(meta[root + "seqinfo.ini"].decode())
        ext = info.get("imExt", ".jpg")
        frames = spaced_frames(int(info["seqLength"]), m17.frames_per_sequence)
        members = [f"{root}{info.get('imDir', 'img1')}/{f:06d}{ext}" for f in frames]
        log.info("fetching MOT17 calibration frames", extra={"sequence": sequence})
        blobs = fetch_members(
            transport, retry, url, m17.archive_bytes, [index[m] for m in members], workers
        )
        short = sequence.split("-")[1]
        for frame, member in zip(frames, members, strict=True):
            image = CalibImage(
                id=f"mot17-{short}-{frame:06d}",
                source="mot17",
                member=member,
                sha256=hashlib.sha256(blobs[member]).hexdigest(),
            )
            _store(images_dir, image, blobs[member])
            out.append(image)
    return out


def fetch(
    spec: CalibrationSpec,
    images: list[CalibImage],
    transport: Transport,
    retry: RetryPolicy,
    images_dir: Path,
    workers: int = 8,
) -> int:
    """Download the listed images that are missing or corrupt; returns how many were fetched."""
    missing = [img for img in images if not image_ok(images_dir, img)]
    archives = {
        "coco": (str(spec.coco.images_url), spec.coco.images_bytes),
        "mot17": (str(spec.mot17.url), spec.mot17.archive_bytes),
    }
    for source, (url, size) in archives.items():
        todo = [img for img in missing if img.source == source]
        if not todo:
            continue
        index = remote_zip_index(transport, retry, url, size)
        blobs = fetch_members(transport, retry, url, size, [index[i.member] for i in todo], workers)
        for img in todo:
            _store(images_dir, img, blobs[img.member])
    return len(missing)


def load_pixels(
    images: list[CalibImage], images_dir: Path, spec: CalibrationSpec
) -> Iterator[NDArray[np.uint8]]:
    """Each listed image as the detector gets it (BGR), checked against its sha256."""
    f = spec.mot_frame
    for image in images:
        data = image.path(images_dir).read_bytes()
        if hashlib.sha256(data).hexdigest() != image.sha256:
            raise ChecksumError(f"{image.path(images_dir)} does not match the calibration list")
        if image.source == "mot17":
            data = prepare_frame(data, (f.width, f.height), f.jpeg_quality)
        yield decode_jpeg(data)
