"""The frozen detection eval subset: spec, ground truth, build and fetch.

Two sources, scored separately because their labels follow different conventions:

- `coco`: images sampled from COCO val2017 with all 80 classes, scored with the COCO protocol
  (crowd boxes are ignore regions). Comparable with published YOLO11 numbers.
- `mot17`: one frame per second of the static MOT17 train sequences that are also demo cameras,
  person boxes only, with MOT's distractor classes and barely visible pedestrians as ignore
  regions.

`build` (run once per version) selects the images, reads their ground truth from the archives
and writes `subset.v<N>.jsonl`, one image per line with its archive member, sha256, size and
boxes in source pixels, plus a manifest with hashes. Both are committed; the images are not.
`fetch` downloads exactly the listed members on a fresh clone and verifies each sha256.
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from vms.common.logs import get_logger
from vms.datasets.http import ChecksumError, RetryPolicy, Transport
from vms.datasets.zipmembers import fetch_members, remote_zip_index

log = get_logger(__name__)

SOURCES = ("coco", "mot17")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CocoSpec(_Model):
    images_url: HttpUrl
    images_bytes: int = Field(gt=0)
    annotations_url: HttpUrl
    annotations_bytes: int = Field(gt=0)
    annotations_member: str
    annotations_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    images: int = Field(ge=1)


class Mot17Spec(_Model):
    url: HttpUrl
    archive_bytes: int = Field(gt=0)
    sequences: list[str] = Field(min_length=1)
    frame_stride: int = Field(ge=1)
    positive_classes: list[int] = Field(min_length=1)
    ignore_classes: list[int] = Field(default_factory=list)
    crowd_classes: list[int] = Field(default_factory=list)
    min_visibility: float = Field(ge=0, le=1)


class SubsetSpec(_Model):
    version: int = Field(ge=1)
    seed: int
    licenses: dict[str, dict[str, Any]]
    coco: CocoSpec
    mot17: Mot17Spec


def load_spec(path: Path) -> SubsetSpec:
    return SubsetSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def spec_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GtImage(_Model):
    """One image of the subset with its ground truth, boxes in source pixels (x1, y1, x2, y2).

    `ignore` marks regions a detection may hit without counting as a true or false positive;
    `crowd` ignore regions may absorb any number of detections (COCO's iscrowd)."""

    id: str
    source: str  # "coco" | "mot17"
    member: str  # archive member name
    sha256: str
    width: int
    height: int
    boxes: list[tuple[float, float, float, float]]
    classes: list[str]
    ignore: list[bool]
    crowd: list[bool]

    def path(self, images_dir: Path) -> Path:
        return images_dir / self.source / f"{self.id}{Path(self.member).suffix}"


class SubsetManifest(_Model):
    version: int
    spec_sha256: str  # of the spec file the subset was built from
    subset_sha256: str  # of the subset JSONL
    coco_annotations_sha256: str
    seed: int
    images: dict[str, int]  # per source
    boxes: dict[str, int]  # positives per source
    ignored: dict[str, int]  # ignore regions per source


class UnpinnedError(RuntimeError):
    """The spec does not pin the annotations yet; carries the sha256 to pin."""

    def __init__(self, sha256: str) -> None:
        super().__init__(f"annotations not pinned; sha256 is {sha256}")
        self.sha256 = sha256


# --- Ground truth ----------------------------------------------------------------------------


def select_coco_ids(image_ids: list[int], n: int, seed: int) -> list[int]:
    """`n` image ids sampled uniformly without replacement, in id order."""
    if n > len(image_ids):
        raise ValueError(f"asked for {n} images, only {len(image_ids)} available")
    ordered = sorted(image_ids)
    picks = np.random.default_rng(seed).choice(len(ordered), size=n, replace=False)
    return sorted(ordered[int(i)] for i in picks)


def coco_ground_truth(
    annotations: dict[str, Any], image_ids: list[int], member_prefix: str = "val2017/"
) -> list[GtImage]:
    """Ground truth for `image_ids` from a COCO instances file. sha256 is filled in later.
    `member_prefix` is the images archive's folder (train2017/ for the fine-tune set, 9.5)."""
    names = {c["id"]: c["name"] for c in annotations["categories"]}
    by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for ann in annotations["annotations"]:
        by_image[ann["image_id"]].append(ann)
    images = {img["id"]: img for img in annotations["images"]}
    out = []
    for image_id in image_ids:
        img = images[image_id]
        anns = sorted(by_image.get(image_id, []), key=lambda a: a["id"])
        boxes = [
            (float(x), float(y), float(x + w), float(y + h))
            for x, y, w, h in (a["bbox"] for a in anns)
        ]
        crowd = [bool(a.get("iscrowd", 0)) for a in anns]
        out.append(
            GtImage(
                id=f"coco-{image_id:012d}",
                source="coco",
                member=f"{member_prefix}{img['file_name']}",
                sha256="",
                width=int(img["width"]),
                height=int(img["height"]),
                boxes=boxes,
                classes=[names[a["category_id"]] for a in anns],
                ignore=crowd,
                crowd=crowd,
            )
        )
    return out


@dataclass(frozen=True)
class MotBox:
    frame: int
    box: tuple[float, float, float, float]
    cls: int
    flag: int  # gt.txt column 7: 1 = considered by the benchmark, 0 = not
    visibility: float


def parse_mot_gt(text: str) -> list[MotBox]:
    """gt.txt rows: frame, id, left, top, width, height, flag, class, visibility."""
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        f = line.split(",")
        left, top, w, h = (float(v) for v in f[2:6])
        # MOT boxes are 1-based; convert to 0-based pixel corners like COCO.
        rows.append(
            MotBox(
                frame=int(f[0]),
                box=(left - 1, top - 1, left - 1 + w, top - 1 + h),
                cls=int(f[7]),
                flag=int(float(f[6])),
                visibility=float(f[8]),
            )
        )
    return rows


def mot_frame_ground_truth(
    rows: list[MotBox], frame: int, width: int, height: int, spec: Mot17Spec
) -> tuple[list[tuple[float, float, float, float]], list[bool], list[bool]]:
    """Person boxes of one frame, clipped to the image: (boxes, ignore, crowd).

    Persons below `min_visibility` become ignore regions: a detector that finds a mostly hidden
    person is not penalized, and one that misses it is not either. Vehicles and occluders are
    dropped (only persons are scored). The flag column is not used: in MOT17 it is 1 exactly
    for class 1, so it would only repeat the benchmark's class choice."""
    boxes, ignore, crowd = [], [], []
    for r in rows:
        if r.frame != frame:
            continue
        positive = r.cls in spec.positive_classes
        if positive and r.visibility >= spec.min_visibility:
            is_ignore, is_crowd = False, False
        elif positive or r.cls in spec.ignore_classes:
            is_ignore, is_crowd = True, False
        elif r.cls in spec.crowd_classes:
            is_ignore, is_crowd = True, True
        else:
            continue
        x1, y1, x2, y2 = r.box
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(float(width), x2), min(float(height), y2)
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2, y2))
        ignore.append(is_ignore)
        crowd.append(is_crowd)
    return boxes, ignore, crowd


def parse_seqinfo(text: str) -> dict[str, str]:
    out = {}
    for line in text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


# --- Files -----------------------------------------------------------------------------------


def write_subset(path: Path, images: list[GtImage]) -> str:
    """One image per line, sorted by id, compact JSON; returns the file's sha256."""
    lines = [img.model_dump_json() for img in sorted(images, key=lambda i: i.id)]
    data = ("\n".join(lines) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def read_subset(path: Path) -> list[GtImage]:
    with path.open(encoding="utf-8") as fh:
        return [GtImage.model_validate_json(line) for line in fh if line.strip()]


def summarize(
    images: list[GtImage], spec: SubsetSpec, spec_hash: str, subset_hash: str, ann_hash: str
) -> SubsetManifest:
    count: dict[str, int] = defaultdict(int)
    boxes: dict[str, int] = defaultdict(int)
    ignored: dict[str, int] = defaultdict(int)
    for img in images:
        count[img.source] += 1
        boxes[img.source] += sum(not i for i in img.ignore)
        ignored[img.source] += sum(img.ignore)
    return SubsetManifest(
        version=spec.version,
        spec_sha256=spec_hash,
        subset_sha256=subset_hash,
        coco_annotations_sha256=ann_hash,
        seed=spec.seed,
        images=dict(count),
        boxes=dict(boxes),
        ignored=dict(ignored),
    )


def store_image(images_dir: Path, image: GtImage, data: bytes) -> None:
    digest = hashlib.sha256(data).hexdigest()
    if image.sha256 and digest != image.sha256:
        raise ChecksumError(f"{image.member}: sha256 {digest} != expected {image.sha256}")
    path = image.path(images_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_bytes(data)
    partial.replace(path)


def image_ok(images_dir: Path, image: GtImage) -> bool:
    path = image.path(images_dir)
    return path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == image.sha256


# --- Build and fetch -------------------------------------------------------------------------


def build(
    spec: SubsetSpec,
    spec_hash: str,
    transport: Transport,
    retry: RetryPolicy,
    images_dir: Path,
    workers: int = 8,
) -> tuple[list[GtImage], str]:
    """Select images, read their ground truth and fetch them. Returns (images with sha256,
    annotations sha256). Images are written to `images_dir`."""
    coco, ann_hash = build_coco(spec.coco, spec.seed, transport, retry, images_dir, workers)
    mot = build_mot17(spec.mot17, transport, retry, images_dir, workers)
    return coco + mot, ann_hash


def build_coco(
    c: CocoSpec,
    seed: int,
    transport: Transport,
    retry: RetryPolicy,
    images_dir: Path,
    workers: int,
    member_prefix: str = "val2017/",
    exclude_ids: frozenset[int] = frozenset(),
) -> tuple[list[GtImage], str]:
    """Sample `c.images` images (never one of `exclude_ids`), read their ground truth and
    fetch them. Returns (images with sha256, annotations sha256)."""
    ann_url = str(c.annotations_url)
    index = remote_zip_index(transport, retry, ann_url, c.annotations_bytes)
    if c.annotations_member not in index:
        raise ChecksumError(f"{ann_url}: no member {c.annotations_member}")
    log.info("reading COCO annotations", extra={"member": c.annotations_member})
    raw = fetch_members(
        transport, retry, ann_url, c.annotations_bytes, [index[c.annotations_member]]
    )
    data = raw[c.annotations_member]
    ann_hash = hashlib.sha256(data).hexdigest()
    if c.annotations_sha256 is None:
        raise UnpinnedError(ann_hash)
    if ann_hash != c.annotations_sha256:
        raise ChecksumError(f"{c.annotations_member}: sha256 {ann_hash} != {c.annotations_sha256}")
    annotations = json.loads(data)
    candidates = [img["id"] for img in annotations["images"] if img["id"] not in exclude_ids]
    ids = select_coco_ids(candidates, c.images, seed)
    gts = coco_ground_truth(annotations, ids, member_prefix)

    url = str(c.images_url)
    img_index = remote_zip_index(transport, retry, url, c.images_bytes)
    log.info("fetching COCO images", extra={"images": len(gts)})
    blobs = fetch_members(
        transport, retry, url, c.images_bytes, [img_index[g.member] for g in gts], workers
    )
    out = []
    for gt in gts:
        image = gt.model_copy(update={"sha256": hashlib.sha256(blobs[gt.member]).hexdigest()})
        store_image(images_dir, image, blobs[gt.member])
        out.append(image)
    return out, ann_hash


def mot_root(index: dict[str, zipfile.ZipInfo], sequence: str) -> str:
    for split in ("train", "test"):
        root = f"MOT17/{split}/{sequence}/"
        if root + "seqinfo.ini" in index:
            return root
    raise ChecksumError(f"MOT17 archive has no sequence {sequence}")


def build_mot17(
    spec: Mot17Spec, transport: Transport, retry: RetryPolicy, images_dir: Path, workers: int
) -> list[GtImage]:
    url = str(spec.url)
    index = remote_zip_index(transport, retry, url, spec.archive_bytes)
    out = []
    for sequence in spec.sequences:
        root = mot_root(index, sequence)
        meta = fetch_members(
            transport,
            retry,
            url,
            spec.archive_bytes,
            [index[root + "seqinfo.ini"], index[root + "gt/gt.txt"]],
        )
        info = parse_seqinfo(meta[root + "seqinfo.ini"].decode())
        width, height, length = int(info["imWidth"]), int(info["imHeight"]), int(info["seqLength"])
        ext = info.get("imExt", ".jpg")
        rows = parse_mot_gt(meta[root + "gt/gt.txt"].decode())
        frames = list(range(1, length + 1, spec.frame_stride))
        members = [f"{root}{info.get('imDir', 'img1')}/{f:06d}{ext}" for f in frames]
        log.info("fetching MOT17 frames", extra={"sequence": sequence, "frames": len(frames)})
        blobs = fetch_members(
            transport, retry, url, spec.archive_bytes, [index[m] for m in members], workers
        )
        short = sequence.split("-")[1]  # "02" from MOT17-02-FRCNN
        for frame, member in zip(frames, members, strict=True):
            boxes, ignore, crowd = mot_frame_ground_truth(rows, frame, width, height, spec)
            image = GtImage(
                id=f"mot17-{short}-{frame:06d}",
                source="mot17",
                member=member,
                sha256=hashlib.sha256(blobs[member]).hexdigest(),
                width=width,
                height=height,
                boxes=boxes,
                classes=["person"] * len(boxes),
                ignore=ignore,
                crowd=crowd,
            )
            store_image(images_dir, image, blobs[member])
            out.append(image)
    return out


def fetch(
    spec: SubsetSpec,
    images: list[GtImage],
    transport: Transport,
    retry: RetryPolicy,
    images_dir: Path,
    workers: int = 8,
) -> int:
    """Download the listed images that are missing or corrupt; returns how many were fetched."""
    return fetch_images(spec.coco, spec.mot17, images, transport, retry, images_dir, workers)


def fetch_images(
    coco: CocoSpec,
    mot17: Mot17Spec,
    images: list[GtImage],
    transport: Transport,
    retry: RetryPolicy,
    images_dir: Path,
    workers: int = 8,
) -> int:
    """`fetch` for any list of COCO and MOT17 images (the eval subset, the fine-tune set)."""
    missing = [img for img in images if not image_ok(images_dir, img)]
    archives = {
        "coco": (str(coco.images_url), coco.images_bytes),
        "mot17": (str(mot17.url), mot17.archive_bytes),
    }
    for source, (url, size) in archives.items():
        todo = [img for img in missing if img.source == source]
        if not todo:
            continue
        index = remote_zip_index(transport, retry, url, size)
        log.info("fetching images", extra={"source": source, "images": len(todo)})
        blobs = fetch_members(transport, retry, url, size, [index[i.member] for i in todo], workers)
        for img in todo:
            store_image(images_dir, img, blobs[img.member])
    return len(missing)
