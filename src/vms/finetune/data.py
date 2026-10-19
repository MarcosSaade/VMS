"""The fine-tune training set (task 9.5): spec, frozen list, fetch and the YOLO-format dataset.

The images are labeled, so the list reuses the eval subset's machinery (`GtImage`, the COCO
and MOT17 readers in `vms.eval.detection.subset`) with other sources: COCO train2017 minus the
INT8 calibration images, and the MOT17 train sequences that are not in the eval. `build`
checks that no image, and no MOT17 sequence, is shared with the eval subset or a calibration
list, so neither the before/after mAP nor the INT8 ranges see a training image.

`materialize` writes one YOLO dataset per training run from the same list: MOT17 frames go
through loadgen's resize and re-encode first (as the eval and the worker see them), then a
seeded share of the images is blurred with `vms.drift.transforms` (the same pixels loadgen
and the drift eval use), each with its own sigma. The draw is per image id, so the control
run (no blur) and the blur-mix run see the same images in the same order with the same
augmentation stream except the blur. Ignore regions do not exist in YOLO training: crowd
boxes, ignore classes and barely visible persons are left unlabeled.
"""

from __future__ import annotations

import hashlib
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from pydantic import BaseModel, ConfigDict, Field

from vms.datasets.http import RetryPolicy, Transport
from vms.drift.transforms import apply as apply_drift
from vms.eval.detection.subset import (
    CocoSpec,
    GtImage,
    Mot17Spec,
    build_coco,
    build_mot17,
    fetch_images,
    read_subset,
)
from vms.eval.detection.subset import load_spec as load_eval_spec
from vms.loadgen.frames import prepare_frame
from vms.worker.calibration import read_lock as read_calibration_lock
from vms.worker.service import decode_jpeg

JPEG_QUALITY = 95  # training images written by materialize: near-lossless after the blur


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FrameSize(_Model):
    width: int = Field(ge=16)
    height: int = Field(ge=16)
    jpeg_quality: int = Field(ge=1, le=100)


class FinetuneDataSpec(_Model):
    version: int = Field(ge=1)
    seed: int
    licenses: dict[str, dict[str, str]] = Field(default_factory=dict)
    coco: CocoSpec
    member_prefix: str = "train2017/"
    exclude_locks: list[Path] = Field(default_factory=list)  # calibration lists to avoid
    mot17: Mot17Spec
    mot_frame: FrameSize


class Blur(_Model):
    """The share of training images blurred, and the range each one's sigma is drawn from
    (pixels at 1280 px width, as `vms.drift.transforms`)."""

    fraction: float = Field(ge=0, le=1)
    sigma_min: float = Field(default=1.0, gt=0)
    sigma_max: float = Field(default=8.0, gt=0)


class DisjointError(ValueError):
    """The training set shares an image or a sequence with the eval or calibration sets."""


def load_spec(path: Path) -> FinetuneDataSpec:
    return FinetuneDataSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def coco_id(member: str) -> int:
    """`train2017/000000123456.jpg` -> 123456."""
    return int(Path(member).stem)


def check_disjoint(
    images: list[GtImage],
    sequences: list[str],
    eval_images: list[GtImage],
    eval_sequences: list[str],
    calibration_members: list[str],
) -> None:
    """Raise unless the training list shares no archive member with the eval subset or the
    calibration list, and no MOT17 sequence with the eval (neighbouring frames of one
    sequence are near copies, so a disjoint frame list is not enough)."""

    def seq(name: str) -> str:
        return name.split("-")[1]  # "05" from MOT17-05-FRCNN

    shared_seq = {seq(s) for s in sequences} & {seq(s) for s in eval_sequences}
    if shared_seq:
        raise DisjointError(f"MOT17 sequences also in the eval: {sorted(shared_seq)}")
    members = {img.member for img in images}
    for name, other in (
        ("eval subset", {img.member for img in eval_images}),
        ("calibration list", set(calibration_members)),
    ):
        shared = members & other
        if shared:
            raise DisjointError(f"{len(shared)} images also in the {name}: {sorted(shared)[:3]}")


def build(
    spec: FinetuneDataSpec,
    transport: Transport,
    retry: RetryPolicy,
    images_dir: Path,
    eval_spec_path: Path,
    eval_subset_path: Path,
    workers: int = 8,
) -> tuple[list[GtImage], str]:
    """Select and fetch the training images; returns (sorted images, annotations sha256)."""
    calibration = [img for lock in spec.exclude_locks for img in read_calibration_lock(lock)]
    exclude = frozenset(coco_id(i.member) for i in calibration if i.source == "coco")
    coco, ann_hash = build_coco(
        spec.coco,
        spec.seed,
        transport,
        retry,
        images_dir,
        workers,
        member_prefix=spec.member_prefix,
        exclude_ids=exclude,
    )
    calibration_members = {i.member for i in calibration}
    mot = [
        i
        for i in build_mot17(spec.mot17, transport, retry, images_dir, workers)
        if i.member not in calibration_members  # the calibration list's frames are skipped
    ]
    images = sorted(coco + mot, key=lambda i: i.id)
    check_disjoint(
        images,
        spec.mot17.sequences,
        read_subset(eval_subset_path),
        load_eval_spec(eval_spec_path).mot17.sequences,
        sorted(calibration_members),
    )
    return images, ann_hash


def fetch(
    spec: FinetuneDataSpec,
    images: list[GtImage],
    transport: Transport,
    retry: RetryPolicy,
    images_dir: Path,
    workers: int = 8,
) -> int:
    return fetch_images(spec.coco, spec.mot17, images, transport, retry, images_dir, workers)


# --- The YOLO dataset ----------------------------------------------------------------------


def blur_draw(image_id: str, seed: int, blur: Blur) -> float | None:
    """The blur sigma for one image, or None to keep it clean. Seeded by the image id, so a
    run's draw does not depend on the order or the other images; two uniform draws are always
    taken, so changing `fraction` changes which images are blurred, not their sigmas."""
    rng = np.random.default_rng([seed, zlib.crc32(image_id.encode())])
    u, v = rng.random(2)
    if u >= blur.fraction:
        return None
    return float(blur.sigma_min + v * (blur.sigma_max - blur.sigma_min))


def yolo_labels(image: GtImage, width: int, height: int, class_index: dict[str, int]) -> list[str]:
    """`class cx cy w h` lines (normalized) for the positive boxes, scaled from the source
    size to (width, height). Unknown class names are an error: the head keeps COCO's 80."""
    sx, sy = width / image.width, height / image.height
    lines = []
    for box, cls, ignore in zip(image.boxes, image.classes, image.ignore, strict=True):
        if ignore:
            continue
        if cls not in class_index:
            raise KeyError(f"{image.id}: class {cls!r} is not one of the model's classes")
        x1, y1, x2, y2 = box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy
        x1, x2 = min(max(x1, 0.0), width), min(max(x2, 0.0), width)
        y1, y2 = min(max(y1, 0.0), height), min(max(y2, 0.0), height)
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        cx, cy = (x1 + x2) / 2 / width, (y1 + y2) / 2 / height
        w, h = (x2 - x1) / width, (y2 - y1) / height
        lines.append(f"{class_index[cls]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return lines


def materialize(
    images: list[GtImage],
    images_dir: Path,
    out_dir: Path,
    spec: FinetuneDataSpec,
    names: dict[int, str],
    blur: Blur,
    val_images: int,
) -> dict[str, Any]:
    """Write a YOLO dataset (images/train, labels/train, val.txt, data.yaml) under `out_dir`;
    returns its summary (counts, blurred images and their sigmas, data.yaml path). The first
    `val_images` images (by id, clean copies) form Ultralytics' validation list, a sanity
    readout only: it is drawn from the training images and is never used to choose a model."""
    img_dir, lbl_dir = out_dir / "images/train", out_dir / "labels/train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    class_index = {name: idx for idx, name in names.items()}
    f = spec.mot_frame
    sigmas: dict[str, float] = {}
    boxes: Counter[str] = Counter()
    digest = hashlib.sha256()
    for image in images:
        data = image.path(images_dir).read_bytes()
        if hashlib.sha256(data).hexdigest() != image.sha256:
            raise ValueError(f"{image.path(images_dir)} does not match the training list")
        if image.source == "mot17":
            data = prepare_frame(data, (f.width, f.height), f.jpeg_quality)
        pixels = decode_jpeg(data)
        height, width = pixels.shape[:2]
        sigma = blur_draw(image.id, spec.seed, blur)
        if sigma is not None:
            pixels = apply_drift("blur", pixels, 1.0, sigma)
            sigmas[image.id] = round(sigma, 4)
        ok, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        if not ok:
            raise RuntimeError(f"JPEG encode failed for {image.id}")
        (img_dir / f"{image.id}.jpg").write_bytes(encoded.tobytes())
        labels = yolo_labels(image, width, height, class_index)
        text = "".join(line + "\n" for line in labels)
        (lbl_dir / f"{image.id}.txt").write_text(text, encoding="utf-8")
        boxes[image.source] += len(labels)
        digest.update(image.id.encode() + encoded.tobytes() + text.encode())
    val = [str((img_dir / f"{img.id}.jpg").resolve()) for img in images[:val_images]]
    (out_dir / "val.txt").write_text("".join(v + "\n" for v in val), encoding="utf-8")
    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(out_dir.resolve()),
                "train": "images/train",
                "val": str((out_dir / "val.txt").resolve()),
                "names": dict(sorted(names.items())),
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    per_source = Counter(img.source for img in images)
    return {
        "data_yaml": str(data_yaml),
        "images": dict(per_source),
        "boxes": dict(boxes),
        "blurred": len(sigmas),
        "blurred_by_source": dict(Counter(i.split("-")[0] for i in sigmas)),
        "sigma_mean": float(np.mean(list(sigmas.values()))) if sigmas else None,
        "blur": blur.model_dump(),
        "val_images": len(val),
        "dataset_sha256": digest.hexdigest(),
        "sigmas": sigmas,
    }
