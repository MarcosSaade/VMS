"""The 9.5 training set: seeded blur draw, YOLO labels, disjointness from the eval and
calibration sets, build and fetch with a fake transport, the materialized dataset, and the
committed v1 list."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from tests.unit.datasets.conftest import FakeTransport, jpeg, make_zip
from vms.datasets.http import RetryPolicy
from vms.eval.detection.subset import GtImage, image_ok, read_subset
from vms.eval.detection.subset import load_spec as load_eval_spec
from vms.finetune import data
from vms.worker.calibration import read_lock

REPO_ROOT = Path(__file__).resolve().parents[3]
COCO_IMAGES = "https://example.test/train2017.zip"
COCO_ANNS = "https://example.test/annotations.zip"
MOT_URL = "https://example.test/MOT17.zip"
EVAL_SPEC = REPO_ROOT / "configs/eval/detection.v1.yaml"
EVAL_SUBSET = REPO_ROOT / "eval/detection/subset.v1.jsonl"
NAMES = {0: "person", 1: "bicycle", 2: "car"}


@pytest.fixture
def retry() -> RetryPolicy:
    return RetryPolicy(attempts=2, base_s=0.01, sleep=lambda _: None)


def gt(
    id_: str = "coco-000000000001",
    source: str = "coco",
    boxes: list[tuple[float, float, float, float]] | None = None,
    classes: list[str] | None = None,
    ignore: list[bool] | None = None,
    width: int = 100,
    height: int = 50,
    member: str = "train2017/000000000001.jpg",
) -> GtImage:
    boxes = boxes if boxes is not None else [(10.0, 5.0, 30.0, 25.0)]
    return GtImage(
        id=id_,
        source=source,
        member=member,
        sha256="",
        width=width,
        height=height,
        boxes=boxes,
        classes=classes or ["person"] * len(boxes),
        ignore=ignore or [False] * len(boxes),
        crowd=[False] * len(boxes),
    )


def test_blur_draw_is_seeded_per_image_and_fraction_only_picks_which() -> None:
    half = data.Blur(fraction=0.5, sigma_min=1.0, sigma_max=8.0)
    ids = [f"coco-{i:012d}" for i in range(400)]
    draws = [data.blur_draw(i, 7, half) for i in ids]
    assert draws == [data.blur_draw(i, 7, half) for i in ids]  # deterministic
    assert draws != [data.blur_draw(i, 8, half) for i in ids]  # the seed matters
    sigmas = [d for d in draws if d is not None]
    assert 160 < len(sigmas) < 240
    assert all(1.0 <= s <= 8.0 for s in sigmas)
    assert [data.blur_draw(i, 7, data.Blur(fraction=0.0)) for i in ids] == [None] * 400
    every = [
        data.blur_draw(i, 7, data.Blur(fraction=1.0, sigma_min=1.0, sigma_max=8.0)) for i in ids
    ]
    # The images blurred at 0.5 keep their sigma when every image is blurred.
    assert all(e == d for e, d in zip(every, draws, strict=True) if d is not None)


def test_yolo_labels_normalize_scale_clip_and_skip_ignored() -> None:
    image = gt(
        boxes=[(10, 5, 30, 25), (90, 40, 120, 60), (0, 0, 50, 50), (5, 5, 5.2, 6)],
        classes=["person", "car", "person", "bicycle"],
        ignore=[False, False, True, False],
    )
    lines = data.yolo_labels(image, 200, 100, {v: k for k, v in NAMES.items()})
    # x2 scale; the car is clipped to the image; the ignored box and the sub-pixel box go.
    assert lines == [
        "0 0.200000 0.300000 0.200000 0.400000",
        "2 0.950000 0.900000 0.100000 0.200000",
    ]
    with pytest.raises(KeyError, match="zebra"):
        data.yolo_labels(gt(classes=["zebra"]), 100, 50, {"person": 0})


def test_check_disjoint_rejects_shared_images_and_sequences() -> None:
    train = [gt(), gt("mot17-05-000001", "mot17", member="MOT17/train/MOT17-05-FRCNN/img1/1.jpg")]
    evals = [gt("x", member="val2017/000000000001.jpg")]
    data.check_disjoint(train, ["MOT17-05-FRCNN"], evals, ["MOT17-02-FRCNN"], ["train2017/9.jpg"])
    with pytest.raises(data.DisjointError, match="sequences"):
        data.check_disjoint(train, ["MOT17-02-SDP"], evals, ["MOT17-02-FRCNN"], [])
    with pytest.raises(data.DisjointError, match="eval subset"):
        data.check_disjoint(train, [], [gt()], [], [])
    with pytest.raises(data.DisjointError, match="calibration"):
        data.check_disjoint(train, [], evals, [], ["train2017/000000000001.jpg"])


def textured(width: int, height: int, seed: int) -> bytes:
    """A JPEG with detail, so a blur changes it (a flat colour stays the same)."""
    pixels = np.random.default_rng(seed).integers(0, 255, (height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", pixels)
    assert ok
    return bytes(buf)


def fake_sources(tmp_path: Path) -> tuple[data.FinetuneDataSpec, FakeTransport]:
    annotations = {
        "images": [
            {"id": i, "file_name": f"{i:012d}.jpg", "width": 32, "height": 24} for i in range(1, 7)
        ],
        "categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}],
        "annotations": [
            {
                "id": i,
                "image_id": i,
                "category_id": 1 + 2 * (i % 2),
                "bbox": [2, 2, 8, 8],
                "iscrowd": int(i == 4),
            }
            for i in range(1, 7)
        ],
    }
    ann_bytes = json.dumps(annotations).encode()
    anns_zip = make_zip({"annotations/instances_train2017.json": ann_bytes})
    images_zip = make_zip({f"train2017/{i:012d}.jpg": textured(32, 24, i) for i in range(1, 7)})
    seq = "MOT17/train/MOT17-05-FRCNN/"
    mot_zip = make_zip(
        {
            seq + "seqinfo.ini": b"[Sequence]\nimDir=img1\nseqLength=6\nimWidth=64\nimHeight=48\n"
            b"imExt=.jpg\n",
            seq + "gt/gt.txt": b"1,1,1,1,10,10,1,1,1.0\n1,2,20,20,10,10,1,1,0.1\n",
            **{f"{seq}img1/{f:06d}.jpg": jpeg(64, 48, 40 * f) for f in range(1, 7)},
        }
    )
    lock = tmp_path / "calibration.lock.jsonl"
    calibration = [
        {"id": "coco-x", "source": "coco", "member": "train2017/000000000002.jpg"},
        {"id": "mot17-05-000003", "source": "mot17", "member": f"{seq}img1/000003.jpg"},
    ]
    lock.write_text("".join(json.dumps(c | {"sha256": "0" * 64}) + "\n" for c in calibration))
    spec = data.FinetuneDataSpec.model_validate(
        {
            "version": 1,
            "seed": 3,
            "coco": {
                "images_url": COCO_IMAGES,
                "images_bytes": len(images_zip),
                "annotations_url": COCO_ANNS,
                "annotations_bytes": len(anns_zip),
                "annotations_member": "annotations/instances_train2017.json",
                "annotations_sha256": hashlib.sha256(ann_bytes).hexdigest(),
                "images": 5,
            },
            "member_prefix": "train2017/",
            "exclude_locks": [str(lock)],
            "mot17": {
                "url": MOT_URL,
                "archive_bytes": len(mot_zip),
                "sequences": ["MOT17-05-FRCNN"],
                "frame_stride": 2,
                "positive_classes": [1],
                "min_visibility": 0.25,
            },
            "mot_frame": {"width": 128, "height": 96, "jpeg_quality": 85},
        }
    )
    transport = FakeTransport({COCO_IMAGES: images_zip, COCO_ANNS: anns_zip, MOT_URL: mot_zip})
    return spec, transport


def test_build_excludes_calibration_images_then_fetch_and_materialize(
    tmp_path: Path, retry: RetryPolicy
) -> None:
    spec, transport = fake_sources(tmp_path)
    images, _ = data.build(spec, transport, retry, tmp_path / "a", EVAL_SPEC, EVAL_SUBSET, 2)
    coco = [i for i in images if i.source == "coco"]
    # 5 of the 6 images: image 2 is a calibration image and never a candidate.
    assert sorted(data.coco_id(i.member) for i in coco) == [1, 3, 4, 5, 6]
    # Frames 1, 3, 5 by the stride; 3 is a calibration frame.
    assert [i.id for i in images if i.source == "mot17"] == ["mot17-05-000001", "mot17-05-000005"]
    assert data.fetch(spec, images, transport, retry, tmp_path / "b", workers=2) == len(images)
    assert all(image_ok(tmp_path / "b", i) for i in images)

    names = {0: "person", 1: "bicycle", 2: "car"}
    out = tmp_path / "ds"
    summary = data.materialize(
        images,
        tmp_path / "b",
        out,
        spec,
        names,
        data.Blur(fraction=1.0, sigma_min=60, sigma_max=60),  # 1.5 px on a 32 px image
        2,
    )
    assert summary["images"] == {"coco": 5, "mot17": 2}
    assert summary["blurred"] == 7
    # Crowd (image 4) and the barely visible MOT person are not labeled.
    assert summary["boxes"] == {"coco": 4, "mot17": 1}
    frame = cv2.imread(str(out / "images/train/mot17-05-000001.jpg"))
    assert frame.shape == (96, 128, 3)  # resized as loadgen sends frames
    labels = (out / "labels/train/mot17-05-000001.txt").read_text().split()
    assert labels[0] == "0"
    assert float(labels[3]) == pytest.approx(10 / 64, abs=1e-6)  # widths scale with the frame
    assert (out / "labels/train/coco-000000000004.txt").read_text() == ""
    data_yaml = yaml.safe_load((out / "data.yaml").read_text())
    assert data_yaml["names"] == names
    assert len((out / "val.txt").read_text().splitlines()) == 2

    clean = data.materialize(
        images, tmp_path / "b", tmp_path / "ds0", spec, names, data.Blur(fraction=0.0), 2
    )
    assert clean["blurred"] == 0
    assert clean["dataset_sha256"] != summary["dataset_sha256"]
    again = data.materialize(
        images, tmp_path / "b", tmp_path / "ds1", spec, names, data.Blur(fraction=0.0), 2
    )
    assert again["dataset_sha256"] == clean["dataset_sha256"]  # byte-reproducible


def test_build_refuses_eval_sequences(tmp_path: Path, retry: RetryPolicy) -> None:
    spec, transport = fake_sources(tmp_path)
    bad = spec.model_copy(
        update={"mot17": spec.mot17.model_copy(update={"sequences": ["MOT17-02-FRCNN"]})}
    )
    seq = "MOT17/train/MOT17-02-FRCNN/"
    transport.files[MOT_URL] = make_zip(
        {
            seq + "seqinfo.ini": b"[Sequence]\nimDir=img1\nseqLength=1\nimWidth=8\nimHeight=8\n",
            seq + "gt/gt.txt": b"",
            f"{seq}img1/000001.jpg": jpeg(8, 8, 5),
        }
    )
    bad = bad.model_copy(
        update={
            "mot17": bad.mot17.model_copy(update={"archive_bytes": len(transport.files[MOT_URL])})
        }
    )
    with pytest.raises(data.DisjointError, match="02"):
        data.build(bad, transport, retry, tmp_path / "a", EVAL_SPEC, EVAL_SUBSET)


def test_committed_v1_list_is_disjoint_and_matches_its_spec() -> None:
    spec = data.load_spec(REPO_ROOT / "configs/finetune.v1.yaml")
    lock = REPO_ROOT / "configs/finetune.v1.lock.jsonl"
    images = read_subset(lock)
    assert sum(i.source == "coco" for i in images) == spec.coco.images
    assert {i.id.split("-")[1] for i in images if i.source == "mot17"} == {"05", "10", "11", "13"}
    assert [i.id for i in images] == sorted(i.id for i in images)
    assert all(len(i.sha256) == 64 for i in images)
    calibration = [img for p in spec.exclude_locks for img in read_lock(REPO_ROOT / p)]
    data.check_disjoint(
        images,
        spec.mot17.sequences,
        read_subset(EVAL_SUBSET),
        load_eval_spec(EVAL_SPEC).mot17.sequences,
        [i.member for i in calibration],
    )
