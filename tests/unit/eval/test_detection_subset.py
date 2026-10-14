"""Detection eval subset: ground-truth conversion, scattered zip reads, build and fetch with a
fake transport, and the committed v1 subset being frozen."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from tests.unit.datasets.conftest import FakeTransport, jpeg
from vms.common.settings import load_settings
from vms.datasets.http import ChecksumError, DownloadError, RetryPolicy
from vms.datasets.zipmembers import fetch_members, member_from_span, remote_zip_index
from vms.eval.detection.cli import check_problems
from vms.eval.detection.settings import DetectionEvalSettings
from vms.eval.detection.subset import (
    Mot17Spec,
    MotBox,
    SubsetSpec,
    UnpinnedError,
    build,
    coco_ground_truth,
    fetch,
    image_ok,
    mot_frame_ground_truth,
    parse_mot_gt,
    read_subset,
    select_coco_ids,
    write_subset,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
COCO_IMAGES = "https://example.test/val2017.zip"
COCO_ANNS = "https://example.test/annotations.zip"
MOT_URL = "https://example.test/MOT17.zip"


def make_zip(members: dict[str, bytes], compression: int = zipfile.ZIP_DEFLATED) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=compression) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buf.getvalue()


@pytest.fixture
def retry() -> RetryPolicy:
    return RetryPolicy(attempts=2, base_s=0.01, sleep=lambda _: None)


MOT_SPEC = Mot17Spec(
    url=MOT_URL,  # type: ignore[arg-type]
    archive_bytes=1,
    sequences=["MOT17-02-FRCNN"],
    frame_stride=2,
    positive_classes=[1, 7],
    ignore_classes=[8],
    crowd_classes=[13],
    min_visibility=0.25,
)


# --- Ground truth ----------------------------------------------------------------------------


def test_coco_sample_is_seeded_sorted_and_bounded() -> None:
    ids = list(range(100, 200))
    a = select_coco_ids(ids, 10, seed=3)
    assert a == select_coco_ids(list(reversed(ids)), 10, seed=3)
    assert a == sorted(a)
    assert len(set(a)) == 10
    assert set(a) <= set(ids)
    assert a != select_coco_ids(ids, 10, seed=4)
    with pytest.raises(ValueError, match="only 100"):
        select_coco_ids(ids, 101, seed=0)


def test_coco_boxes_become_corners_and_crowds_ignore_regions() -> None:
    annotations = {
        "images": [
            {"id": 1, "file_name": "000000000001.jpg", "width": 64, "height": 48},
            {"id": 2, "file_name": "000000000002.jpg", "width": 32, "height": 32},
        ],
        "categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}],
        "annotations": [
            {"id": 11, "image_id": 1, "category_id": 3, "bbox": [1, 2, 10, 20], "iscrowd": 0},
            {"id": 10, "image_id": 1, "category_id": 1, "bbox": [0, 0, 30, 30], "iscrowd": 1},
        ],
    }
    first, empty = coco_ground_truth(annotations, [1, 2])
    assert first.id == "coco-000000000001"
    assert first.member == "val2017/000000000001.jpg"
    assert first.boxes == [(0.0, 0.0, 30.0, 30.0), (1.0, 2.0, 11.0, 22.0)]  # by annotation id
    assert first.classes == ["person", "car"]
    assert first.ignore == [True, False]
    assert first.crowd == [True, False]
    assert empty.boxes == []
    assert (empty.width, empty.height) == (32, 32)


def test_mot_rows_are_converted_to_zero_based_corners() -> None:
    (row,) = parse_mot_gt("7,3,11,21,5,10,1,1,0.5\n\n")
    assert row == MotBox(frame=7, box=(10.0, 20.0, 15.0, 30.0), cls=1, flag=1, visibility=0.5)


def test_mot_frame_keeps_visible_persons_and_ignores_the_rest() -> None:
    rows = [
        MotBox(1, (0, 0, 10, 10), cls=1, flag=1, visibility=0.9),  # pedestrian
        MotBox(1, (10, 10, 20, 20), cls=7, flag=0, visibility=0.6),  # static person
        MotBox(1, (20, 20, 30, 30), cls=1, flag=1, visibility=0.1),  # hidden: ignore
        MotBox(1, (30, 30, 40, 40), cls=8, flag=0, visibility=1.0),  # distractor: ignore
        MotBox(1, (40, 0, 60, 20), cls=13, flag=0, visibility=1.0),  # crowd
        MotBox(1, (0, 0, 5, 5), cls=3, flag=0, visibility=1.0),  # car: dropped
        MotBox(1, (-5, -5, 5, 5), cls=1, flag=1, visibility=1.0),  # clipped to the image
        MotBox(1, (70, 0, 90, 10), cls=1, flag=1, visibility=1.0),  # outside: dropped
        MotBox(2, (0, 0, 10, 10), cls=1, flag=1, visibility=1.0),  # other frame
    ]
    boxes, ignore, crowd = mot_frame_ground_truth(rows, 1, 64, 48, MOT_SPEC)
    assert boxes == [
        (0, 0, 10, 10),
        (10, 10, 20, 20),
        (20, 20, 30, 30),
        (30, 30, 40, 40),
        (40, 0, 60, 20),
        (0.0, 0.0, 5, 5),
    ]
    assert ignore == [False, False, True, True, True, False]
    assert crowd == [False, False, False, False, True, False]


# --- Scattered zip reads ---------------------------------------------------------------------


@pytest.mark.parametrize("compression", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_members_are_read_one_range_request_each(retry: RetryPolicy, compression: int) -> None:
    members = {f"dir/{i:03d}.bin": bytes([i]) * (100 + i) for i in range(20)}
    blob = make_zip(members, compression)
    transport = FakeTransport({"u": blob})
    index = remote_zip_index(transport, retry, "u", len(blob))
    transport.requests.clear()
    wanted = [index["dir/003.bin"], index["dir/019.bin"]]  # the last member ends near EOF
    got = fetch_members(transport, retry, "u", len(blob), wanted, workers=2)
    assert got == {"dir/003.bin": members["dir/003.bin"], "dir/019.bin": members["dir/019.bin"]}
    assert len(transport.requests) == 2


def test_member_corruption_and_wrong_archive_size_are_caught(retry: RetryPolicy) -> None:
    blob = make_zip({"a.bin": b"x" * 500}, zipfile.ZIP_STORED)
    transport = FakeTransport({"u": blob})
    with pytest.raises(ChecksumError, match="spec pins"):
        remote_zip_index(transport, retry, "u", len(blob) + 1)
    info = remote_zip_index(transport, retry, "u", len(blob))["a.bin"]
    span = bytearray(blob[info.header_offset :])
    span[-200] ^= 0xFF  # inside the data (the central directory follows it)
    with pytest.raises(ChecksumError, match="CRC32"):
        member_from_span(info, bytes(span))
    with pytest.raises(DownloadError, match="signature"):
        member_from_span(info, b"\0" * 64)


# --- Build and fetch -------------------------------------------------------------------------


def fake_sources() -> tuple[SubsetSpec, FakeTransport, str]:
    annotations = {
        "images": [
            {"id": i, "file_name": f"{i:012d}.jpg", "width": 32, "height": 24} for i in range(1, 6)
        ],
        "categories": [{"id": 1, "name": "person"}],
        "annotations": [
            {"id": i, "image_id": i, "category_id": 1, "bbox": [2, 2, 8, 8], "iscrowd": 0}
            for i in range(1, 6)
        ],
    }
    ann_bytes = json.dumps(annotations).encode()
    anns_zip = make_zip({"annotations/instances_val2017.json": ann_bytes})
    images_zip = make_zip(
        {f"val2017/{i:012d}.jpg": jpeg(32, 24, 40 * i) for i in range(1, 6)}, zipfile.ZIP_STORED
    )
    seq = "MOT17/train/MOT17-02-FRCNN/"
    mot_zip = make_zip(
        {
            seq + "seqinfo.ini": b"[Sequence]\nimDir=img1\nseqLength=5\nimWidth=64\nimHeight=48\n"
            b"imExt=.jpg\n",
            seq + "gt/gt.txt": b"1,1,1,1,10,10,1,1,1.0\n3,1,1,1,10,10,1,1,1.0\n",
            **{f"{seq}img1/{f:06d}.jpg": jpeg(64, 48, 30 * f) for f in range(1, 6)},
        }
    )
    spec = SubsetSpec.model_validate(
        {
            "version": 1,
            "seed": 9,
            "licenses": {},
            "coco": {
                "images_url": COCO_IMAGES,
                "images_bytes": len(images_zip),
                "annotations_url": COCO_ANNS,
                "annotations_bytes": len(anns_zip),
                "annotations_member": "annotations/instances_val2017.json",
                "annotations_sha256": hashlib.sha256(ann_bytes).hexdigest(),
                "images": 3,
            },
            "mot17": MOT_SPEC.model_dump(mode="json")
            | {"url": MOT_URL, "archive_bytes": len(mot_zip)},
        }
    )
    transport = FakeTransport({COCO_IMAGES: images_zip, COCO_ANNS: anns_zip, MOT_URL: mot_zip})
    return spec, transport, hashlib.sha256(ann_bytes).hexdigest()


def test_build_then_fetch_on_a_fresh_clone(tmp_path: Path, retry: RetryPolicy) -> None:
    spec, transport, ann_hash = fake_sources()
    images, got_hash = build(spec, "spec", transport, retry, tmp_path / "a", workers=2)
    assert got_hash == ann_hash
    assert [i.source for i in images] == ["coco"] * 3 + ["mot17"] * 3  # frames 1, 3, 5
    mot = {i.id: i for i in images if i.source == "mot17"}
    assert mot["mot17-02-000001"].boxes == [(0.0, 0.0, 10.0, 10.0)]
    assert mot["mot17-02-000005"].boxes == []
    assert all(image_ok(tmp_path / "a", i) for i in images)

    subset_file = tmp_path / "subset.jsonl"
    write_subset(subset_file, images)
    listed = read_subset(subset_file)
    fresh = tmp_path / "b"
    assert fetch(spec, listed, transport, retry, fresh, workers=2) == 6
    assert all(image_ok(fresh, i) for i in listed)
    assert fetch(spec, listed, transport, retry, fresh) == 0  # idempotent

    listed[0].path(fresh).write_bytes(b"corrupt")
    assert fetch(spec, listed, transport, retry, fresh) == 1
    assert image_ok(fresh, listed[0])


def test_fetch_rejects_bytes_that_differ_from_the_list(tmp_path: Path, retry: RetryPolicy) -> None:
    spec, transport, _ = fake_sources()
    images, _ = build(spec, "spec", transport, retry, tmp_path / "a")
    tampered = [images[0].model_copy(update={"sha256": "0" * 64})]
    with pytest.raises(ChecksumError, match="sha256"):
        fetch(spec, tampered, transport, retry, tmp_path / "b")


def test_build_stops_before_downloading_images_when_annotations_are_unpinned(
    tmp_path: Path, retry: RetryPolicy
) -> None:
    spec, transport, ann_hash = fake_sources()
    unpinned = spec.model_copy(
        update={"coco": spec.coco.model_copy(update={"annotations_sha256": None})}
    )
    with pytest.raises(UnpinnedError) as exc:
        build(unpinned, "spec", transport, retry, tmp_path)
    assert exc.value.sha256 == ann_hash
    assert not any(url == COCO_IMAGES for url, _, _ in transport.requests)
    wrong = spec.model_copy(
        update={"coco": spec.coco.model_copy(update={"annotations_sha256": "1" * 64})}
    )
    with pytest.raises(ChecksumError):
        build(wrong, "spec", transport, retry, tmp_path)


# --- The committed subset --------------------------------------------------------------------


def test_committed_v1_subset_is_frozen() -> None:
    """Every reported mAP cites this subset: the spec and the file list must not change after
    the build recorded their hashes. A new selection is a new version."""
    settings = load_settings(
        DetectionEvalSettings, "eval-detection", config_dir=REPO_ROOT / "configs"
    )
    paths = settings.paths.model_copy(
        update={
            "spec": REPO_ROOT / settings.paths.spec,
            "subset": REPO_ROOT / settings.paths.subset,
            "manifest": REPO_ROOT / settings.paths.manifest,
        }
    )
    assert check_problems(settings.model_copy(update={"paths": paths})) == []
    images = read_subset(paths.subset)
    assert sum(i.source == "coco" for i in images) == 500
    assert {i.id.rsplit("-", 1)[0] for i in images if i.source == "mot17"} == {
        "mot17-02",
        "mot17-04",
        "mot17-09",
    }
