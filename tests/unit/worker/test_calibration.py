"""INT8 calibration set (task 5.2): seeded selection, build and fetch with a fake transport,
frames prepared as the worker sees them, and the committed v1 list being frozen and disjoint
from the detection eval subset."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from tests.unit.datasets.conftest import FakeTransport, jpeg
from vms.datasets.http import ChecksumError, RetryPolicy
from vms.eval.detection.subset import load_spec as load_eval_spec
from vms.eval.detection.subset import read_subset
from vms.worker import calibration as cal

REPO_ROOT = Path(__file__).resolve().parents[3]
COCO = "https://example.test/train2017.zip"
MOT = "https://example.test/MOT17.zip"


def make_zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buf.getvalue()


@pytest.fixture
def retry() -> RetryPolicy:
    return RetryPolicy(attempts=2, base_s=0.01, sleep=lambda _: None)


def fake() -> tuple[cal.CalibrationSpec, FakeTransport]:
    coco = make_zip(
        {f"train2017/{i:012d}.jpg": jpeg(32, 24, 20 * i) for i in range(1, 9)}
        | {"train2017/": b"", "annotations/readme.txt": b"x"}
    )
    seq = "MOT17/train/MOT17-05-FRCNN/"
    mot = make_zip(
        {
            seq + "seqinfo.ini": b"[Sequence]\nimDir=img1\nseqLength=10\nimWidth=64\n"
            b"imHeight=48\nimExt=.jpg\n",
            **{f"{seq}img1/{f:06d}.jpg": jpeg(64, 48, 10 * f) for f in range(1, 11)},
        }
    )
    spec = cal.CalibrationSpec.model_validate(
        {
            "version": 1,
            "seed": 4,
            "coco": {
                "images_url": COCO,
                "images_bytes": len(coco),
                "member_prefix": "train2017/",
                "images": 3,
            },
            "mot17": {
                "url": MOT,
                "archive_bytes": len(mot),
                "sequences": ["MOT17-05-FRCNN"],
                "frames_per_sequence": 4,
            },
            "mot_frame": {"width": 128, "height": 72, "jpeg_quality": 85},
        }
    )
    return spec, FakeTransport({COCO: coco, MOT: mot})


def test_selection_is_seeded_sorted_and_bounded() -> None:
    names = [f"n{i}" for i in range(20)]
    a = cal.select_members(names, 5, seed=1)
    assert a == cal.select_members(list(reversed(names)), 5, seed=1) == sorted(a)
    assert a != cal.select_members(names, 5, seed=2)
    with pytest.raises(ValueError, match="only 20"):
        cal.select_members(names, 21, seed=1)


def test_frames_are_spread_from_first_to_last() -> None:
    assert cal.spaced_frames(10, 4) == [1, 4, 7, 10]
    assert cal.spaced_frames(5, 5) == [1, 2, 3, 4, 5]
    assert cal.spaced_frames(837, 1) == [1]
    with pytest.raises(ValueError, match="4 frames of a 3-frame"):
        cal.spaced_frames(3, 4)


def test_build_then_fetch_on_a_fresh_clone(tmp_path: Path, retry: RetryPolicy) -> None:
    spec, transport = fake()
    images = cal.build(spec, transport, retry, tmp_path / "a", workers=2)
    assert [i.source for i in images] == ["coco"] * 3 + ["mot17"] * 4
    assert all(i.member.startswith("train2017/") and i.member.endswith(".jpg") for i in images[:3])
    assert [i.id for i in images[3:]] == [f"mot17-05-{f:06d}" for f in (1, 4, 7, 10)]

    lock = tmp_path / "lock.jsonl"
    digest = cal.write_lock(lock, images)
    assert cal.lock_sha256(lock) == digest
    listed = cal.read_lock(lock)
    assert listed == images
    fresh = tmp_path / "b"
    assert cal.fetch(spec, listed, transport, retry, fresh, workers=2) == 7
    assert cal.fetch(spec, listed, transport, retry, fresh) == 0

    pixels = list(cal.load_pixels(listed, fresh, spec))
    assert [p.shape for p in pixels] == [(24, 32, 3)] * 3 + [(72, 128, 3)] * 4  # MOT resized

    listed[0].path(fresh).write_bytes(b"corrupt")
    with pytest.raises(ChecksumError, match="calibration list"):
        list(cal.load_pixels(listed, fresh, spec))


def test_committed_v1_list_is_frozen_and_disjoint_from_the_eval_subset() -> None:
    spec = cal.load_spec(REPO_ROOT / "configs/calibration.v1.yaml")
    images = cal.read_lock(REPO_ROOT / "configs/calibration.v1.lock.jsonl")
    assert len(images) == spec.coco.images + len(spec.mot17.sequences) * (
        spec.mot17.frames_per_sequence
    )
    assert cal.lock_sha256(REPO_ROOT / "configs/calibration.v1.lock.jsonl") == (
        "35018eeb8b98739e67673d85d3ae9098e29d89b2c75d955c97364bcecd2cd43c"
    )
    eval_spec = load_eval_spec(REPO_ROOT / "configs/eval/detection.v1.yaml")
    eval_members = {i.member for i in read_subset(REPO_ROOT / "eval/detection/subset.v1.jsonl")}
    assert not eval_members.intersection(i.member for i in images)
    eval_sequences = {s.split("-")[1] for s in eval_spec.mot17.sequences}
    assert not eval_sequences.intersection(s.split("-")[1] for s in spec.mot17.sequences)
    assert all(i.member.startswith("train2017/") for i in images if i.source == "coco")
