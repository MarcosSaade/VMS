import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from tests.unit.datasets.conftest import (
    FakeTransport,
    make_zip,
    sequence_members,
    sha,
    write_video,
)
from vms.datasets.cli import compare_with_manifest, fetch, write_manifest
from vms.datasets.clips import (
    ChecksumError,
    fetch_clip,
    frame_step,
    members_digest,
    normalize,
    read_manifest,
    read_zip_sequence,
)
from vms.datasets.http import RetryPolicy
from vms.datasets.spec import DataSpec, Processing, ZipSequenceSource

ZIP_URL = "https://example.test/MOT.zip"
VIDEO_URL = "https://example.test/clip.avi"
PREFIX = "MOT/train/SEQ-01/img1/"
PROCESSING = Processing(max_height=24, max_fps=10, jpeg_quality=90)


@pytest.fixture
def archive() -> tuple[bytes, dict[str, bytes]]:
    seq = sequence_members(PREFIX, 12)
    other = sequence_members("MOT/train/SEQ-02/img1/", 5)
    return make_zip({**seq, **other, "MOT/train/SEQ-01/seqinfo.ini": b"[Sequence]"}), seq


def zip_source(data: bytes, members: dict[str, bytes], **overrides: Any) -> ZipSequenceSource:
    fields: dict[str, Any] = {
        "url": ZIP_URL,
        "archive_bytes": len(data),
        "frames_prefix": PREFIX,
        "fps": 30,
        "checksum": members_digest(sorted(members.items())),
    }
    fields.update(overrides)
    return ZipSequenceSource.model_validate(fields)


def spec_with(clips: list[dict[str, Any]], **subsets: list[str]) -> DataSpec:
    return DataSpec.model_validate(
        {
            "version": 1,
            "processing": PROCESSING.model_dump(),
            "licenses": {
                "open": {
                    "name": "Open",
                    "url": "https://l.test/open",
                    "commercial_use": True,
                    "share_alike": False,
                },
                "gated": {
                    "name": "Gated",
                    "url": "https://l.test/gated",
                    "commercial_use": True,
                    "share_alike": False,
                    "requires_acceptance": True,
                },
            },
            "clips": clips,
            "subsets": subsets or {"all": [c["id"] for c in clips]},
        }
    )


def clip_dict(clip_id: str, source: dict[str, Any], license_: str = "open") -> dict[str, Any]:
    return {
        "id": clip_id,
        "title": clip_id,
        "license": license_,
        "attribution": "someone",
        "page_url": "https://example.test/page",
        "scene": {
            "description": "d",
            "location": "outdoor",
            "lighting": "day",
            "viewpoint": "elevated",
            "density": "sparse",
            "objects": ["person"],
        },
        "source": source,
    }


# --- Frame helpers ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("fps", "max_fps", "step"), [(30, 30, 1), (30, 10, 3), (25, 10, 3), (60, 30, 2), (14, 30, 1)]
)
def test_frame_step(fps: float, max_fps: float, step: int) -> None:
    assert frame_step(fps, max_fps) == step


def test_normalize_downscales_keeping_aspect_and_never_upscales() -> None:
    big = np.zeros((1080, 1920, 3), dtype=np.uint8)
    assert normalize(big, Processing(max_height=720)).shape == (720, 1280, 3)
    odd = np.zeros((880, 1173, 3), dtype=np.uint8)
    assert normalize(odd, Processing(max_height=720)).shape == (720, 960, 3)
    small = np.zeros((480, 640, 3), dtype=np.uint8)
    assert normalize(small, Processing(max_height=720)) is small


def test_members_digest_is_order_independent_and_content_sensitive() -> None:
    a = [("a.jpg", b"1"), ("b.jpg", b"2")]
    assert members_digest(a) == members_digest(list(reversed(a)))
    assert members_digest(a) != members_digest([("a.jpg", b"1"), ("b.jpg", b"3")])
    assert members_digest(a) != members_digest([("a.jpg", b"12"), ("b.jpg", b"")])


# --- Zip sequences ---------------------------------------------------------------------------


def test_reads_only_the_selected_frames(
    tmp_path: Path, retry: RetryPolicy, archive: tuple[bytes, dict[str, bytes]]
) -> None:
    data, seq = archive
    wanted = {k: v for k, v in seq.items() if 3 <= int(k[-10:-4]) <= 7}
    source = zip_source(data, wanted, first_frame=3, last_frame=7)
    members = read_zip_sequence(source, FakeTransport({ZIP_URL: data}), retry, tmp_path)
    assert [name[-10:] for name, _ in members] == [f"{i:06d}.jpg" for i in range(3, 8)]


def test_zip_sequence_rejects_wrong_digest_and_wrong_archive(
    tmp_path: Path, retry: RetryPolicy, archive: tuple[bytes, dict[str, bytes]]
) -> None:
    data, seq = archive
    transport = FakeTransport({ZIP_URL: data})
    with pytest.raises(ChecksumError, match="expected"):
        read_zip_sequence(
            zip_source(data, seq, checksum="sha256:" + "0" * 64), transport, retry, tmp_path
        )
    with pytest.raises(ChecksumError, match="bytes"):
        read_zip_sequence(zip_source(data, seq, archive_bytes=1), transport, retry, tmp_path)
    with pytest.raises(ChecksumError, match="no members"):
        read_zip_sequence(
            zip_source(data, seq, frames_prefix="MOT/none/"), transport, retry, tmp_path
        )


def test_zip_sequence_is_fetched_as_one_span(
    tmp_path: Path, retry: RetryPolicy, archive: tuple[bytes, dict[str, bytes]]
) -> None:
    data, seq = archive
    transport = FakeTransport({ZIP_URL: data})
    read_zip_sequence(zip_source(data, seq), transport, retry, tmp_path)
    # Central directory reads plus one span, not one request per member.
    assert len(transport.requests) < len(seq)


def test_fetch_zip_clip_writes_normalized_frames_and_manifest(
    tmp_path: Path, retry: RetryPolicy, archive: tuple[bytes, dict[str, bytes]]
) -> None:
    data, seq = archive
    spec = spec_with([clip_dict("seq-01", zip_source(data, seq).model_dump(mode="json"))])
    manifest = fetch_clip(
        tmp_path, spec.clips[0], PROCESSING, FakeTransport({ZIP_URL: data}), retry
    )
    frames = sorted((tmp_path / "clips" / "seq-01" / "frames").iterdir())
    # 12 frames at 30 fps, max 10 fps => every 3rd frame: 1, 4, 7, 10.
    assert [f.name for f in frames] == ["000001.jpg", "000002.jpg", "000003.jpg", "000004.jpg"]
    assert manifest.frames == 4
    assert manifest.fps == 10
    assert manifest.duration_s == 0.4
    assert (manifest.width, manifest.height) == (32, 24)
    assert manifest.content_sha256.startswith("sha256:")
    assert read_manifest(tmp_path, "seq-01") == manifest


# --- Video files -----------------------------------------------------------------------------


def test_fetch_video_clip_trims_and_resamples(tmp_path: Path, retry: RetryPolicy) -> None:
    video = write_video(tmp_path / "src.avi", frames=40, fps=20)
    source = {
        "kind": "video_file",
        "url": VIDEO_URL,
        "file_bytes": len(video),
        "checksum": sha("sha1", video),
        "start_s": 0.5,
        "duration_s": 1.0,
    }
    spec = spec_with([clip_dict("vid", source)])
    data_dir = tmp_path / "data"
    manifest = fetch_clip(
        data_dir, spec.clips[0], PROCESSING, FakeTransport({VIDEO_URL: video}), retry
    )
    # 1 s from 0.5 s at 20 fps = frames 10..29; max 10 fps => every 2nd => 10 frames.
    assert manifest.frames == 10
    assert manifest.fps == 10
    assert (manifest.width, manifest.height) == (32, 24)
    assert (data_dir / "raw" / "vid" / "clip.avi").is_file()  # cached for re-runs


# --- Orchestration ---------------------------------------------------------------------------


def test_fetch_is_idempotent(
    tmp_path: Path, retry: RetryPolicy, archive: tuple[bytes, dict[str, bytes]]
) -> None:
    data, seq = archive
    spec = spec_with([clip_dict("seq-01", zip_source(data, seq).model_dump(mode="json"))])
    transport = FakeTransport({ZIP_URL: data})
    first, _ = fetch(spec, "all", tmp_path, transport, retry)
    calls = len(transport.requests)
    second, _ = fetch(spec, "all", tmp_path, transport, retry)
    assert first == second
    assert len(transport.requests) == calls  # nothing downloaded again


def test_processing_change_refetches(
    tmp_path: Path, retry: RetryPolicy, archive: tuple[bytes, dict[str, bytes]]
) -> None:
    data, seq = archive
    spec = spec_with([clip_dict("seq-01", zip_source(data, seq).model_dump(mode="json"))])
    transport = FakeTransport({ZIP_URL: data})
    fetch(spec, "all", tmp_path, transport, retry)
    changed = spec.model_copy(update={"processing": Processing(max_height=16, max_fps=10)})
    (manifest,), _ = fetch(changed, "all", tmp_path, transport, retry)
    assert manifest.height == 16


def test_gated_license_is_skipped_unless_accepted(
    tmp_path: Path, retry: RetryPolicy, archive: tuple[bytes, dict[str, bytes]]
) -> None:
    data, seq = archive
    source = zip_source(data, seq).model_dump(mode="json")
    spec = spec_with([clip_dict("gated-01", source, license_="gated")])
    transport = FakeTransport({ZIP_URL: data})
    present, skipped = fetch(spec, "all", tmp_path, transport, retry)
    assert (present, skipped) == ([], ["gated-01"])
    assert transport.requests == []
    present, skipped = fetch(spec, "all", tmp_path, transport, retry, frozenset({"gated"}))
    assert [m.id for m in present] == ["gated-01"]
    assert skipped == []


def test_failed_fetch_leaves_no_complete_clip(
    tmp_path: Path, retry: RetryPolicy, archive: tuple[bytes, dict[str, bytes]]
) -> None:
    data, seq = archive
    bad = zip_source(data, seq, checksum="sha256:" + "1" * 64).model_dump(mode="json")
    spec = spec_with([clip_dict("seq-01", bad)])
    with pytest.raises(ChecksumError):
        fetch(spec, "all", tmp_path, FakeTransport({ZIP_URL: data}), retry)
    assert read_manifest(tmp_path, "seq-01") is None


def test_manifest_merges_and_detects_content_drift(
    tmp_path: Path, retry: RetryPolicy, archive: tuple[bytes, dict[str, bytes]]
) -> None:
    data, seq = archive
    spec = spec_with([clip_dict("seq-01", zip_source(data, seq).model_dump(mode="json"))])
    (clip,), _ = fetch(spec, "all", tmp_path / "d", FakeTransport({ZIP_URL: data}), retry)
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump({"spec_version": 1, "clips": [{"id": "older"}]}))
    write_manifest(path, 1, [clip])
    written = yaml.safe_load(path.read_text())
    assert [c["id"] for c in written["clips"]] == ["older", "seq-01"]
    assert compare_with_manifest(path, [clip]) == []

    tampered = clip.model_copy(update={"content_sha256": "sha256:" + "f" * 64})
    assert compare_with_manifest(path, [tampered]) == ["seq-01"]
    json.dumps(written)  # plain data only


def test_spec_fps_overrides_the_probed_rate(tmp_path: Path, retry: RetryPolicy) -> None:
    # Some containers report a wrong rate (a VP8 WebM read as 30000 fps instead of 29.97);
    # a rate pinned in the spec wins.
    video = write_video(tmp_path / "src.avi", frames=20, fps=20)
    source = {
        "kind": "video_file",
        "url": VIDEO_URL,
        "file_bytes": len(video),
        "checksum": sha("sha1", video),
        "fps": 10,
    }
    spec = spec_with([clip_dict("vid", source)])
    manifest = fetch_clip(
        tmp_path / "data", spec.clips[0], PROCESSING, FakeTransport({VIDEO_URL: video}), retry
    )
    assert manifest.fps == 10
    assert manifest.frames == 20  # 10 fps <= max_fps: every frame kept


def test_implausible_probed_rate_fails_loudly(
    tmp_path: Path, retry: RetryPolicy, monkeypatch: pytest.MonkeyPatch
) -> None:
    import vms.datasets.clips as clips

    video = write_video(tmp_path / "src.avi", frames=20, fps=20)
    monkeypatch.setattr(clips, "video_fps", lambda path: 30000.0)
    source = {
        "kind": "video_file",
        "url": VIDEO_URL,
        "file_bytes": len(video),
        "checksum": sha("sha1", video),
    }
    spec = spec_with([clip_dict("vid", source)])
    with pytest.raises(ValueError, match="implausible frame rate"):
        fetch_clip(
            tmp_path / "data", spec.clips[0], PROCESSING, FakeTransport({VIDEO_URL: video}), retry
        )


def test_manifest_drops_clips_no_longer_in_the_spec(
    tmp_path: Path, retry: RetryPolicy, archive: tuple[bytes, dict[str, bytes]]
) -> None:
    data, seq = archive
    spec = spec_with([clip_dict("seq-01", zip_source(data, seq).model_dump(mode="json"))])
    (clip,), _ = fetch(spec, "all", tmp_path / "d", FakeTransport({ZIP_URL: data}), retry)
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump({"spec_version": 1, "clips": [{"id": "removed-clip"}]}))
    write_manifest(path, 1, [clip], known={"seq-01"})
    assert [c["id"] for c in yaml.safe_load(path.read_text())["clips"]] == ["seq-01"]
