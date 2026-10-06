from pathlib import Path
from typing import Any

import pytest
from pydantic import HttpUrl, ValidationError

from vms.datasets.spec import (
    DataSpec,
    Processing,
    VideoFileSource,
    ZipSequenceSource,
    clip_spec_hash,
    load_spec,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def spec() -> DataSpec:
    return load_spec(REPO_ROOT / "configs/data/sources.yaml")


def test_shipped_spec_is_consistent(spec: DataSpec) -> None:
    assert set(spec.subsets) >= {"full", "mini"}
    for clip in spec.subset("full") + spec.subset("mini"):
        assert clip.scene.objects
        assert str(clip.page_url).startswith("https://")


def test_every_zip_clip_is_pinned(spec: DataSpec) -> None:
    for clip in spec.clips:
        if isinstance(clip.source, ZipSequenceSource):
            assert clip.source.checksum != "sha256:" + "0" * 64, f"{clip.id} is not pinned"


def test_mini_subset_is_small_and_needs_no_license_acceptance(spec: DataSpec) -> None:
    mini = spec.subset("mini")
    assert 1 <= len(mini) <= 4
    assert not any(spec.licenses[c.license].requires_acceptance for c in mini)
    for clip in mini:
        source = clip.source
        if isinstance(source, ZipSequenceSource):
            assert source.last_frame is not None
            assert source.last_frame <= 600
        else:
            assert source.file_bytes < 20_000_000


def test_default_subset_has_scene_variety(spec: DataSpec) -> None:
    open_clips = [
        c for c in spec.subset("full") if not spec.licenses[c.license].requires_acceptance
    ]
    assert {c.scene.location for c in open_clips} == {"indoor", "outdoor"}
    assert {"day", "night"} <= {c.scene.lighting for c in open_clips}
    objects = {o for c in open_clips for o in c.scene.objects}
    assert {"person", "car"} <= objects


def test_licenses_requiring_acceptance_are_flagged(spec: DataSpec) -> None:
    assert spec.licenses["virat"].requires_acceptance
    assert not spec.licenses["cc-by-nc-sa-3.0"].requires_acceptance


def test_spec_hash_tracks_source_and_processing_but_not_labels(spec: DataSpec) -> None:
    clip = spec.clips[0]
    base = clip_spec_hash(clip, spec.processing)
    relabeled = clip.model_copy(update={"title": "new title"})
    assert clip_spec_hash(relabeled, spec.processing) == base
    assert clip_spec_hash(clip, Processing(max_height=480)) != base
    new_source = clip.source.model_copy(update={"url": HttpUrl("https://x.test/a")})
    moved = clip.model_copy(update={"source": new_source})
    assert clip_spec_hash(moved, spec.processing) != base


def base_spec(**changes: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "version": 1,
        "licenses": {
            "l": {
                "name": "L",
                "url": "https://l.test",
                "commercial_use": True,
                "share_alike": False,
            }
        },
        "clips": [
            {
                "id": "a",
                "title": "A",
                "license": "l",
                "attribution": "x",
                "page_url": "https://p.test",
                "scene": {
                    "description": "d",
                    "location": "indoor",
                    "lighting": "day",
                    "viewpoint": "elevated",
                    "density": "sparse",
                    "objects": ["person"],
                },
                "source": {
                    "kind": "video_file",
                    "url": "https://v.test/a.mp4",
                    "file_bytes": 1,
                    "checksum": "sha1:" + "a" * 40,
                },
            }
        ],
        "subsets": {"all": ["a"]},
    }
    data.update(changes)
    return data


@pytest.mark.parametrize(
    ("changes", "fragment"),
    [
        ({"subsets": {"all": ["missing"]}}, "unknown clips"),
        ({"licenses": {}}, "unknown license"),
    ],
)
def test_spec_reference_errors(changes: dict[str, Any], fragment: str) -> None:
    with pytest.raises(ValidationError, match=fragment):
        DataSpec.model_validate(base_spec(**changes))


def test_spec_rejects_duplicate_clip_ids() -> None:
    data = base_spec()
    data["clips"] = data["clips"] * 2
    with pytest.raises(ValidationError, match="duplicate"):
        DataSpec.model_validate(data)


@pytest.mark.parametrize(
    "checksum",
    ["sha1:xyz", "sha1:" + "a" * 39, "crc32:deadbeef", "a" * 40, "sha256:" + "A" * 64],
)
def test_video_checksum_format_is_validated(checksum: str) -> None:
    with pytest.raises(ValidationError):
        VideoFileSource(url="https://v.test/a.mp4", file_bytes=1, checksum=checksum)  # type: ignore[arg-type]


def test_zip_source_validation() -> None:
    ok = {
        "url": "https://z.test/a.zip",
        "archive_bytes": 1,
        "frames_prefix": "a/img1/",
        "fps": 30,
        "checksum": "sha256:" + "0" * 64,
    }
    ZipSequenceSource.model_validate(ok)
    with pytest.raises(ValidationError, match="end with"):
        ZipSequenceSource.model_validate({**ok, "frames_prefix": "a/img1"})
    with pytest.raises(ValidationError, match="last_frame"):
        ZipSequenceSource.model_validate({**ok, "first_frame": 10, "last_frame": 5})


def test_subset_lookup(spec: DataSpec) -> None:
    with pytest.raises(KeyError, match="unknown subset"):
        spec.subset("nope")
