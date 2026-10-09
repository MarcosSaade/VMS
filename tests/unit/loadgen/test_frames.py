from pathlib import Path

import pytest

from tests.unit.loadgen.conftest import write_clip
from vms.datasets.clips import decode_jpeg
from vms.loadgen.frames import ClipNotFoundError, load_clip, prepare_frame
from vms.loadgen.settings import FrameSettings


def test_resizes_and_reencodes(tmp_path: Path) -> None:
    write_clip(tmp_path, "c", n=3, width=64, height=48)
    clip = load_clip(tmp_path, "c", FrameSettings(width=32, height=24, jpeg_quality=50))
    assert (clip.width, clip.height, clip.fps, len(clip.frames)) == (32, 24, 30.0, 3)
    for data in clip.frames:
        assert decode_jpeg(data).shape == (24, 32, 3)


def test_passes_stored_bytes_through_when_nothing_changes(tmp_path: Path) -> None:
    blobs = write_clip(tmp_path, "c", n=3, width=64, height=48)
    for frame_settings in (
        FrameSettings(width=None, height=None, jpeg_quality=None),
        FrameSettings(width=64, height=48, jpeg_quality=None),
    ):
        assert load_clip(tmp_path, "c", frame_settings).frames == tuple(blobs)


def test_frames_are_in_order_and_capped(tmp_path: Path) -> None:
    blobs = write_clip(tmp_path, "c", n=5, width=64, height=48)
    passthrough = FrameSettings(width=None, height=None, jpeg_quality=None)
    assert load_clip(tmp_path, "c", passthrough, max_frames=2).frames == tuple(blobs[:2])


def test_missing_clip_says_how_to_fetch_it(tmp_path: Path) -> None:
    with pytest.raises(ClipNotFoundError, match="make data-mini"):
        load_clip(tmp_path, "nope", FrameSettings())


def test_resize_without_quality_still_encodes(tmp_path: Path) -> None:
    blobs = write_clip(tmp_path, "c", n=1, width=64, height=48)
    out = prepare_frame(blobs[0], (16, 12), None)
    assert decode_jpeg(out).shape == (12, 16, 3)


def test_frame_size_needs_both_dimensions() -> None:
    with pytest.raises(ValueError, match="both"):
        FrameSettings(width=640, height=None)
