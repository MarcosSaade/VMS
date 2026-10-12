"""Review-video renderer: the worker's tracking path drawn on frames, written as a video."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from tests.unit.worker.conftest import FakeDetector
from vms.storage.catalog import load_catalog
from vms.worker import render as render_module
from vms.worker.detector import Detection
from vms.worker.render import Segment, render, zones_for
from vms.worker.tracking import CameraTracker, TrackerSettings
from vms.worker.zones import ZoneDef, ZoneSettings

REPO_ROOT = Path(__file__).resolve().parents[3]


def frame(i: int) -> bytes:
    image = np.full((90, 160, 3), 60, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return bytes(buf)


def test_render_draws_a_walk_into_a_zone(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render_module.shutil, "which", lambda _: None)  # no ffmpeg dependency
    clip_fps, fps, n = 30.0, 5.0, 60  # 2 s of clip -> 10 sampled frames
    zone = ZoneDef("cam-x:right", "right", ((0.5, 0.0), (1.0, 0.0), (1.0, 1.0), (0.5, 1.0)))
    # The detector sees a person walking right by 6 px per sampled frame.
    detector = FakeDetector(
        script=[[Detection("person", 0.9, 40 + 6 * k, 20, 60 + 6 * k, 70)] for k in range(10)]
    )
    out = tmp_path / "review.mp4"
    summaries = render(
        [Segment("clip", "cam-x", "x", clip_fps, [frame(i) for i in range(n)], [zone])],
        detector,
        CameraTracker,
        TrackerSettings(fps=fps),
        ZoneSettings(min_frames=2, loiter_s=None),
        out,
        fps=fps,
    )
    (s,) = summaries
    assert s.frames == 10
    assert len(s.tracks) == 1
    assert dict(s.events) == {"zone_enter": 1}
    video = cv2.VideoCapture(str(out))
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == 10
    ok, first = video.read()
    assert ok
    assert first.shape == (90, 160, 3)


def test_render_respects_max_seconds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render_module.shutil, "which", lambda _: None)
    (s,) = render(
        [Segment("clip", "cam-x", "x", 30.0, [frame(i) for i in range(300)], [])],
        FakeDetector(detections=[]),
        CameraTracker,
        TrackerSettings(),
        ZoneSettings(),
        tmp_path / "v.mp4",
        fps=5.0,
        max_seconds=1.0,
    )
    assert s.frames == 5


def test_zones_for_uses_the_catalog() -> None:
    catalog = load_catalog(REPO_ROOT / "configs/cameras.yaml")
    camera_id, zones = zones_for(catalog, "checkout")
    assert camera_id == "cam-002"
    assert [z.zone_id for z in zones] == ["cam-002:register_1", "cam-002:register_2"]
    with pytest.raises(KeyError, match="not in the catalog"):
        zones_for(catalog, "roof")
