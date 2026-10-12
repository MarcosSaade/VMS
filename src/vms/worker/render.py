"""`python -m vms.worker render`: draw detections, tracks and zone events on a clip, for review.

Runs the worker's own per-frame path (detector, then `pipeline.track_frame` with ByteTrack and
the zone state machine) on frames sampled the way loadgen sends them (every clip_fps / fps-th
frame, resized to 1280x720), so the video shows what the worker would store. No database:
zones come from the camera catalog (`configs/cameras.yaml`) by camera name.

Drawn per frame:
- zones: outline and light fill, filled stronger while a track is inside (after its enter);
- tracked objects: a box in the track's colour, `#<n> <class> <conf>`, and the anchor point
  (bottom centre) that zone membership uses; the anchor turns red while inside a zone;
- stored detections without a track (untracked classes, or tracks not yet confirmed): thin grey;
- weak detections that are dropped (below `tracker.store_min_conf`, no track): not drawn;
- zone events as they happen, kept on screen for a moment, plus a header line.

The video plays at the sampling rate (default 5 FPS), so in real time.
"""

from __future__ import annotations

import hashlib
import math
import shutil
import subprocess
import tempfile
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import cv2
import numpy as np
from numpy.typing import NDArray

from vms.storage.catalog import CameraCatalog
from vms.storage.work import DetectionRow
from vms.worker.detector import Detector
from vms.worker.pipeline import CameraState, Tracker, track_frame
from vms.worker.service import decode_jpeg
from vms.worker.tracking import TrackerSettings
from vms.worker.zones import EventType, ZoneDef, ZoneSettings, ZoneTracker

T0 = datetime(2026, 1, 1, tzinfo=UTC)
EVENT_HOLD_S = 2.0
FONT = cv2.FONT_HERSHEY_SIMPLEX
GREY = (170, 170, 170)
RED = (40, 40, 230)
ZONE = (60, 200, 255)
EVENT_COLOURS = {
    EventType.ENTER: (80, 220, 80),
    EventType.EXIT: (60, 160, 255),
    EventType.LOITER: (230, 90, 230),
}


@dataclass
class Segment:
    """One clip played as one camera."""

    clip_id: str
    camera_id: str
    camera_name: str
    clip_fps: float
    frames: list[bytes]  # every clip frame, as loadgen would send them
    zones: list[ZoneDef]


@dataclass
class SegmentSummary:
    clip_id: str
    camera_name: str
    frames: int = 0
    stored_detections: int = 0
    tracked_detections: int = 0
    tracks: set[str] = field(default_factory=set)
    events: Counter[str] = field(default_factory=Counter)


def zones_for(catalog: CameraCatalog, camera_name: str) -> tuple[str, list[ZoneDef]]:
    """(camera_id, zones) of a catalogued camera."""
    for camera in catalog.cameras:
        if camera.name == camera_name:
            return camera.camera_id, [
                ZoneDef(camera.zone_id(z), z.name, z.polygon) for z in camera.zones
            ]
    names = ", ".join(c.name for c in catalog.cameras)
    raise KeyError(f"camera {camera_name!r} is not in the catalog ({names})")


def colour(track_id: str) -> tuple[int, int, int]:
    digest = hashlib.sha256(track_id.encode()).digest()
    hue = digest[0] * 180 // 256
    hsv = np.array([[[hue, 200, 255]]], dtype=np.uint8)
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def _label(img: NDArray[np.uint8], text: str, x: int, y: int, bg: tuple[int, int, int]) -> None:
    (w, h), base = cv2.getTextSize(text, FONT, 0.5, 1)
    y = max(y, h + 4)
    cv2.rectangle(img, (x, y - h - base - 2), (x + w + 4, y), bg, -1)
    cv2.putText(img, text, (x + 2, y - base), FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw(
    image: NDArray[np.uint8],
    detections: tuple[DetectionRow, ...],
    zones: list[ZoneDef],
    occupied: set[tuple[str, str]],
    recent: list[tuple[datetime, str, tuple[int, int, int]]],
    header: str,
) -> NDArray[np.uint8]:
    height, width = image.shape[:2]
    out = image.copy()
    overlay = image.copy()
    inside_tracks = {t for t, _ in occupied}
    for zone in zones:
        pts = np.array([[x * width, y * height] for x, y in zone.polygon], dtype=np.int32)
        busy = any(z == zone.zone_id for _, z in occupied)
        cv2.fillPoly(overlay, [pts], ZONE)
        alpha = 0.35 if busy else 0.12
        out = cast(NDArray[np.uint8], cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0))
        overlay = out.copy()
        cv2.polylines(out, [pts], True, ZONE, 2, cv2.LINE_AA)
        x0, y0 = pts.min(axis=0)
        _label(out, f"{zone.name}{' (occupied)' if busy else ''}", int(x0), int(y0) - 2, ZONE)

    for d in detections:
        p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
        if d.track_id is None:
            cv2.rectangle(out, p1, p2, GREY, 1)
            cv2.putText(out, d.class_name, (p1[0], p1[1] - 3), FONT, 0.4, GREY, 1, cv2.LINE_AA)
            continue
        c = colour(d.track_id)
        cv2.rectangle(out, p1, p2, c, 2)
        local = d.track_id.rsplit(":", 1)[-1]
        _label(out, f"#{local} {d.class_name} {d.confidence:.2f}", p1[0], p1[1] - 2, c)
        foot = (int((d.x1 + d.x2) / 2), int(d.y2))
        cv2.circle(out, foot, 5, RED if d.track_id in inside_tracks else c, -1)
        cv2.circle(out, foot, 5, (0, 0, 0), 1)

    cv2.rectangle(out, (0, 0), (width, 26), (0, 0, 0), -1)
    cv2.putText(out, header, (8, 18), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    for i, (_, text, c) in enumerate(recent[-6:]):
        _label(out, text, width - 330, 52 + 24 * i, c)
    return out


def render(
    segments: list[Segment],
    detector: Detector,
    tracker_factory: Callable[[TrackerSettings], Tracker],
    tracking: TrackerSettings,
    zone_settings: ZoneSettings,
    out_path: Path,
    *,
    fps: float,
    max_seconds: float | None = None,
) -> list[SegmentSummary]:
    """Render every segment into one video at `fps`; returns per-segment counts."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summaries: list[SegmentSummary] = []
    writer: cv2.VideoWriter | None = None
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.mp4"
        for seg in segments:
            state = CameraState(
                session="render",
                tracker=tracker_factory(tracking),
                zones=ZoneTracker(seg.zones, zone_settings, tracking.buffer_s),
                zones_loaded_at=0.0,
            )
            summary = SegmentSummary(seg.clip_id, seg.camera_name)
            occupied: set[tuple[str, str]] = set()
            recent: list[tuple[datetime, str, tuple[int, int, int]]] = []
            n = math.floor(len(seg.frames) * fps / seg.clip_fps)
            if max_seconds is not None:
                n = min(n, math.floor(max_seconds * fps))
            for k in range(n):
                image = decode_jpeg(seg.frames[math.floor(k * seg.clip_fps / fps + 1e-9)])
                height, width = image.shape[:2]
                ts = T0 + timedelta(seconds=k / fps)
                tracked = track_frame(
                    detector.detect(image),
                    camera_id=seg.camera_id,
                    ts=ts,
                    seq=k,
                    state=state,
                    width=width,
                    height=height,
                    store_min_conf=tracking.store_min_conf,
                )
                for e in tracked.zone_events:
                    key = (e.track_id, e.zone_id)
                    if e.event_type is EventType.ENTER:
                        occupied.add(key)
                    elif e.event_type is EventType.EXIT:
                        occupied.discard(key)
                    zone_name = next(z.name for z in seg.zones if z.zone_id == e.zone_id)
                    local = e.track_id.rsplit(":", 1)[-1]
                    at = (e.ts - T0).total_seconds()
                    text = f"t={at:5.1f}s {e.event_type.value} {zone_name} #{local}"
                    recent.append((ts, text, EVENT_COLOURS[e.event_type]))
                    summary.events[e.event_type.value] += 1
                recent = [r for r in recent if (ts - r[0]).total_seconds() < EVENT_HOLD_S]
                summary.frames += 1
                summary.stored_detections += len(tracked.detections)
                tracked_now = [d for d in tracked.detections if d.track_id is not None]
                summary.tracked_detections += len(tracked_now)
                summary.tracks.update(d.track_id for d in tracked_now if d.track_id)
                header = (
                    f"{seg.camera_name} ({seg.camera_id}) | {seg.clip_id} | "
                    f"t={k / fps:5.1f}s | tracks now {len(tracked_now)} | "
                    f"total {len(summary.tracks)} | {detector.model_version}"
                )
                frame = draw(image, tracked.detections, seg.zones, occupied, recent, header)
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(raw), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height)
                    )
                if (frame.shape[1], frame.shape[0]) != (width, height):
                    frame = cast(NDArray[np.uint8], cv2.resize(frame, (width, height)))
                writer.write(frame)
            summaries.append(summary)
        if writer is not None:
            writer.release()
        _finish(raw, out_path)
    return summaries


def _finish(raw: Path, out_path: Path) -> None:
    """Re-encode to H.264 (plays in browsers and default players) if ffmpeg is installed;
    otherwise keep OpenCV's MPEG-4 Part 2 file (plays in VLC)."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        shutil.copyfile(raw, out_path)
        return
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-loglevel",
            "error",
            "-i",
            str(raw),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "23",
            str(out_path),
        ],
        check=True,
        timeout=600,
    )
