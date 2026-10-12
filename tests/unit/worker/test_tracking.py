"""ByteTrack adapter on synthetic detections: ids persist, gaps, classes, low scores."""

from __future__ import annotations

from vms.worker.detector import Detection
from vms.worker.tracking import CameraTracker, TrackerSettings


def box(x: float, cls: str = "person", conf: float = 0.9, y: float = 100) -> Detection:
    return Detection(cls, conf, x, y, x + 50, y + 100)


def ids(tracker: CameraTracker, dets: list[Detection]) -> dict[int, int]:
    """detection index -> local track id"""
    return {b.det_index: b.local_id for b in tracker.update(dets)}


def test_moving_objects_keep_their_ids() -> None:
    tr = CameraTracker(TrackerSettings())
    seen = [ids(tr, [box(10 + 8 * f), box(600 - 8 * f, "car", y=300)]) for f in range(10)]
    assert all(s == seen[0] for s in seen)
    assert len(set(seen[0].values())) == 2


def test_gap_shorter_than_buffer_keeps_id_longer_gets_a_new_one() -> None:
    settings = TrackerSettings(fps=5.0, buffer_s=1.0)  # 5 frames
    tr = CameraTracker(settings)
    first = ids(tr, [box(10)])[0]
    for _ in range(3):
        assert ids(tr, []) == {}
    assert ids(tr, [box(10)]) == {0: first}  # re-found after 3 missing frames

    for _ in range(8):
        ids(tr, [])
    ids(tr, [box(10)])  # unconfirmed new track (ByteTrack confirms on the second match)
    assert ids(tr, [box(10)]) != {0: first}


def test_low_confidence_detection_extends_a_track_but_does_not_start_one() -> None:
    tr = CameraTracker(TrackerSettings())
    first = ids(tr, [box(10)])[0]
    assert ids(tr, [box(12, conf=0.15)]) == {0: first}  # second-round match
    # A lone weak box far away never becomes a track.
    for _ in range(3):
        assert 1 not in ids(tr, [box(14), box(900, conf=0.15)])


def test_untracked_classes_are_ignored() -> None:
    tr = CameraTracker(TrackerSettings(classes=["person"]))
    out = [ids(tr, [box(10, "traffic light"), box(300)]) for _ in range(3)]
    assert all(set(o) == {1} for o in out)


def test_new_track_is_confirmed_on_its_second_frame() -> None:
    tr = CameraTracker(TrackerSettings())
    ids(tr, [box(10)])  # first frame of a tracker: activated at once
    assert 1 not in ids(tr, [box(12), box(500)])  # a newcomer is not reported yet
    assert 1 in ids(tr, [box(14), box(502)])
