"""Zone state machine on synthetic trajectories: crossings, jitter, lost and re-found tracks,
loiter, and the anchor point."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from vms.worker.zones import (
    EventType,
    ZoneDef,
    ZoneSettings,
    ZoneTracker,
    anchor,
    point_in_polygon,
)

T0 = datetime(2026, 10, 12, 9, 0, tzinfo=UTC)
DT = 0.2  # 5 FPS
DOOR = ZoneDef("cam:door", "door", ((0.4, 0.4), (0.6, 0.4), (0.6, 0.6), (0.4, 0.6)))
IN, OUT = (0.5, 0.5), (0.1, 0.1)


def t(i: int) -> datetime:
    return T0 + timedelta(seconds=i * DT)


def tracker(min_frames: int = 2, loiter_s: float | None = None, lost_s: float = 1.0):
    return ZoneTracker([DOOR], ZoneSettings(min_frames=min_frames, loiter_s=loiter_s), lost_s)


def run(zt: ZoneTracker, path: list[tuple[float, float] | None], track: str = "a"):
    """Feed one point per frame (None: the track is not visible); collect all events."""
    events = []
    for i, point in enumerate(path):
        events += zt.update(t(i), {} if point is None else {track: point})
    return events


def kinds(events) -> list[tuple[EventType, int]]:  # type: ignore[no-untyped-def]
    return [(e.event_type, round((e.ts - T0).total_seconds() / DT)) for e in events]


def test_walk_in_then_out_reports_first_frame_on_each_side() -> None:
    events = run(tracker(), [OUT, OUT, IN, IN, IN, OUT, OUT, OUT])
    assert kinds(events) == [(EventType.ENTER, 2), (EventType.EXIT, 5)]
    assert {(e.track_id, e.zone_id) for e in events} == {("a", "cam:door")}


def test_boundary_jitter_emits_nothing() -> None:
    assert run(tracker(), [OUT, IN, OUT, IN, OUT, IN, OUT]) == []


def test_one_frame_blip_inside_a_zone_is_ignored_but_a_real_visit_is_not() -> None:
    assert kinds(run(tracker(), [OUT, IN, OUT, OUT, IN, IN])) == [(EventType.ENTER, 4)]


def test_track_first_seen_inside_enters() -> None:
    assert kinds(run(tracker(), [IN, IN, IN])) == [(EventType.ENTER, 0)]


def test_min_frames_one_reacts_immediately() -> None:
    assert kinds(run(tracker(min_frames=1), [OUT, IN, OUT])) == [
        (EventType.ENTER, 1),
        (EventType.EXIT, 2),
    ]


def test_lost_and_refound_within_the_buffer_keeps_the_visit() -> None:
    # Occluded for 4 frames (0.8 s < 1 s) while inside: no exit, no second enter.
    events = run(tracker(), [IN, IN, None, None, None, None, IN, IN])
    assert kinds(events) == [(EventType.ENTER, 0)]


def test_lost_for_good_while_inside_exits_at_last_sighting() -> None:
    events = run(tracker(), [IN, IN, IN, None, None, None, None, None, None, None])
    assert kinds(events) == [(EventType.ENTER, 0), (EventType.EXIT, 2)]


def test_lost_while_outside_emits_nothing_and_a_new_sighting_starts_fresh() -> None:
    zt = tracker()
    assert run(zt, [OUT, OUT, None, None, None, None, None, None]) == []
    assert zt.open_tracks == 0


def test_loiter_once_per_visit() -> None:
    zt = tracker(loiter_s=1.0)
    path = [IN] * 10 + [OUT, OUT] + [IN] * 8
    events = kinds(run(zt, path))
    assert events == [
        (EventType.ENTER, 0),
        (EventType.LOITER, 5),  # 1.0 s after the enter
        (EventType.EXIT, 10),
        (EventType.ENTER, 12),
        (EventType.LOITER, 17),
    ]


def test_tracks_and_zones_are_independent() -> None:
    lot = ZoneDef("cam:lot", "lot", ((0.0, 0.0), (0.3, 0.0), (0.3, 0.3), (0.0, 0.3)))
    zt = ZoneTracker([DOOR, lot], ZoneSettings(min_frames=2, loiter_s=None), 1.0)
    events = []
    for i in range(3):
        events += zt.update(t(i), {"a": IN, "b": (0.1, 0.1)})
    assert sorted((e.track_id, e.zone_id) for e in events) == [("a", "cam:door"), ("b", "cam:lot")]


def test_set_zones_drops_state_for_removed_zones() -> None:
    zt = tracker()
    run(zt, [IN, IN])
    zt.set_zones([])
    assert zt.update(t(3), {"a": OUT}) == []


@pytest.mark.parametrize(
    ("point", "inside"),
    [((0.5, 0.5), True), ((0.39, 0.5), False), ((0.5, 0.61), False), ((0.55, 0.45), True)],
)
def test_point_in_polygon(point: tuple[float, float], inside: bool) -> None:
    assert point_in_polygon(point, DOOR.polygon) is inside


@pytest.mark.parametrize("point", [(0.5, 1.0), (1.0, 0.3), (0.0, 0.0), (0.3, 0.0)])
def test_points_on_the_frame_border_are_inside_a_full_frame_zone(
    point: tuple[float, float],
) -> None:
    full = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    assert point_in_polygon(point, full)


def test_point_in_concave_polygon() -> None:
    l_shape = ((0.0, 0.0), (0.6, 0.0), (0.6, 0.2), (0.2, 0.2), (0.2, 0.6), (0.0, 0.6))
    assert point_in_polygon((0.1, 0.5), l_shape)
    assert not point_in_polygon((0.4, 0.4), l_shape)  # in the notch


def test_anchor_is_the_bottom_centre_normalized() -> None:
    assert anchor(100, 50, 300, 650, 1280, 720) == pytest.approx((200 / 1280, 650 / 720))
