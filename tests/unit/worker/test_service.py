"""Worker loop logic with fakes: ordering, results, failure handling, leases, shutdown."""

from __future__ import annotations

import asyncio

import pytest

from tests.unit.worker.conftest import Harness, jpeg
from vms.worker.service import DecodeError, Outcome, decode_jpeg


def run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


def test_frames_are_processed_in_capture_order_with_stats_and_detections(h: Harness) -> None:
    # Inserted out of order: ingest can receive them that way (D17, in-flight cap > 1).
    for seq, ts in [(2, 0.4), (0, 0.0), (1, 0.2)]:
        h.queue.add("cam-a", seq, ts)
    h.fetcher.objects["cam-a-0"] = jpeg(value=40, width=80, height=60)

    assert run(h.worker.step()) == 2  # claim_batch
    assert run(h.worker.step()) == 1
    assert run(h.worker.step()) == 0

    assert [h.queue.status(f"cam-a-{i}") for i in range(3)] == ["done"] * 3
    done_order = [k for k in h.queue.results]
    assert done_order == ["cam-a-0", "cam-a-1", "cam-a-2"]
    first = h.queue.results["cam-a-0"]
    assert (first.width, first.height) == (80, 60)
    assert first.brightness < 80  # mostly value 40
    assert first.contrast > 0
    assert first.blur_var > 0
    assert first.model_version == "fake@000000000000"
    # det_idx follows the detector's order (highest confidence first).
    assert [(d.det_idx, d.class_name) for d in first.detections] == [(0, "person"), (1, "car")]
    assert h.sample("vms_worker_frames_total", outcome="done") == 3
    assert h.sample("vms_worker_detections_total") == 6
    assert h.sample("vms_worker_late_frames_total") == 0


def test_undecodable_frame_fails_without_retry(h: Harness) -> None:
    h.queue.add("cam-a", 0, 0.0)
    h.fetcher.objects["cam-a-0"] = b"\xff\xd8\xff not really a jpeg"
    run(h.worker.step())
    assert h.queue.status("cam-a-0") == "failed"
    assert h.queue.rows["cam-a-0"].attempts == 1
    assert "decode" in (h.queue.rows["cam-a-0"].error or "")
    assert h.detector.seen == []
    assert h.sample("vms_worker_frames_total", outcome="failed") == 1


def test_fetch_errors_retry_until_max_attempts(h: Harness) -> None:
    h.queue.add("cam-a", 0, 0.0)
    h.fetcher.fail_keys.add("cam-a-0")
    run(h.worker.step())
    assert h.queue.status("cam-a-0") == "pending"
    run(h.worker.step())  # attempt 2 of max_attempts=2
    assert h.queue.status("cam-a-0") == "failed"
    assert h.sample("vms_worker_frames_total", outcome="retry") == 1
    assert h.sample("vms_worker_frames_total", outcome="failed") == 1


def test_model_errors_are_retried(h: Harness) -> None:
    h.queue.add("cam-a", 0, 0.0)
    h.detector.error = RuntimeError("CUDA out of memory")
    run(h.worker.step())
    assert h.queue.status("cam-a-0") == "pending"
    h.detector.error = None
    run(h.worker.step())
    assert h.queue.status("cam-a-0") == "done"


def test_lost_claim_writes_nothing(h: Harness) -> None:
    h.queue.add("cam-a", 0, 0.0)

    def taken_over() -> None:
        # Another worker took the camera while this one was inferring.
        row = h.queue.rows["cam-a-0"]
        row.claimed_by = "w2"

    h.detector.on_detect = taken_over
    run(h.worker.step())
    assert "cam-a-0" not in h.queue.results
    assert h.sample("vms_worker_frames_total", outcome="lost") == 1
    assert h.sample("vms_worker_detections_total") == 0


def test_write_error_requeues_and_unreported_failure_is_recovered_next_step(
    h: Harness,
) -> None:
    h.queue.add("cam-a", 0, 0.0)
    h.queue.fail_ops = {"complete", "fail"}  # the database went away mid-frame
    run(h.worker.step())
    assert h.queue.status("cam-a-0") == "processing"  # stuck under our claim for now
    assert h.sample("vms_worker_loop_errors_total", op="fail") == 1

    h.queue.fail_ops = set()
    run(h.worker.step())  # requeues first, then claims and completes it
    assert h.queue.calls.count("release_frames") == 1
    assert h.queue.status("cam-a-0") == "done"


def test_late_frames_are_stored_and_counted(h: Harness) -> None:
    h.queue.add("cam-a", 5, 1.0)
    run(h.worker.step())
    h.queue.add("cam-a", 4, 0.8)  # arrived after a later frame was processed
    run(h.worker.step())
    assert h.queue.status("cam-a-4") == "done"
    assert h.sample("vms_worker_late_frames_total") == 1


def test_leases_cap_cameras_and_renew_on_cadence(h: Harness) -> None:
    for cam in ["cam-a", "cam-b", "cam-c"]:
        h.queue.add(cam, 0, 0.0)
    run(h.worker.step())
    assert h.worker.leased_cameras == {"cam-a", "cam-b"}  # max_cameras=2
    assert h.sample("vms_worker_leased_cameras") == 2
    assert h.queue.status("cam-c-0") == "pending"

    renewals = h.queue.calls.count("renew")
    run(h.worker.step())
    assert h.queue.calls.count("renew") == renewals  # renew_every_s not elapsed
    h.clock.advance(5.0)
    run(h.worker.step())
    assert h.queue.calls.count("renew") == renewals + 1


def test_idle_worker_looks_for_new_cameras_every_step(h: Harness) -> None:
    assert run(h.worker.step()) == 0
    h.queue.add("cam-new", 0, 0.0)
    assert run(h.worker.step()) == 0  # leased now, claimed next step
    assert h.worker.leased_cameras == {"cam-new"}
    assert run(h.worker.step()) == 1


def test_stop_mid_batch_hands_back_unprocessed_frames(h: Harness) -> None:
    h.queue.add("cam-a", 0, 0.0)
    h.queue.add("cam-a", 1, 0.2)

    async def scenario() -> None:
        stop = asyncio.Event()
        h.detector.on_detect = stop.set  # SIGTERM arrives during the first frame
        await h.worker.step(stop)
        await h.worker.release()

    run(scenario())
    assert h.queue.status("cam-a-0") == "done"
    assert h.queue.status("cam-a-1") == "pending"
    assert h.queue.rows["cam-a-1"].attempts == 0  # a graceful hand-back is not an attempt
    assert h.queue.leases == {}


def test_run_stops_on_event_and_releases(h: Harness) -> None:
    h.queue.add("cam-a", 0, 0.0)

    async def scenario() -> None:
        stop = asyncio.Event()
        h.detector.on_detect = stop.set
        await asyncio.wait_for(h.worker.run(stop), timeout=5)

    run(scenario())
    assert h.queue.status("cam-a-0") == "done"
    assert h.queue.calls[-1] == "release"


def test_queue_outage_does_not_crash_the_loop(h: Harness) -> None:
    h.queue.fail_ops = {"renew"}

    async def scenario() -> None:
        stop = asyncio.Event()
        task = asyncio.create_task(h.worker.run(stop))
        while h.queue.calls.count("renew") < 2:  # it keeps trying with backoff
            await asyncio.sleep(0)
        stop.set()
        await asyncio.wait_for(task, timeout=5)

    run(scenario())


def test_decode_jpeg_rejects_garbage() -> None:
    assert decode_jpeg(jpeg()).shape == (48, 64, 3)
    with pytest.raises(DecodeError):
        decode_jpeg(b"nope")


def test_outcome_values_match_metric_labels() -> None:
    assert {o.value for o in Outcome} == {"done", "retry", "failed", "lost"}


# --- Tracking and zones -----------------------------------------------------------------

from vms.storage.work import ZoneRow  # noqa: E402
from vms.worker.detector import Detection  # noqa: E402

# Frames are 64x48 (conftest.jpeg); a person box whose bottom centre walks right.
LOWER_RIGHT = ZoneRow("cam-a:door", "door", ((0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)))


def person(x: float, conf: float = 0.9) -> Detection:
    return Detection("person", conf, x, 20.0, x + 8.0, 40.0)  # anchor y = 40/48 = 0.83


def process_all(h: Harness) -> None:
    """Step until two idle steps in a row (the first idle step may only lease cameras)."""
    idle = 0
    while idle < 2:
        idle = 0 if run(h.worker.step()) else idle + 1


def test_detections_get_stable_track_ids_and_tracks_are_reported(h: Harness) -> None:
    for seq in range(4):
        h.queue.add("cam-a", seq, seq * 0.2)
    h.detector.script = [
        [person(4 + 2 * f), Detection("traffic light", 0.9, 50, 1, 54, 9)] for f in range(4)
    ]
    process_all(h)

    track_ids = {
        d.track_id
        for r in h.queue.results.values()
        for d in r.detections
        if d.class_name == "person"
    }
    assert len(track_ids) == 1
    (track_id,) = track_ids
    camera, session, local = track_id.split(":")
    assert (camera, local) == ("cam-a", "1")
    assert len(session) == 8
    # Untracked classes are stored without a track.
    lights = [
        d for r in h.queue.results.values() for d in r.detections if d.class_name == "traffic light"
    ]
    assert len(lights) == 4
    assert all(d.track_id is None for d in lights)
    assert [t.track_id for r in h.queue.results.values() for t in r.tracks] == [track_id] * 4
    assert h.sample("vms_worker_track_sightings_total") == 4


def test_weak_detections_are_stored_only_when_tracked(h: Harness) -> None:
    for seq in range(3):
        h.queue.add("cam-a", seq, seq * 0.2)
    lone_weak = Detection("person", 0.15, 50, 0, 58, 10)
    h.detector.script = [
        [person(4), lone_weak],
        [person(6, conf=0.15), lone_weak],  # the tracked person turns weak for one frame
        [person(8), lone_weak],
    ]
    process_all(h)
    stored = [
        [(d.confidence, d.track_id is not None) for d in h.queue.results[f"cam-a-{i}"].detections]
        for i in range(3)
    ]
    assert stored == [
        [(0.9, True)],
        [(0.15, True)],  # kept: it belongs to a track
        [(0.9, True)],
    ]


def test_zone_events_are_written_with_the_frame(h: Harness) -> None:
    h.queue.zone_rows["cam-a"] = [LOWER_RIGHT]
    # Anchor x = (x + 4) / 64: 0.44, 0.47, then 0.53, 0.56, 0.59 (inside). Small steps, as
    # real motion at 5 FPS: ByteTrack matches by box overlap, so a jump would start a new track.
    xs = [24, 26, 30, 32, 34]
    for seq in range(len(xs)):
        h.queue.add("cam-a", seq, seq * 0.2)
    h.detector.script = [[person(x)] for x in xs]
    process_all(h)

    events = [
        (r.frame_id, e.event_type, e.zone_id, e.ts)
        for r in h.queue.results.values()
        for e in r.events
    ]
    ((frame_id, kind, zone_id, ts),) = events
    # Confirmed on frame 3 (min_frames=2), timestamped at frame 2, the first one inside.
    assert (frame_id, kind, zone_id) == ("cam-a-3", "zone_enter", "cam-a:door")
    assert ts == h.queue.rows["cam-a-2"].frame.ts
    event_ids = [e.event_id for r in h.queue.results.values() for e in r.events]
    assert len(event_ids[0]) == 26  # a ULID
    assert h.sample("vms_worker_zone_events_total", event_type="zone_enter") == 1


def test_late_and_retried_frames_are_stored_but_not_tracked(h: Harness) -> None:
    h.queue.add("cam-a", 5, 1.0)
    run(h.worker.step())
    h.queue.add("cam-a", 4, 0.8)  # arrived after a later frame was tracked
    run(h.worker.step())
    late = h.queue.results["cam-a-4"]
    assert late.detections
    assert all(d.track_id is None for d in late.detections)
    assert late.tracks == ()

    # A frame whose write failed comes back: the tracker has already seen it.
    h.queue.add("cam-a", 6, 1.2)
    h.queue.fail_ops = {"complete"}
    run(h.worker.step())
    h.queue.fail_ops = set()
    run(h.worker.step())
    assert h.queue.status("cam-a-6") == "done"
    assert h.queue.results["cam-a-6"].tracks == ()
    assert h.sample("vms_worker_late_frames_total") == 2


def test_losing_a_lease_drops_the_camera_state(h: Harness) -> None:
    h.queue.add("cam-a", 0, 0.0)
    process_all(h)
    first = h.queue.results["cam-a-0"].detections[0].track_id
    assert first is not None

    h.queue.leases["cam-a"] = "w2"  # another worker took it over
    h.clock.advance(5.0)  # next renew notices
    run(h.worker.step())
    h.queue.leases.pop("cam-a")  # and later gave it back
    h.queue.add("cam-a", 1, 0.2)
    process_all(h)
    second = h.queue.results["cam-a-1"].detections[0].track_id
    assert second is not None
    assert second.split(":")[1] != first.split(":")[1]  # a new tracker session


def test_idle_cameras_are_handed_back_so_new_cameras_get_a_slot(h: Harness) -> None:
    # A worker at max_cameras used to keep the leases of cameras that had stopped sending
    # forever, so cameras that started later were never processed (seen in task 9.3).
    for cam in ["cam-a", "cam-b"]:
        h.queue.add(cam, 0, 0.0)
    process_all(h)
    assert h.worker.leased_cameras == {"cam-a", "cam-b"}  # max_cameras=2
    h.queue.add("cam-c", 0, 0.0)
    h.clock.advance(5.0)
    process_all(h)
    assert h.queue.status("cam-c-0") == "pending"  # no room yet: idle for less than 60 s

    h.queue.add("cam-a", 1, 0.2)  # cam-a is still sending
    process_all(h)
    h.clock.advance(60.0)
    h.queue.add("cam-a", 2, 0.4)  # pending at renew time: never released with work left
    process_all(h)
    assert h.worker.leased_cameras == {"cam-a", "cam-c"}
    assert h.queue.status("cam-c-0") == "done"
    assert h.queue.status("cam-a-2") == "done"
    assert "cam-b" not in h.queue.leases
    assert h.sample("vms_worker_leased_cameras") == 2


def test_zones_are_reloaded_on_cadence_and_a_failed_load_is_not_fatal(h: Harness) -> None:
    h.queue.add("cam-a", 0, 0.0)
    h.queue.fail_ops = {"zones"}
    process_all(h)
    assert h.queue.status("cam-a-0") == "done"
    assert h.sample("vms_worker_loop_errors_total", op="zones") == 1

    h.queue.fail_ops = set()
    h.queue.add("cam-a", 1, 0.2)
    process_all(h)  # retried at once because no load has succeeded yet
    loads = h.queue.calls.count("zones")
    h.queue.add("cam-a", 2, 0.4)
    process_all(h)
    assert h.queue.calls.count("zones") == loads  # refresh_s not elapsed
    h.clock.advance(60.0)
    h.queue.add("cam-a", 3, 0.6)
    process_all(h)
    assert h.queue.calls.count("zones") == loads + 1


def test_a_time_gap_longer_than_the_buffer_starts_a_new_tracker(h: Harness) -> None:
    # ByteTrack's buffer counts frames, not time: without a reset it would join an object
    # seen before a pause (camera offline, old backlog) to whatever is there afterwards.
    h.queue.zone_rows["cam-a"] = [LOWER_RIGHT]
    times = [0.0, 0.2, 0.4, 3600.0, 3600.2, 3600.4]  # one hour offline
    for seq, ts in enumerate(times):
        h.queue.add("cam-a", seq, ts)
    h.detector.script = [[person(40)] for _ in times]  # same place before and after
    process_all(h)

    ids = [h.queue.results[f"cam-a-{i}"].detections[0].track_id for i in range(len(times))]
    assert ids[0] == ids[1] == ids[2]
    assert ids[3] == ids[4] == ids[5]
    assert ids[0] != ids[3]
    # The first visit is closed at its last sighting; the new track enters afresh.
    events = [(e.track_id, e.event_type, e.ts) for r in h.queue.results.values() for e in r.events]
    t = {i: h.queue.rows[f"cam-a-{i}"].frame.ts for i in range(len(times))}
    assert events == [
        (ids[0], "zone_enter", t[0]),
        (ids[0], "zone_exit", t[2]),
        (ids[3], "zone_enter", t[3]),
    ]
    assert h.sample("vms_worker_tracker_resets_total") == 1
