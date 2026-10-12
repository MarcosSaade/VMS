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
