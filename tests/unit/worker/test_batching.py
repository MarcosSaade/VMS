"""Cross-camera batching in the worker (task 5.5): frames claimed together are detected in one
`detect_batch` call per image shape, tracked in capture order per camera, and written one by
one, with the same results as one frame at a time and per-frame failure handling."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import ValidationError

from tests.unit.worker.conftest import Harness, jpeg, make_harness
from vms.worker.detector import Detection
from vms.worker.settings import QueueSettings


def run(coro):  # type: ignore[no-untyped-def]
    return asyncio.run(coro)


def batched(infer_batch: int = 4) -> Harness:
    return make_harness(
        QueueSettings(max_cameras=4, claim_batch=4, infer_batch=infer_batch, max_attempts=2)
    )


def process_all(h: Harness) -> None:
    idle = 0
    while idle < 2:
        idle = 0 if run(h.worker.step()) else idle + 1


def person(x: float, conf: float = 0.9) -> Detection:
    return Detection("person", conf, x, 20.0, x + 8.0, 40.0)


def load(h: Harness) -> None:
    """Two cameras, four frames each, a person walking right on each."""
    for cam in ("cam-a", "cam-b"):
        for seq in range(4):
            h.queue.add(cam, seq, seq * 0.2)


def scripted(h: Harness) -> None:
    # The fake detector answers by call order; claim order is (ts, seq) across cameras.
    h.detector.script = [[person(4 + 2 * (k // 2))] for k in range(8)]


def test_infer_batch_cannot_exceed_claim_batch() -> None:
    with pytest.raises(ValidationError, match="infer_batch"):
        QueueSettings(claim_batch=2, infer_batch=4)


def test_frames_of_several_cameras_share_one_forward_pass() -> None:
    h = batched(4)
    load(h)
    process_all(h)
    assert h.detector.batches == [4, 4]
    assert all(h.queue.status(f"{c}-{s}") == "done" for c in ("cam-a", "cam-b") for s in range(4))
    assert h.sample("vms_worker_infer_batch_frames_count") == 2
    assert h.sample("vms_worker_infer_batch_frames_sum") == 8
    assert h.sample("vms_worker_stage_duration_seconds_count", stage="infer") == 2
    assert h.sample("vms_worker_stage_duration_seconds_count", stage="decode") == 8


def test_batched_results_equal_one_frame_at_a_time() -> None:
    results = []
    for size in (1, 4):
        h = batched(size)
        load(h)
        scripted(h)
        process_all(h)
        results.append(
            {
                fid: (
                    [(d.class_name, d.x1, d.track_id is not None) for d in r.detections],
                    [t.track_id.split(":")[2] for t in r.tracks],
                )
                for fid, r in h.queue.results.items()
            }
        )
    assert results[0] == results[1]
    # Each camera has one track that persists over its four frames.
    assert {v[1][0] for k, v in results[1].items() if k.startswith("cam-a")} == {"1"}


def test_two_frames_of_one_camera_in_a_batch_are_tracked_in_order() -> None:
    h = batched(4)
    for seq in range(4):
        h.queue.add("cam-a", seq, seq * 0.2)
    h.detector.script = [[person(4 + 2 * f)] for f in range(4)]
    process_all(h)
    assert h.detector.batches == [4]
    assert h.sample("vms_worker_late_frames_total") == 0
    track_ids = {t.track_id for r in h.queue.results.values() for t in r.tracks}
    assert len(track_ids) == 1


def test_mixed_image_shapes_get_one_call_each() -> None:
    h = batched(4)
    load(h)
    h.fetcher.objects["cam-b-0"] = jpeg(width=96, height=48)
    process_all(h)
    assert h.detector.batches == [3, 1, 4]  # first claim: 3 frames 64x48, 1 frame 96x48
    assert h.queue.results["cam-b-0"].width == 96


def test_a_bad_frame_fails_alone() -> None:
    h = batched(4)
    load(h)
    h.fetcher.objects["cam-a-1"] = b"not a jpeg"
    h.fetcher.fail_keys.add("cam-b-0")
    run(h.worker.step())  # leases, claims the four oldest frames, processes them
    assert h.queue.status("cam-a-1") == "failed"
    assert h.queue.status("cam-b-0") == "pending"  # fetch errors retry
    assert h.queue.status("cam-a-0") == "done"
    assert h.detector.batches == [2]


def test_a_model_error_retries_the_whole_batch() -> None:
    h = batched(4)
    load(h)
    h.detector.error = RuntimeError("out of memory")
    run(h.worker.step())
    assert sorted(r.status for r in h.queue.rows.values()) == ["pending"] * 8
    assert {r.attempts for r in h.queue.rows.values()} == {0, 1}
    assert h.sample("vms_worker_frames_total", outcome="retry") == 4
    h.detector.error = None
    process_all(h)
    assert all(r.status == "done" for r in h.queue.rows.values())


def test_each_frame_trace_has_the_shared_forward_pass() -> None:
    h = batched(4)
    for cam in ("cam-a", "cam-b"):
        h.queue.add(cam, 0, 0.0, traceparent=None)
    process_all(h)
    spans = h.spans.get_finished_spans()
    frames = {s.context.span_id: s for s in spans if s.name == "worker.frame"}
    infers = [s for s in spans if s.name == "worker.infer"]
    assert len(frames) == 2
    assert len(infers) == 2
    assert {s.parent.span_id for s in infers if s.parent} == set(frames)
    assert {s.attributes["batch_size"] for s in infers} == {2}
    assert len({(s.start_time, s.end_time) for s in infers}) == 1  # one call, shown twice
    for name in ("worker.fetch", "worker.decode", "worker.track", "worker.write"):
        children = [s for s in spans if s.name == name]
        assert {s.parent.span_id for s in children if s.parent} == set(frames), name
