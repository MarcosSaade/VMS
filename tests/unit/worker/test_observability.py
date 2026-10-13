"""Worker traces and metrics (task 2.1): the frame's trace continues from ingest through the
queue, and the metrics dashboards and benchmarks read are emitted."""

from __future__ import annotations

import asyncio

from opentelemetry.sdk.trace import ReadableSpan

from tests.unit.worker.conftest import T0, Harness
from vms.worker.service import FRESHNESS_BUCKETS_S

TRACE_ID = "4bf92f3577b34da6a3ce929d0e0e4736"
INGEST_SPAN_ID = "00f067aa0ba902b7"


def traceparent(sampled: bool = True) -> str:
    return f"00-{TRACE_ID}-{INGEST_SPAN_ID}-{'01' if sampled else '00'}"


def by_name(spans: tuple[ReadableSpan, ...]) -> dict[str, ReadableSpan]:
    return {s.name: s for s in spans}


def test_frame_trace_continues_the_ingest_span_stored_on_the_row(h: Harness) -> None:
    h.queue.add("cam-a", 0, 0.0, traceparent=traceparent())
    asyncio.run(h.worker.step())

    spans = by_name(h.spans.get_finished_spans())
    frame, wait = spans["worker.frame"], spans["worker.queue_wait"]
    for span in (frame, wait):
        assert span.context is not None
        assert format(span.context.trace_id, "032x") == TRACE_ID
        assert span.parent is not None
        assert format(span.parent.span_id, "016x") == INGEST_SPAN_ID
    # Stages are children of the frame span, in the same trace.
    assert frame.context is not None
    for stage in ("fetch", "decode", "stats", "infer", "track", "write"):
        child = spans[f"worker.{stage}"]
        assert child.parent is not None
        assert child.parent.span_id == frame.context.span_id, stage
    # Queue wait: from ingest's receive time (T0) to the claim (the clock reads T0 + 60 s).
    assert wait.start_time == int(T0.timestamp() * 1e9)
    assert wait.end_time == int(h.clock.now().timestamp() * 1e9)
    assert frame.attributes is not None
    assert frame.attributes["outcome"] == "done"
    assert frame.attributes["freshness_s"] == 60.0


def test_unsampled_frames_record_no_spans(h: Harness) -> None:
    # Loadgen and ingest decided not to sample this frame; the worker keeps that decision.
    h.queue.add("cam-a", 0, 0.0, traceparent=traceparent(sampled=False))
    asyncio.run(h.worker.step())
    assert h.queue.status("cam-a-0") == "done"
    assert h.spans.get_finished_spans() == ()


def test_frame_without_trace_context_starts_a_new_trace(h: Harness) -> None:
    h.queue.add("cam-a", 0, 0.0, traceparent=None)
    h.queue.add("cam-a", 1, 0.2, traceparent="not-a-traceparent")
    asyncio.run(h.worker.step())

    frames = [s for s in h.spans.get_finished_spans() if s.name == "worker.frame"]
    assert len(frames) == 2
    assert all(s.parent is None for s in frames)
    assert "worker.queue_wait" not in by_name(h.spans.get_finished_spans())


def test_key_metrics_are_emitted(h: Harness) -> None:
    """The names dashboards (task 2.2) and the benchmark runner (2.3) read."""
    h.queue.add("cam-a", 0, 0.0)
    h.queue.add("cam-a", 1, 0.2)
    h.clock.advance(0.5)  # the claim happens 60.5 s after the first frame was received
    asyncio.run(h.worker.step())

    # Backlog is read before the claim: both frames pending, the oldest waiting 60.5 s.
    assert h.sample("vms_worker_backlog_frames") == 2
    assert h.sample("vms_worker_backlog_age_seconds") == 60.5
    assert h.sample("vms_worker_frames_total", outcome="done") == 2
    assert h.sample("vms_worker_capture_to_done_seconds_count") == 2
    assert h.sample("vms_worker_capture_to_done_seconds_sum") == 60.5 + 60.3
    assert h.sample("vms_worker_queue_wait_seconds_count") == 2
    for stage in ("claim", "fetch", "decode", "stats", "infer", "track", "write"):
        assert h.sample("vms_worker_stage_duration_seconds_count", stage=stage) >= 1, stage
    assert h.sample("vms_worker_leased_cameras") == 1


def test_freshness_buckets_have_an_edge_at_the_slo() -> None:
    # ARCHITECTURE 8: freshness p95 < 2 s. A bucket edge at 2 s makes "share within SLO" exact.
    assert 2.0 in FRESHNESS_BUCKETS_S
    assert list(FRESHNESS_BUCKETS_S) == sorted(FRESHNESS_BUCKETS_S)
