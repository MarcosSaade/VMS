import pytest

from vms.common.metrics import Metrics
from vms.loadgen.recorder import Recorder, percentiles
from vms.loadgen.transport import Outcome, SendResult, from_status

OK = from_status(202)


def test_percentiles_in_ms() -> None:
    stats = percentiles([i / 1000 for i in range(1, 101)])  # 1..100 ms
    assert stats["p50"] == pytest.approx(50.5)
    assert stats["p99"] == pytest.approx(99.01)
    assert stats["max"] == 100.0
    assert percentiles([]) == {"p50": None, "p95": None, "p99": None, "mean": None, "max": None}


def test_warmup_samples_are_excluded_from_the_summary() -> None:
    rec = Recorder(Metrics("loadgen"), warmup_s=2.0)
    for due in (0.0, 1.0, 2.0, 3.0):
        rec.sent(due, lag_s=0.001)
        rec.completed(due, OK, latency_s=0.010)
    rec.dropped(0.5)
    rec.dropped(2.5)
    s = rec.summary(steady_s=2.0, offered_fps=1.0)
    assert s["frames"]["sent"] == 2
    assert s["frames"]["accepted"] == 2
    assert s["frames"]["dropped"] == 1
    assert rec.total_sent == 4
    assert s["sent_fps"] == 1.0


def test_outcomes_failures_and_incomplete() -> None:
    rec = Recorder(Metrics("loadgen"), warmup_s=0.0)
    results = [
        OK,
        from_status(429),
        SendResult(Outcome.FAILED, None, "timeout", attempts=3),
        from_status(500),
    ]
    for i, result in enumerate(results):
        rec.sent(float(i), 0.0)
        rec.completed(float(i), result, 0.02)
    rec.sent(4.0, 0.0)  # never completed
    s = rec.summary(steady_s=5.0, offered_fps=None)
    assert s["frames"] == {
        "sent": 5,
        "accepted": 1,
        "rejected": 1,
        "failed": 2,
        "dropped": 0,
        "incomplete": 1,
        "retries": 2,
    }
    assert s["failure_reasons"] == {"500": 1, "timeout": 1}
    assert s["error_rate"] == pytest.approx(0.4)
    # Latency percentiles are over accepted frames only.
    assert s["send_latency_ms"]["p50"] == pytest.approx(20.0)


def test_per_second_spread() -> None:
    rec = Recorder(Metrics("loadgen"), warmup_s=0.0)
    for due in (0.1, 0.2, 1.1, 1.2, 1.3, 1.4):  # 2 frames in second 0, 4 in second 1
        rec.sent(due, 0.0)
    assert rec.summary(steady_s=2.0, offered_fps=None)["sent_fps_per_second_std"] == 1.0


def test_prometheus_names() -> None:
    metrics = Metrics("loadgen")
    rec = Recorder(metrics, warmup_s=0.0)
    rec.sent(0.0, 0.0)
    rec.completed(0.0, SendResult(Outcome.FAILED, None, "timeout"), 0.1)
    text = metrics.exposition().decode()
    for name in (
        "vms_loadgen_sent_frames_total 1.0",
        "vms_loadgen_accepted_frames_total 0.0",
        "vms_loadgen_rejected_frames_total 0.0",
        'vms_loadgen_failed_frames_total{reason="timeout"} 1.0',
        "vms_loadgen_dropped_frames_total 0.0",
        "vms_loadgen_send_duration_seconds_bucket",
        "vms_loadgen_schedule_lag_seconds_bucket",
        "vms_loadgen_in_flight_requests 0.0",
    ):
        assert name in text
