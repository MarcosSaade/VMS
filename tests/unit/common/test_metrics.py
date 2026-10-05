import pytest

from vms.common.metrics import Metrics, metric_name


def test_metric_name_format() -> None:
    assert metric_name("ingest", "request_duration", "seconds") == (
        "vms_ingest_request_duration_seconds"
    )


@pytest.mark.parametrize(
    ("component", "metric", "unit"),
    [
        ("Ingest", "frames", "total"),  # uppercase
        ("ingest-gw", "frames", "total"),  # dash
        ("ingest", "frames__x", "total"),  # double underscore
        ("ingest", "1frames", "total"),  # leading digit
        ("ingest", "latency", "ms"),  # non-base unit
        ("ingest", "latency_seconds", "seconds"),  # unit repeated
        ("ingest", "total", "total"),
    ],
)
def test_bad_names_are_rejected(component: str, metric: str, unit: str) -> None:
    with pytest.raises(ValueError, match=r"snake_case|unknown unit|already ends"):
        metric_name(component, metric, unit)


def test_counter_requires_total_and_others_forbid_it() -> None:
    metrics = Metrics("ingest")
    with pytest.raises(ValueError, match="counters"):
        metrics.counter("frames", "frames", "x")
    with pytest.raises(ValueError, match="reserved"):
        metrics.gauge("queue_depth", "total", "x")
    with pytest.raises(ValueError, match="reserved"):
        metrics.histogram("latency", "total", "x")


def test_invalid_component_fails_at_construction() -> None:
    with pytest.raises(ValueError, match="snake_case"):
        Metrics("Bad-Component")


def test_exposition_contains_named_metrics() -> None:
    metrics = Metrics("ingest")
    metrics.counter("received_frames", "total", "Frames", labels=["camera_id"]).labels(
        camera_id="cam-01"
    ).inc(3)
    metrics.gauge("queue_depth", "messages", "Depth").set(7)
    metrics.histogram("request_duration", "seconds", "Latency", buckets=[0.1, 1.0]).observe(0.05)
    text = metrics.exposition().decode()
    assert 'vms_ingest_received_frames_total{camera_id="cam-01"} 3.0' in text
    assert "vms_ingest_queue_depth_messages 7.0" in text
    assert 'vms_ingest_request_duration_seconds_bucket{le="0.1"} 1.0' in text


def test_same_name_returns_same_metric_and_type_conflicts_fail() -> None:
    metrics = Metrics("ingest")
    first = metrics.counter("received_frames", "total", "Frames")
    assert metrics.counter("received_frames", "total", "Frames") is first
    hist = metrics.histogram("request_duration", "seconds", "Latency")
    assert metrics.histogram("request_duration", "seconds", "Latency") is hist
    metrics.gauge("batch_size", "frames", "x")
    with pytest.raises(ValueError, match="already registered"):
        metrics.histogram("batch_size", "frames", "x")


def test_registries_are_isolated() -> None:
    Metrics("ingest").counter("received_frames", "total", "Frames")
    Metrics("ingest").counter("received_frames", "total", "Frames")  # no duplicate error


def test_process_metrics_are_opt_in() -> None:
    assert b"process_" not in Metrics("ingest").exposition()
    assert b"process_cpu_seconds_total" in Metrics("ingest", process_metrics=True).exposition()
