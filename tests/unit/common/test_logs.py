import io
import json
import logging

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

from vms.common.logs import configure_logging, get_logger

pytestmark = pytest.mark.usefixtures("restore_root_logger")


def capture(level: str = "INFO") -> io.StringIO:
    stream = io.StringIO()
    configure_logging("ingest", level, stream=stream)
    return stream


def lines(stream: io.StringIO) -> list[dict[str, object]]:
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def test_one_json_object_per_line_with_core_fields() -> None:
    stream = capture()
    get_logger("vms.test").info("frame %s stored", "f1")
    [entry] = lines(stream)
    assert entry["msg"] == "frame f1 stored"
    assert entry["level"] == "INFO"
    assert entry["logger"] == "vms.test"
    assert entry["service"] == "ingest"
    assert str(entry["ts"]).endswith("+00:00")
    assert "trace_id" not in entry  # no active span


def test_extra_fields_are_top_level_and_non_json_values_stringified() -> None:
    stream = capture()
    get_logger("vms.test").info("x", extra={"camera_id": "cam-01", "bytes": 10, "obj": object()})
    [entry] = lines(stream)
    assert entry["camera_id"] == "cam-01"
    assert entry["bytes"] == 10
    assert str(entry["obj"]).startswith("<object")


def test_trace_and_span_ids_from_active_span() -> None:
    stream = capture()
    tracer = TracerProvider().get_tracer("test")
    with tracer.start_as_current_span("work") as span:
        get_logger("vms.test").info("inside")
        ctx = span.get_span_context()
    [entry] = lines(stream)
    assert entry["trace_id"] == trace.format_trace_id(ctx.trace_id)
    assert entry["span_id"] == trace.format_span_id(ctx.span_id)
    assert len(str(entry["trace_id"])) == 32


def test_exceptions_are_included() -> None:
    stream = capture()
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        get_logger("vms.test").exception("failed")
    [entry] = lines(stream)
    assert "RuntimeError: boom" in str(entry["exc_info"])


def test_level_filtering_and_reconfigure_replaces_handler() -> None:
    capture("INFO")
    stream = capture("WARNING")
    log = get_logger("vms.test")
    log.info("hidden")
    log.warning("shown")
    assert [e["msg"] for e in lines(stream)] == ["shown"]
    assert len(logging.getLogger().handlers) == 1
