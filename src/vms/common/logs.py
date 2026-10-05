"""Structured JSON logging with trace correlation.

Every record becomes one JSON object per line, with the OpenTelemetry trace and span IDs of the
active span, so a log line can be joined to its trace in Grafana (Tempo) and logs can be
filtered by field instead of grepped. Extra fields go through the standard `extra=` argument:

    log = get_logger(__name__)
    log.info("frame stored", extra={"camera_id": "cam-01", "frame_id": fid, "bytes": n})

Uses the standard library `logging` module, so third-party libraries' logs come out as JSON too.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, TextIO

from opentelemetry import trace

# Attributes every LogRecord has; anything else on a record came from `extra=`.
_STANDARD_ATTRS = frozenset(
    vars(logging.LogRecord("", logging.INFO, "", 0, "", None, None)).keys()
    | {"message", "asctime", "taskName"}
)


class JsonFormatter(logging.Formatter):
    def __init__(self, service: str) -> None:
        super().__init__()
        self._service = service

    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "service": self._service,
        }
        span_context = trace.get_current_span().get_span_context()
        if span_context.is_valid:
            entry["trace_id"] = trace.format_trace_id(span_context.trace_id)
            entry["span_id"] = trace.format_span_id(span_context.span_id)
        for key, value in vars(record).items():
            if key not in _STANDARD_ATTRS and key not in entry:
                entry[key] = value
        if record.exc_info:
            entry["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str, ensure_ascii=False)


def configure_logging(service: str, level: str = "INFO", stream: TextIO | None = None) -> None:
    """Route all logging through one JSON handler on the root logger. Safe to call again."""
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter(service))
    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
