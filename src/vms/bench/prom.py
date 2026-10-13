"""Read a step's steady-state window from Prometheus.

Each configured query is a PromQL template evaluated once, at the window's end, with `$window`
replaced by the window length, so `rate(x[$window])` covers exactly the steady state. Quantiles
come from the services' histogram buckets (`histogram_quantile`), so they are interpolated
within a bucket: good enough to compare steps and to judge the 2 s freshness SLO (the freshness
buckets have an edge at 2 s), not exact percentiles.
"""

from __future__ import annotations

import asyncio
import json
import math
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from string import Template
from typing import Any, Protocol

from vms.bench.settings import PromQuery

FetchJson = Callable[[str], Any]
Value = float | None
Reading = Value | dict[str, Value]


class Prometheus(Protocol):
    async def query(self, promql: str, at_unix_s: float) -> list[tuple[dict[str, str], Value]]:
        """Instant query: (labels, value) per series. NaN and +-Inf come back as None."""
        ...


def http_fetch_json(timeout_s: float) -> FetchJson:
    def fetch(url: str) -> Any:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return json.load(resp)

    return fetch


class HttpPrometheus:
    """The Prometheus HTTP API. The blocking call runs in a thread, off the event loop."""

    def __init__(self, base_url: str, fetch_json: FetchJson) -> None:
        self._base = base_url.rstrip("/")
        self._fetch = fetch_json

    async def query(self, promql: str, at_unix_s: float) -> list[tuple[dict[str, str], Value]]:
        params = urllib.parse.urlencode({"query": promql, "time": f"{at_unix_s:.3f}"})
        data = await asyncio.to_thread(self._fetch, f"{self._base}/api/v1/query?{params}")
        if data.get("status") != "success":
            raise RuntimeError(f"Prometheus query failed: {data.get('error')!r} for {promql}")
        result = data["data"]
        if result["resultType"] == "scalar":
            return [({}, _number(result["result"][1]))]
        return [(series["metric"], _number(series["value"][1])) for series in result["result"]]


def _number(raw: str) -> Value:
    value = float(raw)
    return value if math.isfinite(value) else None


def render(template: str, window_s: float) -> str:
    return Template(template).safe_substitute(window=f"{max(1, round(window_s))}s")


async def read_window(
    prom: Prometheus, queries: Mapping[str, PromQuery], end_unix_s: float, window_s: float
) -> dict[str, Reading]:
    """Every query over the window ending at `end_unix_s`.

    A query without a `by` label gives one number (None when there is no data, e.g. no frames
    reached the worker); with `by`, a mapping from label value to number.
    """
    out: dict[str, Reading] = {}
    for name, q in queries.items():
        series = await prom.query(render(q.query, window_s), end_unix_s)
        if q.by is None:
            values = [v for _, v in series]
            out[name] = _round(values[0]) if values else None
        else:
            out[name] = {
                labels.get(q.by, ""): _round(v) for labels, v in sorted(series, key=_label(q.by))
            }
    return out


def _label(by: str) -> Callable[[tuple[dict[str, str], Value]], str]:
    return lambda item: item[0].get(by, "")


def _round(value: Value) -> Value:
    return None if value is None else round(value, 4)


async def backlog(prom: Prometheus, at_unix_s: float) -> float | None:
    """The worker backlog (frames pending or processing); None when no worker reports it."""
    series = await prom.query("max(vms_worker_backlog_frames)", at_unix_s)
    return series[0][1] if series else None


def as_float(value: Reading) -> float | None:
    return value if isinstance(value, float | int) else None
