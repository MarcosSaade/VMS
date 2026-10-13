import asyncio
import urllib.parse
from typing import Any

import pytest

from vms.bench.prom import HttpPrometheus, Value, backlog, read_window, render
from vms.bench.settings import PromQuery


def test_render_substitutes_the_window_in_whole_seconds() -> None:
    assert render("rate(x[$window])", 30.4) == "rate(x[30s])"
    assert render('sum(x{a="b"})', 30) == 'sum(x{a="b"})'  # braces are left alone
    assert render("rate(x[$window])", 0.2) == "rate(x[1s])"


class Scripted:
    def __init__(self, answers: dict[str, list[tuple[dict[str, str], Value]]]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, float]] = []

    async def query(self, promql: str, at_unix_s: float) -> list[tuple[dict[str, str], Value]]:
        self.calls.append((promql, at_unix_s))
        return self.answers.get(promql, [])


def test_read_window_shapes_results() -> None:
    prom = Scripted(
        {
            "a[20s]": [({}, 1.234567)],
            "b[20s]": [({"job": "worker"}, 2.0), ({"job": "ingest"}, None)],
        }
    )
    queries = {
        "a": PromQuery(query="a[$window]"),
        "b": PromQuery(query="b[$window]", by="job"),
        "missing": PromQuery(query="c"),
    }
    out = asyncio.run(read_window(prom, queries, end_unix_s=1000.0, window_s=20))
    assert out == {"a": 1.2346, "b": {"ingest": None, "worker": 2.0}, "missing": None}
    assert list(out["b"]) == ["ingest", "worker"]  # type: ignore[arg-type]  # sorted by label
    assert {at for _, at in prom.calls} == {1000.0}  # every query at the window's end


def test_backlog_is_none_without_a_worker() -> None:
    assert asyncio.run(backlog(Scripted({}), 1.0)) is None
    prom = Scripted({"max(vms_worker_backlog_frames)": [({}, 7.0)]})
    assert asyncio.run(backlog(prom, 1.0)) == 7.0


def _fetch(response: dict[str, Any], urls: list[str]) -> Any:
    def fetch(url: str) -> Any:
        urls.append(url)
        return response

    return fetch


def test_http_prometheus_builds_the_instant_query_and_maps_non_finite() -> None:
    urls: list[str] = []
    response = {
        "status": "success",
        "data": {
            "resultType": "vector",
            "result": [
                {"metric": {"stage": "infer"}, "value": [1000, "0.25"]},
                {"metric": {"stage": "fetch"}, "value": [1000, "NaN"]},
                {"metric": {"stage": "x"}, "value": [1000, "+Inf"]},
            ],
        },
    }
    prom = HttpPrometheus("http://prom:9090/", _fetch(response, urls))
    series = asyncio.run(prom.query('sum(rate(x{a="b"}[30s]))', 1234.5))
    assert series == [
        ({"stage": "infer"}, 0.25),
        ({"stage": "fetch"}, None),
        ({"stage": "x"}, None),
    ]
    query = urllib.parse.parse_qs(urllib.parse.urlparse(urls[0]).query)
    assert urls[0].startswith("http://prom:9090/api/v1/query?")
    assert query == {"query": ['sum(rate(x{a="b"}[30s]))'], "time": ["1234.500"]}


def test_http_prometheus_scalar_and_errors() -> None:
    scalar = {"status": "success", "data": {"resultType": "scalar", "result": [1, "3"]}}
    assert asyncio.run(HttpPrometheus("http://p", _fetch(scalar, [])).query("3", 1)) == [({}, 3.0)]
    error = {"status": "error", "error": "parse error"}
    with pytest.raises(RuntimeError, match="parse error"):
        asyncio.run(HttpPrometheus("http://p", _fetch(error, [])).query("(", 1))
