"""Dashboards as code: the committed JSON is what the code generates, every panel is complete,
and every metric a panel queries is one the services actually declare."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from prometheus_client.core import CollectorRegistry

from vms.common.clock import FakeClock
from vms.common.ids import UlidGenerator
from vms.common.metrics import Metrics
from vms.ingest.app import create_app as ingest_app
from vms.ingest.service import FrameIngestor
from vms.ingest.settings import FrameLimits
from vms.observe.dashboards import (
    GRID_WIDTH,
    Dashboard,
    all_dashboards,
    dashboard_json,
    main,
    panel_queries,
)
from vms.query.app import create_app as query_app
from vms.query.engine import PlanEngine
from vms.worker.app import create_app as worker_app
from vms.worker.service import Worker
from vms.worker.settings import QueueSettings

DASHBOARDS = all_dashboards()
METRIC = re.compile(r"\b(vms_[a-z0-9_]+)")
SUFFIXES = ("_total", "_bucket", "_count", "_sum", "_created")


def family(name: str) -> str:
    for suffix in SUFFIXES:
        if name.endswith(suffix):
            return name.removesuffix(suffix)
    return name


def families(registry: CollectorRegistry) -> set[str]:
    return {m.name for m in registry.collect()}


def declared_metrics() -> set[str]:
    """Metric families the services register, from their real constructors (collaborators
    mocked: constructors only store them)."""
    clock = FakeClock()
    ids = UlidGenerator(clock)
    ingest, worker, query = Metrics("ingest"), Metrics("worker"), Metrics("query")
    ingestor = FrameIngestor(MagicMock(), MagicMock(), FrameLimits(), ingest, clock=clock, ids=ids)
    ingest_app(ingestor, ingest, max_jpeg_bytes=1000, ready_checks={})
    Worker(
        MagicMock(),
        MagicMock(),
        MagicMock(),
        QueueSettings(),
        worker,
        worker_id="w",
        clock=clock,
        ids=ids,
        executor=MagicMock(),
    )
    worker_app(worker, ready_checks={}, on_startup=MagicMock(), on_shutdown=MagicMock())
    PlanEngine(
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        MagicMock(),
        default_limit=100,
        max_limit=1000,
        clock=clock,
        ids=ids,
        metrics=query,
        tracer=MagicMock(),
        logger=MagicMock(),
    )
    query_app(MagicMock(), MagicMock(), MagicMock(), MagicMock(), query, ready_checks={})
    return families(ingest.registry) | families(worker.registry) | families(query.registry)


def test_committed_json_matches_the_code(tmp_path: Path) -> None:
    # `make dashboards` regenerates; this fails if someone edited one side only.
    assert main(["--check"]) == 0
    assert main(["--out", str(tmp_path / "missing"), "--check"]) == 1


def test_every_metric_queried_is_declared_by_a_service() -> None:
    declared = declared_metrics()
    used = {
        family(name)
        for d in DASHBOARDS
        for _, expr in panel_queries(d, include_optional=True)
        for name in METRIC.findall(expr)
    }
    assert used, "no vms_ metrics found in the dashboards"
    assert used <= declared, sorted(used - declared)


@pytest.mark.parametrize("dashboard", DASHBOARDS, ids=lambda d: d.uid)
def test_panels_are_complete_and_laid_out_on_the_grid(dashboard: Dashboard) -> None:
    data = dashboard_json(dashboard)
    panels = [p for top in data["panels"] for p in [top, *top.get("panels", [])]]
    ids = [p["id"] for p in panels]
    assert len(ids) == len(set(ids))
    for p in panels:
        pos = p["gridPos"]
        assert pos["x"] >= 0, p["title"]
        assert pos["x"] + pos["w"] <= GRID_WIDTH, p["title"]
        if p["type"] == "row":
            continue
        assert p["title"]
        assert len(p["description"]) > 40, f"{p['title']}: say what decision it informs"
        if p["type"] != "text":
            assert p["targets"], p["title"]
            assert p["fieldConfig"]["defaults"]["unit"], p["title"]
            assert all(t["datasource"]["uid"] == "prometheus" for t in p["targets"])
    # Visible panels do not overlap.
    cells: set[tuple[int, int]] = set()
    for p in data["panels"]:
        pos = p["gridPos"]
        mine = {
            (x, y)
            for x in range(pos["x"], pos["x"] + pos["w"])
            for y in range(pos["y"], pos["y"] + pos["h"])
        }
        assert not cells & mine, p["title"]
        cells |= mine


def test_dashboards_have_unique_uids_and_link_to_each_other() -> None:
    uids = [d.uid for d in DASHBOARDS]
    assert len(uids) == len(set(uids))
    assert {"vms-pipeline", "vms-inference", "vms-query", "vms-resources", "vms-drift"} == set(uids)
    for d in DASHBOARDS:
        data = dashboard_json(d)
        assert data["links"][0]["tags"] == ["vms"]
        assert "vms" in data["tags"]


def test_only_gpu_and_placeholder_panels_may_be_empty() -> None:
    optional = {p.title for d in DASHBOARDS for p in d.panels() if p.optional}
    assert optional == {"GPU utilization", "GPU memory used", "What goes here"}
