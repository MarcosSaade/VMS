"""Paired comparison of two detection eval runs from their saved detections (task 5.2, D32)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.unit.eval.test_detection_run import BoxesByWidth, write_fixture
from vms.eval.detection.compare import (
    IncomparableRunsError,
    compare,
    load_run,
    run_compare,
)
from vms.eval.detection.run import run
from vms.worker.detector import Detection

PERFECT = {
    40: [Detection("dog", 0.9, 4, 4, 20, 20)],
    32: [Detection("person", 0.9, 4, 4, 20, 20)],
}
# Same boxes on COCO; on MOT the person box is 4 px too tall (IoU 0.8): a hit at the IoU
# thresholds 0.50 to 0.80, a miss at 0.85, 0.90 and 0.95.
LOOSE = PERFECT | {32: [Detection("person", 0.9, 4, 4, 20, 24)]}


def two_runs(tmp_path: Path) -> tuple[Path, Path]:
    settings = write_fixture(tmp_path)
    a = run(settings, tmp_path, name="a", detector=BoxesByWidth(PERFECT))
    b = run(settings, tmp_path, name="b", detector=BoxesByWidth(LOOSE))
    return a, b


def test_compare_reports_the_paired_drop_and_the_d32_verdict(tmp_path: Path) -> None:
    a, b = two_runs(tmp_path)
    result = compare(load_run(a), load_run(b), tmp_path, name="cmp")
    coco, mot = result.metrics["sources"]["coco"], result.metrics["sources"]["mot17"]
    assert coco["delta_map50_95"]["value"] == pytest.approx(0.0)
    assert coco["d32"]["pass"] is True
    assert mot["delta_map50"]["value"] == pytest.approx(0.0)  # still a hit at IoU 0.5
    assert mot["delta_map50_95"]["value"] == pytest.approx(-0.3)  # 3 of 10 thresholds
    assert mot["d32"] == {
        "drop_map50_95": pytest.approx(0.3),
        "drop_upper": pytest.approx(0.3),  # one image: every resample is the same
        "pass": False,
    }
    assert result.metrics["d32_pass"] is False
    assert mot["runs_reported_map50_95"]["base"] == pytest.approx(mot["base"]["map50_95"]["value"])
    assert result.kind == "eval.detection.compare"


def test_compare_writes_a_results_file(tmp_path: Path) -> None:
    a, _ = two_runs(tmp_path)
    path = run_compare(a, a, tmp_path, None)
    data = json.loads(path.read_text())
    assert path.parent == tmp_path / "results"
    assert data["metrics"]["d32_pass"] is True


def test_runs_scored_on_different_terms_are_refused(tmp_path: Path) -> None:
    a, b = two_runs(tmp_path)
    base, other = load_run(a), load_run(b)
    changed = other.model_copy(update={"config": other.config | {"max_dets_per_class": 5}})
    with pytest.raises(IncomparableRunsError, match="max_dets_per_class"):
        compare(base, changed, tmp_path)
    metrics = other.metrics | {"subset": other.metrics["subset"] | {"partial": True}}
    with pytest.raises(IncomparableRunsError, match="partial"):
        compare(base, other.model_copy(update={"metrics": metrics}), tmp_path)
    metrics = other.metrics | {"subset": other.metrics["subset"] | {"subset_sha256": "x"}}
    with pytest.raises(IncomparableRunsError, match="different subsets"):
        compare(base, other.model_copy(update={"metrics": metrics}), tmp_path)
