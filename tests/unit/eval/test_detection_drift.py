"""Detection eval under drift: the transform reaches the detector, ground truth is unchanged,
and the summary pairs every condition with the clean run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import ValidationError

from tests.unit.eval.test_detection_run import write_fixture
from vms.eval.detection.drift import Condition, DriftEvalScenario, run_drift
from vms.eval.detection.run import drifted
from vms.eval.detection.settings import DetectionEvalSettings, EvalDrift
from vms.worker.detector import Detection


class SeesOnlyInTheLight:
    """The fixture's perfect boxes while the image is bright; nothing once it is dark."""

    model_version = "fake@0"

    def __init__(self) -> None:
        self.means: list[float] = []

    def detect(self, image: NDArray[np.uint8]) -> list[Detection]:
        self.means.append(float(image.mean()))
        if image.mean() < 50:
            return []
        if image.shape[1] == 40:
            return [Detection("dog", 0.9, 4, 4, 20, 20)]
        return [Detection("person", 0.9, 4, 4, 20, 20)]


def test_drift_eval_pairs_each_condition_with_clean(tmp_path: Path) -> None:
    settings = write_fixture(tmp_path)
    scenario = DriftEvalScenario(
        name="t",
        conditions=[
            Condition(name="dark", drift=EvalDrift(kind="darken", strength=0.2)),
            Condition(name="noisy", drift=EvalDrift(kind="noise", strength=2.0)),
        ],
    )
    detector = SeesOnlyInTheLight()
    path = run_drift(scenario, tmp_path, settings, detector=detector)
    result = json.loads(path.read_text())
    assert result["kind"] == "eval.detection.drift"
    clean = result["metrics"]["clean"]["sources"]
    assert clean["coco"]["map50_95"]["value"] == pytest.approx(1.0)
    dark, noisy = result["metrics"]["conditions"]
    assert dark["drift"] == {"kind": "darken", "strength": 0.2, "seed": 20261016}
    for source in ("coco", "mot17"):
        assert dark["sources"][source]["map50_95"]["value"] == 0.0
        assert dark["sources"][source]["delta_map50_95"]["value"] == pytest.approx(-1.0)
        assert dark["sources"][source]["relative_change_map50_95"] == pytest.approx(-1.0)
        assert noisy["sources"][source]["delta_map50_95"]["value"] == pytest.approx(0.0)
    # Every run's results file exists and records its drift.
    run_file = json.loads((tmp_path / dark["results_file"]).read_text())
    assert run_file["metrics"]["drift"]["kind"] == "darken"
    # clean (with warmup), dark (with warmup), noisy (with warmup): 3 x 3 calls
    assert len(detector.means) == 9


def test_drifted_pixels_are_seeded_per_image() -> None:
    settings = DetectionEvalSettings.model_validate({"drift": {"kind": "noise", "strength": 5.0}})
    image = np.full((30, 40, 3), 128, dtype=np.uint8)
    a = drifted(image, "img-1", settings)
    assert np.array_equal(a, drifted(image, "img-1", settings))
    assert not np.array_equal(a, drifted(image, "img-2", settings))
    assert drifted(image, "img-1", settings.model_copy(update={"drift": None})) is image


def test_drift_settings_are_photometric_and_in_range() -> None:
    with pytest.raises(ValidationError):
        EvalDrift(kind="shift", strength=0.1)  # type: ignore[arg-type]
    with pytest.raises(ValidationError, match="out of range"):
        EvalDrift(kind="darken", strength=1.0)


def test_partial_runs_are_refused(tmp_path: Path) -> None:
    settings = write_fixture(tmp_path).model_copy(update={"max_images": 1})
    scenario = DriftEvalScenario(
        name="t", conditions=[Condition(name="d", drift=EvalDrift(kind="darken", strength=0.5))]
    )
    with pytest.raises(ValueError, match="whole subset"):
        run_drift(scenario, tmp_path, settings, detector=SeesOnlyInTheLight())
