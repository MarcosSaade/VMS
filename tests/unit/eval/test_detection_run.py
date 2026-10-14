"""The detection eval run with a fake detector: frames as the worker sees them, boxes scaled to
match, a results JSON that cites the subset."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray

from tests.unit.datasets.conftest import jpeg
from vms.common.settings import load_settings
from vms.eval.detection.run import SubsetNotReadyError, run
from vms.eval.detection.settings import DetectionEvalSettings
from vms.eval.detection.subset import GtImage, SubsetManifest, write_subset
from vms.worker.detector import Detection


class BoxesByWidth:
    """Returns fixed boxes for each input width: a perfect detector for the fixture images."""

    model_version = "fake@0"

    def __init__(self, boxes: dict[int, list[Detection]]) -> None:
        self.boxes = boxes
        self.shapes: list[tuple[int, ...]] = []

    def detect(self, image: NDArray[np.uint8]) -> list[Detection]:
        self.shapes.append(image.shape)
        return self.boxes[image.shape[1]]


def write_fixture(root: Path) -> DetectionEvalSettings:
    images_dir = root / "images"
    coco = jpeg(40, 30, 90)
    mot = jpeg(64, 48, 120)
    entries = [
        GtImage(
            id="coco-000000000001",
            source="coco",
            member="val2017/000000000001.jpg",
            sha256="",
            width=40,
            height=30,
            boxes=[(4, 4, 20, 20), (0, 0, 40, 30)],
            classes=["dog", "person"],
            ignore=[False, True],
            crowd=[False, True],
        ),
        GtImage(
            id="mot17-02-000001",
            source="mot17",
            member="MOT17/train/MOT17-02-FRCNN/img1/000001.jpg",
            sha256="",
            width=64,
            height=48,
            boxes=[(8, 8, 40, 40)],
            classes=["person"],
            ignore=[False],
            crowd=[False],
        ),
    ]
    stored = []
    for entry, data in zip(entries, [coco, mot], strict=True):
        img = entry.model_copy(update={"sha256": hashlib.sha256(data).hexdigest()})
        img.path(images_dir).parent.mkdir(parents=True, exist_ok=True)
        img.path(images_dir).write_bytes(data)
        stored.append(img)
    subset_hash = write_subset(root / "subset.jsonl", stored)
    manifest = SubsetManifest(
        version=1,
        spec_sha256="s",
        subset_sha256=subset_hash,
        coco_annotations_sha256="a",
        seed=1,
        images={"coco": 1, "mot17": 1},
        boxes={"coco": 1, "mot17": 1},
        ignored={"coco": 1, "mot17": 0},
    )
    (root / "manifest.json").write_text(manifest.model_dump_json())
    return DetectionEvalSettings.model_validate(
        {
            "paths": {
                "spec": "spec.yaml",
                "subset": "subset.jsonl",
                "manifest": "manifest.json",
                "images_dir": "images",
            },
            "results_dir": "results",
            "predictions_dir": "predictions",
            "mot_frame": {"width": 32, "height": 24, "jpeg_quality": 85},
            "bootstrap": {"resamples": 50},
        }
    )


def test_run_scores_a_perfect_detector_on_worker_sized_frames(tmp_path: Path) -> None:
    settings = write_fixture(tmp_path)
    detector = BoxesByWidth(
        {
            40: [
                Detection("dog", 0.9, 4, 4, 20, 20),
                Detection("person", 0.8, 1, 1, 30, 20),  # inside the crowd: ignored
            ],
            # The MOT frame was resized from 64x48 to 32x24: its box halves.
            32: [Detection("person", 0.9, 4, 4, 20, 20)],
        }
    )
    path = run(settings, tmp_path, detector=detector)
    assert detector.shapes == [(30, 40, 3), (30, 40, 3), (24, 32, 3)]  # first call is warmup
    result = json.loads(path.read_text())
    assert result["kind"] == "eval.detection"
    assert path.parent == tmp_path / "results"
    coco, mot = result["metrics"]["sources"]["coco"], result["metrics"]["sources"]["mot17"]
    # precision is tp / (tp + fp + eps), so "perfect" is 1 - 2e-16
    assert coco["map50_95"]["value"] == pytest.approx(1.0)
    assert mot["map50_95"]["value"] == pytest.approx(1.0)
    assert coco["per_class"] == {
        "dog": {"ap50": pytest.approx(1.0), "ap50_95": pytest.approx(1.0), "n_pos": 1}
    }
    assert (coco["gt_boxes"], coco["ignore_regions"], coco["predictions"]) == (1, 1, 2)
    assert result["metrics"]["subset"]["partial"] is False
    assert result["metrics"]["model_version"] == "fake@0"
    dumped = gzip.decompress((tmp_path / result["metrics"]["predictions_file"]).read_bytes())
    lines = [json.loads(line) for line in dumped.decode().splitlines()]
    assert [line["id"] for line in lines] == ["coco-000000000001", "mot17-02-000001"]
    assert lines[1] == {
        "id": "mot17-02-000001",
        "boxes": [[4, 4, 20, 20]],
        "scores": [0.9],
        "classes": ["person"],
    }


def test_unscaled_boxes_would_fail_on_mot_frames(tmp_path: Path) -> None:
    settings = write_fixture(tmp_path)
    detector = BoxesByWidth(
        {40: [Detection("dog", 0.9, 4, 4, 20, 20)], 32: [Detection("person", 0.9, 8, 8, 40, 40)]}
    )
    result = json.loads(run(settings, tmp_path, detector=detector).read_text())
    assert result["metrics"]["sources"]["mot17"]["map50"]["value"] == 0.0


def test_run_refuses_missing_images(tmp_path: Path) -> None:
    settings = write_fixture(tmp_path)
    next((tmp_path / "images" / "mot17").iterdir()).unlink()
    with pytest.raises(SubsetNotReadyError, match="eval-detection-data"):
        run(settings, tmp_path, detector=BoxesByWidth({}))


def test_sources_and_max_images_limit_the_run(tmp_path: Path) -> None:
    settings = write_fixture(tmp_path).model_copy(update={"sources": ["mot17"], "max_images": 1})
    detector = BoxesByWidth({32: []})
    result = json.loads(run(settings, tmp_path, detector=detector).read_text())
    assert list(result["metrics"]["sources"]) == ["mot17"]
    assert result["metrics"]["sources"]["mot17"]["map50"]["value"] == 0.0
    assert result["metrics"]["subset"]["partial"] is True


def test_bootstrap_seed_ignores_the_shared_vms_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VMS_SEED", "1234")
    root = Path(__file__).resolve().parents[3]
    settings = load_settings(DetectionEvalSettings, "eval-detection", config_dir=root / "configs")
    assert settings.bootstrap.seed == 20261014
