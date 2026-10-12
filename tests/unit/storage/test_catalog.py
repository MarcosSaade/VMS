"""Camera catalog: the committed YAML is valid and matches the few-shot bank; bad input fails."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from vms.common.settings import load_settings
from vms.query.fewshots import load_fewshots
from vms.query.settings import QuerySettings
from vms.storage.catalog import CameraCatalog, Zone, is_simple, load_catalog, signed_area

REPO_ROOT = Path(__file__).resolve().parents[3]
SQUARE = [[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.1, 0.5]]


def catalog(**camera: Any) -> dict[str, Any]:
    base = {"camera_id": "cam-001", "name": "entrance", "site": "s", "timezone": "UTC"}
    return {"version": 1, "cameras": [base | camera]}


def test_committed_catalog_loads() -> None:
    cat = load_catalog(REPO_ROOT / "configs/cameras.yaml")
    assert {c.name for c in cat.cameras} == {"entrance", "checkout", "parking_lot", "loading_dock"}
    entrance = next(c for c in cat.cameras if c.name == "entrance")
    assert entrance.zone_id(entrance.zones[0]) == "cam-001:front_door"


def test_every_camera_and_zone_in_the_fewshot_bank_is_in_the_catalog() -> None:
    settings = load_settings(QuerySettings, "query", config_dir=REPO_ROOT / "configs")
    cat = load_catalog(REPO_ROOT / "configs/cameras.yaml")
    cameras = {c.name for c in cat.cameras}
    zones = {z.name for c in cat.cameras for z in c.zones}
    used_cameras: set[str] = set()
    used_zones: set[str] = set()
    for example in load_fewshots(REPO_ROOT / settings.assets.fewshots):
        if example.plan.filters is not None:
            used_cameras.update(example.plan.filters.cameras)
            used_zones.update(example.plan.filters.zones)
    assert used_cameras
    assert used_zones
    assert used_cameras <= cameras
    assert used_zones <= zones


def test_reader_grants_in_migration_match_the_validator_allowlist() -> None:
    # The DB enforces the same allowlist as the validator (migration 0003); they must agree.
    path = REPO_ROOT / "src/vms/storage/migrations/versions/0003_query_reader_role.py"
    spec = importlib.util.spec_from_file_location("m0003", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    settings = load_settings(QuerySettings, "query", config_dir=REPO_ROOT / "configs")
    grants = {t: set(cols) for t, cols in module.GRANTS.items()}
    assert grants == {t: set(cols) for t, cols in settings.sql.tables.items()}


@pytest.mark.parametrize("tz", ["Not/AZone", "", "../etc/passwd", "/usr/share/zoneinfo/UTC"])
def test_unknown_timezone_is_rejected(tz: str) -> None:
    with pytest.raises(ValidationError, match="timezone"):
        CameraCatalog.model_validate(catalog(timezone=tz))


@pytest.mark.parametrize("bad", ["cam 1", "../x", "", "a" * 65])
def test_unsafe_camera_id_is_rejected(bad: str) -> None:
    with pytest.raises(ValidationError):
        CameraCatalog.model_validate(catalog(camera_id=bad))


def test_duplicate_camera_names_and_ids_are_rejected() -> None:
    data = catalog()
    data["cameras"].append(data["cameras"][0] | {"camera_id": "cam-002"})
    with pytest.raises(ValidationError, match="duplicate camera name: entrance"):
        CameraCatalog.model_validate(data)
    data["cameras"][1] |= {"camera_id": "cam-001", "name": "other"}
    with pytest.raises(ValidationError, match="duplicate camera_id: cam-001"):
        CameraCatalog.model_validate(data)


def test_zone_names_are_unique_across_cameras() -> None:
    # Plans name a zone without its camera, so two cameras cannot share a zone name.
    data = catalog(zones=[{"name": "door", "polygon": SQUARE}])
    data["cameras"].append(
        {
            "camera_id": "cam-002",
            "name": "back",
            "site": "s",
            "timezone": "UTC",
            "zones": [{"name": "door", "polygon": SQUARE}],
        }
    )
    with pytest.raises(ValidationError, match="duplicate zone name: door"):
        CameraCatalog.model_validate(data)


@pytest.mark.parametrize(
    ("polygon", "message"),
    [
        ([[0.1, 0.1], [0.5, 0.1]], "at least 3"),
        ([[0.1, 0.1], [1.2, 0.1], [0.5, 0.5]], "outside the normalized frame"),
        ([[0.1, 0.1], [0.3, 0.3], [0.5, 0.5]], "zero area"),  # collinear
        ([[0.1, 0.1], [0.5, 0.5], [0.5, 0.1], [0.1, 0.5]], "intersect"),  # bowtie
        ([[0.1, 0.1], [0.5, 0.1], [0.5, 0.5], [0.5, 0.1]], "intersect"),  # repeated vertex
    ],
)
def test_invalid_polygons_are_rejected(polygon: list[list[float]], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        Zone.model_validate({"name": "z", "polygon": polygon})


def test_polygon_on_the_frame_border_is_valid() -> None:
    full = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    assert Zone(name="all", polygon=full).polygon == full
    assert signed_area(full) == pytest.approx(1.0)
    assert signed_area(full[::-1]) == pytest.approx(-1.0)


def test_concave_polygon_is_simple() -> None:
    l_shape = ((0.0, 0.0), (0.6, 0.0), (0.6, 0.2), (0.2, 0.2), (0.2, 0.6), (0.0, 0.6))
    assert is_simple(l_shape)
    # An edge that touches a non-adjacent vertex is not simple.
    assert not is_simple(((0.0, 0.0), (0.4, 0.0), (0.2, 0.0001), (0.4, 0.4), (0.2, 0.0)))


@pytest.mark.parametrize("name", ["Entrance", "front door", "1st", "a-b"])
def test_names_must_be_lower_snake_case(name: str) -> None:
    with pytest.raises(ValidationError):
        CameraCatalog.model_validate(catalog(name=name))


def test_unknown_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        CameraCatalog.model_validate(catalog(tz="UTC"))
