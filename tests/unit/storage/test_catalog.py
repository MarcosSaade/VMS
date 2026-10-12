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
from vms.storage.catalog import (
    CameraCatalog,
    Zone,
    catalog_header,
    dump_catalog,
    is_simple,
    load_catalog,
    signed_area,
    with_zones,
    write_catalog,
)

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


# --- writing (zone editor) ----------------------------------------------------------------------


def test_shipped_catalog_round_trips_through_dump(tmp_path: Path) -> None:
    source = (REPO_ROOT / "configs/cameras.yaml").read_text(encoding="utf-8")
    catalog = load_catalog(REPO_ROOT / "configs/cameras.yaml")
    text = dump_catalog(catalog, catalog_header(source))
    assert text.startswith("# Camera catalog")
    path = tmp_path / "cameras.yaml"
    path.write_text(text, encoding="utf-8")
    assert load_catalog(path) == catalog


def test_with_zones_replaces_one_camera_and_validates(tmp_path: Path) -> None:
    path = tmp_path / "cameras.yaml"
    path.write_text((REPO_ROOT / "configs/cameras.yaml").read_text(encoding="utf-8"))
    catalog = load_catalog(path)
    drawn = [{"name": "bay_1", "polygon": [[0.1, 0.2], [0.512345678, 0.2], [0.3, 0.9]]}]
    updated = with_zones(catalog, "cam-004", drawn)
    write_catalog(path, updated)
    reloaded = load_catalog(path)
    dock = next(c for c in reloaded.cameras if c.camera_id == "cam-004")
    assert [z.name for z in dock.zones] == ["bay_1"]
    assert dock.zones[0].polygon[1] == (0.5123, 0.2)  # rounded to 4 decimals
    assert reloaded.cameras[0] == catalog.cameras[0]  # other cameras untouched
    assert path.read_text().startswith("# Camera catalog")  # header kept
    assert not (tmp_path / ".cameras.yaml.tmp").exists()


@pytest.mark.parametrize(
    ("zones", "fragment"),
    [
        ([{"name": "x", "polygon": [[0, 0], [1, 1], [1, 0], [0, 1]]}], "intersect"),
        ([{"name": "x", "polygon": [[0, 0], [1, 0]]}], "at least 3"),
        ([{"name": "Bad Name", "polygon": [[0, 0], [1, 0], [0, 1]]}], "pattern"),
        ([{"name": "front_door", "polygon": [[0, 0], [1, 0], [0, 1]]}], "duplicate zone name"),
        ([{"name": "x", "polygon": [[0, 0], [1.5, 0], [0, 1]]}], "outside"),
    ],
)
def test_with_zones_rejects_invalid_zones(zones: list[dict[str, object]], fragment: str) -> None:
    catalog = load_catalog(REPO_ROOT / "configs/cameras.yaml")
    with pytest.raises(ValueError, match=fragment):
        with_zones(catalog, "cam-004", zones)
    with pytest.raises(ValueError, match="unknown camera"):
        with_zones(catalog, "cam-999", [])
