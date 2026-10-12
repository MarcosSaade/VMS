"""Zone editor API: cameras and their demo clips, frames, saving through catalog validation."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.unit.query.conftest import REPO_ROOT
from vms.storage.catalog import load_catalog
from vms.zones.editor import create_app, demo_clips

CLIPS = ["mot17-09-mini", "mot17-04-mini", "commons-roncesvalles"]


@pytest.fixture
def catalog_path(tmp_path: Path) -> Path:
    path = tmp_path / "cameras.yaml"
    shutil.copy(REPO_ROOT / "configs/cameras.yaml", path)
    return path


@pytest.fixture
def client(tmp_path: Path, catalog_path: Path) -> Iterator[TestClient]:
    frames = tmp_path / "data/clips/mot17-09-mini/frames"
    frames.mkdir(parents=True)
    for i in range(3):
        (frames / f"{i:06d}.jpg").write_bytes(b"\xff\xd8" + bytes([i]))
    ui = tmp_path / "dist"
    ui.mkdir()
    (ui / "zones.html").write_text("<title>zone editor</title>")
    clips = demo_clips(load_catalog(catalog_path), CLIPS, "cam-")
    with TestClient(create_app(catalog_path, tmp_path / "data", clips, ui)) as c:
        yield c


def test_demo_clips_follow_loadgen_round_robin(catalog_path: Path) -> None:
    assert demo_clips(load_catalog(catalog_path), CLIPS, "cam-") == {
        "cam-001": "mot17-09-mini",
        "cam-002": "mot17-04-mini",
        "cam-003": "commons-roncesvalles",
        "cam-004": "mot17-09-mini",
    }


def test_lists_cameras_with_clips_frames_and_zones(client: TestClient) -> None:
    cameras = {c["camera_id"]: c for c in client.get("/api/cameras").json()["cameras"]}
    assert cameras["cam-001"]["name"] == "entrance"
    assert cameras["cam-001"]["frames"] == 3
    assert cameras["cam-002"]["frames"] == 0  # clip not downloaded in this test
    assert [z["name"] for z in cameras["cam-002"]["zones"]] == ["register_1", "register_2"]


def test_serves_frames_clamped_and_404_without_clip(client: TestClient) -> None:
    assert client.get("/api/clips/mot17-09-mini/frames/1").content == b"\xff\xd8\x01"
    assert client.get("/api/clips/mot17-09-mini/frames/99").content == b"\xff\xd8\x02"
    assert client.get("/api/clips/mot17-04-mini/frames/0").status_code == 404
    assert client.get("/api/clips/..%2Fsecrets/frames/0").status_code in (404, 422)


def test_saves_valid_zones(client: TestClient, catalog_path: Path) -> None:
    zones = [
        {"name": "dock_bay", "polygon": [[0.1, 0.2], [0.9, 0.2], [0.9, 0.95], [0.1, 0.95]]},
        {"name": "dock_door", "polygon": [[0.4, 0.1], [0.6, 0.1], [0.5, 0.3]]},
    ]
    response = client.put("/api/cameras/cam-004/zones", json={"zones": zones})
    assert response.status_code == 200
    assert response.json()["saved"] is True
    dock = next(c for c in load_catalog(catalog_path).cameras if c.camera_id == "cam-004")
    assert [z.name for z in dock.zones] == ["dock_bay", "dock_door"]
    assert catalog_path.read_text().startswith("# Camera catalog")


def test_refuses_invalid_zones_and_leaves_the_file(client: TestClient, catalog_path: Path) -> None:
    before = catalog_path.read_text()
    bow_tie = [{"name": "x", "polygon": [[0, 0], [1, 1], [1, 0], [0, 1]]}]
    response = client.put("/api/cameras/cam-004/zones", json={"zones": bow_tie})
    assert response.status_code == 422
    assert any("intersect" in e for e in response.json()["errors"])
    dupe = [{"name": "front_door", "polygon": [[0, 0], [1, 0], [0, 1]]}]
    response = client.put("/api/cameras/cam-004/zones", json={"zones": dupe})
    assert "duplicate zone name" in response.json()["errors"][0]
    assert client.put("/api/cameras/cam-999/zones", json={"zones": []}).status_code == 422
    assert catalog_path.read_text() == before


def test_clearing_zones_is_allowed(client: TestClient, catalog_path: Path) -> None:
    assert client.put("/api/cameras/cam-001/zones", json={"zones": []}).status_code == 200
    entrance = next(c for c in load_catalog(catalog_path).cameras if c.camera_id == "cam-001")
    assert entrance.zones == ()


def test_serves_the_page(client: TestClient) -> None:
    assert "zone editor" in client.get("/").text
