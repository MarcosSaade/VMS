"""The synthetic eval snapshot: reproducible bytes, and rows that keep the pipeline's semantics
(a track's detections are on its frames, events lie inside their track, nothing after now,
nothing during an outage)."""

from __future__ import annotations

import datetime as dt
import gzip
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from vms.eval.snapshot import (
    COLUMNS,
    Snapshot,
    SnapshotSpec,
    file_digests,
    generate,
    load_spec,
    poisson,
    verify_files,
    write_snapshot,
)
from vms.storage.catalog import CameraCatalog, load_catalog

REPO_ROOT = Path(__file__).resolve().parents[3]
CATALOG = REPO_ROOT / "configs/eval/cameras.v1.yaml"
UTC = dt.UTC


def mini_spec(**overrides: Any) -> SnapshotSpec:
    """Two cameras over 26 hours with a 2-hour outage; busy enough for every row kind."""
    data: dict[str, Any] = {
        "version": 1,
        "seed": 7,
        "catalog": str(CATALOG),
        "start": "2026-03-28T22:00:00Z",
        "now": "2026-03-30T00:00:00Z",  # spans London's DST change
        "frame_interval_s": 20,
        "loiter_s": 60,
        "model_version": "test",
        "run_id": "t",
        "shapes": {"flat": [1.0] * 24},
        "confidence": {"default": [0.6, 0.1], "person": [0.8, 0.05]},
        "false_positive_classes": ["bird"],
        "false_positive_confidence": [0.25, 0.4],
        "cameras": [
            {
                "name": "checkout",
                "shape": "flat",
                "weekend": 1.0,
                "size": [1280, 720],
                "indoor": True,
                "rates": {"person": 20, "handbag": 2},
                "duration_s": {"default": [120, 0.5]},
                "zones": {
                    "register_1": {"classes": ["person"], "p_visit": 0.5, "dwell_s": [60, 0.5]},
                    "register_2": {
                        "classes": ["person"],
                        "p_visit": 0.3,
                        "dwell_s": [80, 0.5],
                        "p_revisit": 0.3,
                    },
                },
                "false_positives_per_hour": 20,
            },
            {
                "name": "loading_dock",
                "shape": "flat",
                "weekend": 0.5,
                "size": [1280, 720],
                "indoor": False,
                "rates": {"truck": 3},
                "duration_s": {"default": [600, 0.5]},
                "outages": [["2026-03-29T10:00:00", "2026-03-29T12:00:00"]],
                "false_positives_per_hour": 0,
            },
        ],
        "max_duration_s": 3600,
    }
    data.update(overrides)
    return SnapshotSpec.model_validate(data)


@pytest.fixture(scope="module")
def catalog() -> CameraCatalog:
    return load_catalog(CATALOG)


@pytest.fixture(scope="module")
def snapshot(catalog: CameraCatalog) -> Snapshot:
    return generate(mini_spec(), catalog)


def rows(snapshot: Snapshot, table: str) -> list[dict[str, Any]]:
    return [dict(zip(COLUMNS[table], r, strict=True)) for r in snapshot.rows[table]]


def ts(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value)


def test_same_spec_and_seed_give_the_same_bytes(catalog: CameraCatalog, tmp_path: Path) -> None:
    a = write_snapshot(generate(mini_spec(), catalog), tmp_path / "a")
    b = write_snapshot(generate(mini_spec(), catalog), tmp_path / "b")
    assert a == b
    assert file_digests(tmp_path / "a") == a
    c = write_snapshot(generate(mini_spec(seed=8), catalog), tmp_path / "c")
    assert c != a


def test_every_table_has_rows(snapshot: Snapshot) -> None:
    counts = snapshot.counts()
    assert all(counts[t] > 0 for t in COLUMNS), counts
    kinds = Counter(r["event_type"] for r in rows(snapshot, "events"))
    assert set(kinds) == {"zone_enter", "zone_exit", "loiter"}


def test_track_detections_cover_its_frames(snapshot: Snapshot) -> None:
    by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for d in rows(snapshot, "detections"):
        if d["track_id"] is not None:
            by_track[d["track_id"]].append(d)
    for t in rows(snapshot, "tracks"):
        dets = by_track[t["track_id"]]
        assert len(dets) == t["n_frames"]
        assert min(d["ts"] for d in dets) == t["first_seen"]
        assert max(d["ts"] for d in dets) == t["last_seen"]
        assert {d["class_name"] for d in dets} == {t["class_name"]}
        assert all(d["camera_id"] == t["camera_id"] for d in dets)


def test_detections_are_on_existing_frames_with_unique_keys(snapshot: Snapshot) -> None:
    frames = {f["frame_id"]: f for f in rows(snapshot, "frames")}
    keys = set()
    for d in rows(snapshot, "detections"):
        frame = frames[d["frame_id"]]
        assert (d["ts"], d["camera_id"]) == (frame["ts"], frame["camera_id"])
        assert 0.25 <= d["confidence"] <= 0.99
        assert 0 <= d["x1"] <= d["x2"] <= frame["width"]
        assert 0 <= d["y1"] <= d["y2"] <= frame["height"]
        keys.add((d["frame_id"], d["det_idx"]))
    assert len(keys) == snapshot.counts()["detections"]


def test_untracked_detections_are_low_confidence_noise(snapshot: Snapshot) -> None:
    noise = [d for d in rows(snapshot, "detections") if d["track_id"] is None]
    assert noise
    assert all(d["class_name"] == "bird" and d["confidence"] <= 0.4 for d in noise)


def test_events_lie_inside_their_track_in_visit_order(snapshot: Snapshot) -> None:
    tracks = {t["track_id"]: t for t in rows(snapshot, "tracks")}
    by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in rows(snapshot, "events"):
        t = tracks[e["track_id"]]
        assert t["first_seen"] <= e["ts"] <= t["last_seen"]
        assert e["zone_id"].startswith(f"{t['camera_id']}:")
        by_track[e["track_id"]].append(e)
    for events in by_track.values():
        visits = [e["event_type"] for e in events if e["event_type"] != "loiter"]
        # enter, exit, enter, exit, ...; a visit may end without an exit (track lost inside).
        assert visits[0::2] == ["zone_enter"] * len(visits[0::2])
        assert visits[1::2] == ["zone_exit"] * len(visits[1::2])
        assert [ts(e["ts"]) for e in events] == sorted(ts(e["ts"]) for e in events)


def test_loiter_follows_a_long_enough_visit(snapshot: Snapshot) -> None:
    by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for e in rows(snapshot, "events"):
        by_track[e["track_id"]].append(e)
    for events in by_track.values():
        for i, e in enumerate(events):
            if e["event_type"] == "loiter":
                enter = events[i - 1]
                assert enter["event_type"] == "zone_enter"
                assert ts(e["ts"]) - ts(enter["ts"]) >= dt.timedelta(seconds=60)


def test_nothing_after_now_or_during_an_outage(snapshot: Snapshot) -> None:
    spec = mini_spec()
    # 10:00-12:00 London on 29 March, after the change to summer time: 09:00-11:00 UTC.
    outage = (dt.datetime(2026, 3, 29, 9, tzinfo=UTC), dt.datetime(2026, 3, 29, 11, tzinfo=UTC))
    for table, col in (("frames", "ts"), ("detections", "ts"), ("events", "ts")):
        for r in rows(snapshot, table):
            assert spec.start <= ts(r[col]) < spec.now
            if r["camera_id"] == "cam-004":
                assert not outage[0] <= ts(r[col]) < outage[1]
    dock = [ts(f["ts"]) for f in rows(snapshot, "frames") if f["camera_id"] == "cam-004"]
    # 26 h of frames every 20 s, less the 2 h outage.
    assert len(dock) == 24 * 180


def test_written_files_read_back_and_verify(snapshot: Snapshot, tmp_path: Path) -> None:
    digests = write_snapshot(snapshot, tmp_path)
    with gzip.open(tmp_path / "events.csv.gz", "rt") as fh:
        header = fh.readline().strip().split(",")
    assert tuple(header) == COLUMNS["events"]
    from vms.eval.snapshot import SnapshotManifest

    manifest = SnapshotManifest(
        version=1,
        seed=7,
        now=mini_spec().now,
        spec="x",
        spec_sha256="",
        catalog="y",
        catalog_sha256="",
        files=digests,
        rows=snapshot.counts(),
    )
    assert verify_files(tmp_path, manifest) == []
    (tmp_path / "tracks.csv.gz").write_bytes(b"tampered")
    assert verify_files(tmp_path, manifest) == [
        f"{tmp_path / 'tracks.csv.gz'} does not match the manifest's sha256"
    ]


def test_poisson_mean_is_close_to_lambda() -> None:
    rng = random.Random(1)
    draws = [poisson(rng, 12.0) for _ in range(4000)]
    assert abs(sum(draws) / len(draws) - 12.0) < 0.3
    assert poisson(rng, 0.0) == 0


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"shapes": {"flat": [1.0] * 23}}, "24 hourly values"),
        ({"now": "2026-03-28T00:00:00Z"}, "start must be before now"),
        ({"start": "2026-03-28T22:00:00+01:00"}, "must be UTC"),
        ({"confidence": {"person": [0.8, 0.1]}}, "'default'"),
    ],
)
def test_invalid_specs_are_rejected(override: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        mini_spec(**override)


def test_zone_probabilities_must_not_exceed_one() -> None:
    spec = mini_spec().model_dump(mode="json")
    zones = spec["cameras"][0]["zones"]
    zones["register_1"]["p_visit"] = 0.8
    with pytest.raises(ValidationError, match="sum to at most 1"):
        SnapshotSpec.model_validate(spec)


def test_profiles_must_match_the_catalog(catalog: CameraCatalog) -> None:
    spec = mini_spec().model_dump(mode="json")
    spec["cameras"][0]["name"] = "nowhere"
    with pytest.raises(ValueError, match="not in the catalog"):
        generate(SnapshotSpec.model_validate(spec), catalog)
    spec = mini_spec().model_dump(mode="json")
    spec["cameras"][1]["zones"] = {
        "gate": {"classes": ["truck"], "p_visit": 0.5, "dwell_s": [10, 0.1]}
    }
    with pytest.raises(ValueError, match="zones not in the catalog"):
        generate(SnapshotSpec.model_validate(spec), catalog)


def test_the_v1_spec_loads_and_matches_its_catalog() -> None:
    spec = load_spec(REPO_ROOT / "configs/eval/snapshot.v1.yaml")
    catalog = load_catalog(REPO_ROOT / spec.catalog)
    assert {c.name for c in spec.cameras} == {c.name for c in catalog.cameras}
    assert spec.now == dt.datetime(2026, 3, 31, 14, 30, tzinfo=UTC)
