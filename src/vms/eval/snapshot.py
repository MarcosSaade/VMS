"""Synthetic eval snapshot: a seeded generator of pipeline output for the structured-query eval.

Gold answers must never change, so they are computed on a frozen database rather than on
whatever the demo last ingested. This module samples that database from a versioned spec
(`configs/eval/snapshot.v<N>.yaml`) and a seed: frames every `frame_interval_s` per camera,
tracks arriving as a Poisson process shaped by local hour and weekday, one detection per frame
a track is on, zone enter/exit/loiter events, and untracked low-confidence noise (D29).

Why synthetic instead of replaying the demo clips: the clips are a few minutes long, so a
replay holds one short window on one day and cannot exercise "last week", hourly trends, DST
or a camera outage; and CPU inference is not bit-reproducible across machines, so a replay
would not regenerate identically. The data mimics the pipeline's semantics (a track's
detections are on consecutive frames; events lie within their track), not its accuracy.

Determinism: one `random.Random` stream per camera seeded from (seed, camera name), integer
frame grid, values rounded before writing, gzip with mtime 0. The same spec and seed give the
same bytes, which `eval/structured/snapshot.v<N>.json` pins by SHA-256.
"""

from __future__ import annotations

import csv
import datetime as dt
import gzip
import hashlib
import io
import json
import math
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo

import psycopg
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from vms.storage.catalog import Camera, CameraCatalog, load_catalog

UTC = dt.UTC

# Column order of each table's CSV file, the order COPY reads them in. Load order matters
# only for readability: pipeline tables have no foreign keys (D19).
COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "frames": (
        "frame_id",
        "camera_id",
        "run_id",
        "seq",
        "ts",
        "received_at",
        "object_key",
        "size_bytes",
        "status",
        "width",
        "height",
        "brightness",
        "contrast",
        "blur_var",
        "model_version",
        "processed_at",
    ),
    "tracks": ("track_id", "camera_id", "class_name", "first_seen", "last_seen", "n_frames"),
    "detections": (
        "frame_id",
        "det_idx",
        "ts",
        "camera_id",
        "track_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "model_version",
    ),
    "events": ("event_id", "ts", "camera_id", "track_id", "zone_id", "event_type"),
}
TABLES: Final = tuple(COLUMNS)

Row = tuple[Any, ...]

# Chance that a visit outlasting its track ends with an exit on the track's last frame.
P_EXIT_AT_TRACK_END: Final = 0.85


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Lognormal(_Model):
    """A positive duration: `median * exp(sigma * N(0, 1))`. Written `[median, sigma]`."""

    median: float = Field(gt=0)
    sigma: float = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def _from_pair(cls, value: Any) -> Any:
        if isinstance(value, list | tuple) and len(value) == 2:
            return {"median": value[0], "sigma": value[1]}
        return value

    def sample(self, rng: random.Random) -> float:
        return self.median * math.exp(self.sigma * rng.gauss(0.0, 1.0))


class ZoneProfile(_Model):
    classes: tuple[str, ...] = Field(min_length=1)
    p_visit: float = Field(ge=0, le=1)  # chance a track of these classes enters the zone
    dwell_s: Lognormal
    p_revisit: float = Field(default=0.0, ge=0, le=1)  # chance of a second visit


class CameraProfile(_Model):
    name: str
    shape: str
    weekend: float = Field(ge=0)  # multiplier on Saturdays and Sundays (local)
    size: tuple[int, int]
    indoor: bool
    rates: dict[str, float]  # tracks per hour at shape 1.0, per class
    duration_s: dict[str, Lognormal]  # per class, with a "default"
    zones: dict[str, ZoneProfile] = Field(default_factory=dict)
    outages: tuple[tuple[dt.datetime, dt.datetime], ...] = ()  # camera-local, no frames
    false_positives_per_hour: float = Field(ge=0)

    @field_validator("duration_s")
    @classmethod
    def _has_default(cls, value: dict[str, Lognormal]) -> dict[str, Lognormal]:
        if "default" not in value:
            raise ValueError("duration_s needs a 'default' entry")
        return value

    @model_validator(mode="after")
    def _check(self) -> CameraProfile:
        if sum(z.p_visit for z in self.zones.values()) > 1.0:
            raise ValueError(f"{self.name}: zone p_visit values must sum to at most 1")
        for start, end in self.outages:
            if start.tzinfo is not None or end.tzinfo is not None or start >= end:
                raise ValueError(f"{self.name}: outages are local [start, end) without offset")
        return self

    def duration(self, class_name: str) -> Lognormal:
        return self.duration_s.get(class_name, self.duration_s["default"])


class SnapshotSpec(_Model):
    version: int = Field(ge=1)
    seed: int
    catalog: Path
    start: dt.datetime
    now: dt.datetime
    frame_interval_s: int = Field(ge=1)
    loiter_s: int = Field(ge=1)
    model_version: str
    run_id: str
    shapes: dict[str, tuple[float, ...]]
    confidence: dict[str, tuple[float, float]]
    false_positive_classes: tuple[str, ...]
    false_positive_confidence: tuple[float, float]
    cameras: tuple[CameraProfile, ...]
    max_duration_s: int = Field(ge=1)

    @model_validator(mode="after")
    def _check(self) -> SnapshotSpec:
        for name in ("start", "now"):
            value: dt.datetime = getattr(self, name)
            if value.utcoffset() != dt.timedelta(0):
                raise ValueError(f"{name} must be UTC")
        if self.start >= self.now:
            raise ValueError("start must be before now")
        for name, shape in self.shapes.items():
            if len(shape) != 24:
                raise ValueError(f"shape {name!r} needs 24 hourly values")
        for cam in self.cameras:
            if cam.shape not in self.shapes:
                raise ValueError(f"{cam.name}: unknown shape {cam.shape!r}")
        if "default" not in self.confidence:
            raise ValueError("confidence needs a 'default' entry")
        return self


def load_spec(path: Path) -> SnapshotSpec:
    with path.open(encoding="utf-8") as fh:
        return SnapshotSpec.model_validate(yaml.safe_load(fh))


def spec_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass
class Snapshot:
    """Rows per table, in COLUMNS order."""

    rows: dict[str, list[Row]] = field(default_factory=lambda: {t: [] for t in TABLES})

    def counts(self) -> dict[str, int]:
        return {t: len(r) for t, r in self.rows.items()}


# --- generation ------------------------------------------------------------------------------


def poisson(rng: random.Random, lam: float) -> int:
    """Knuth's method; fine for the rates used here (lam < 100)."""
    if lam <= 0:
        return 0
    threshold, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= threshold:
            return k
        k += 1


def _iso(ts: dt.datetime) -> str:
    return ts.astimezone(UTC).isoformat()


class _CameraGen:
    """Generates one camera's rows from its own random stream."""

    def __init__(self, spec: SnapshotSpec, profile: CameraProfile, camera: Camera, offset_s: int):
        self.spec = spec
        self.p = profile
        self.cam = camera
        self.tz = ZoneInfo(camera.timezone)
        self.rng = random.Random(f"{spec.seed}:{profile.name}")
        self.step = dt.timedelta(seconds=spec.frame_interval_s)
        self.t0 = spec.start + dt.timedelta(seconds=offset_s)
        n = math.ceil((spec.now - self.t0) / self.step)
        outages = [
            (a.replace(tzinfo=self.tz).astimezone(UTC), b.replace(tzinfo=self.tz).astimezone(UTC))
            for a, b in profile.outages
        ]
        # present[k]: frame k exists (before now and outside every outage).
        self.present = [
            not any(a <= self.ts(k) < b for a, b in outages) and self.ts(k) < spec.now
            for k in range(n)
        ]
        self.det_count = [0] * n
        self.zone_ids = {z.name: camera.zone_id(z) for z in camera.zones}
        self.track_seq = 0
        self.event_seq = 0

    def ts(self, k: int) -> dt.datetime:
        return self.t0 + k * self.step

    def frame_id(self, k: int) -> str:
        return f"{self.cam.camera_id}-f{k:07d}"

    def run(self, out: Snapshot) -> None:
        tracks: list[tuple[str, int, int]] = []  # class, first k, last k
        hour = self.spec.start
        while hour < self.spec.now:
            local = hour.astimezone(self.tz)
            factor = self.spec.shapes[self.p.shape][local.hour]
            if local.weekday() >= 5:
                factor *= self.p.weekend
            for cls in sorted(self.p.rates):
                for _ in range(poisson(self.rng, self.p.rates[cls] * factor)):
                    at = hour + dt.timedelta(seconds=self.rng.uniform(0, 3600))
                    span = self.place(at, self.p.duration(cls).sample(self.rng))
                    if span is not None:
                        tracks.append((cls, *span))
            hour += dt.timedelta(hours=1)
        for cls, first, last in tracks:
            self.track(out, cls, first, last)
        self.noise(out)
        self.frames(out)

    def place(self, at: dt.datetime, duration_s: float) -> tuple[int, int] | None:
        """Frames a track covers: from the first frame at or after `at`, as many as its
        duration spans, stopping at an outage or at now. None if it starts in no frame."""
        first = max(0, math.ceil((at - self.t0) / self.step))
        if first >= len(self.present) or not self.present[first]:
            return None
        duration_s = min(duration_s, self.spec.max_duration_s)
        last = first + int(duration_s // self.spec.frame_interval_s)
        k = first
        while k < last and k + 1 < len(self.present) and self.present[k + 1]:
            k += 1
        return first, k

    def track(self, out: Snapshot, cls: str, first: int, last: int) -> None:
        self.track_seq += 1
        track_id = f"{self.cam.camera_id}:t{self.track_seq:06d}"
        out.rows["tracks"].append(
            (track_id, self.cam.camera_id, cls, _iso(self.ts(first)), _iso(self.ts(last)),
             last - first + 1)
        )  # fmt: skip
        w, h = self.p.size
        bw = w * self.rng.uniform(0.04, 0.12)
        bh = h * self.rng.uniform(0.15, 0.45)
        cx, cy = self.rng.uniform(bw / 2, w - bw / 2), self.rng.uniform(bh / 2, h - bh / 2)
        mean, sd = self.spec.confidence.get(cls, self.spec.confidence["default"])
        for k in range(first, last + 1):
            cx = min(max(cx + self.rng.gauss(0, w * 0.01), bw / 2), w - bw / 2)
            cy = min(max(cy + self.rng.gauss(0, h * 0.01), bh / 2), h - bh / 2)
            conf = min(max(self.rng.gauss(mean, sd), 0.25), 0.99)
            self.detection(out, k, track_id, cls, conf, (cx - bw / 2, cy - bh / 2, bw, bh))
        self.zone_events(out, track_id, cls, first, last)

    def detection(
        self,
        out: Snapshot,
        k: int,
        track_id: str | None,
        cls: str,
        conf: float,
        box: tuple[float, float, float, float],
    ) -> None:
        x, y, bw, bh = box
        idx = self.det_count[k]
        self.det_count[k] += 1
        out.rows["detections"].append(
            (self.frame_id(k), idx, _iso(self.ts(k)), self.cam.camera_id, track_id, cls,
             round(conf, 4), round(x, 1), round(y, 1), round(x + bw, 1), round(y + bh, 1),
             self.spec.model_version)
        )  # fmt: skip

    def event(self, out: Snapshot, k: int, track_id: str, zone: str, kind: str) -> None:
        self.event_seq += 1
        out.rows["events"].append(
            (f"{self.cam.camera_id}:e{self.event_seq:07d}", _iso(self.ts(k)), self.cam.camera_id,
             track_id, self.zone_ids[zone], kind)
        )  # fmt: skip

    def zone_events(self, out: Snapshot, track_id: str, cls: str, first: int, last: int) -> None:
        u, acc, chosen = self.rng.random(), 0.0, None
        for name, zone in self.p.zones.items():
            if cls not in zone.classes:
                continue
            acc += zone.p_visit
            if u < acc:
                chosen = name
                break
        if chosen is None:
            return
        zone = self.p.zones[chosen]
        enter = first + self.rng.randint(0, (last - first) // 3)
        visits = 2 if self.rng.random() < zone.p_revisit else 1
        for _ in range(visits):
            if enter > last:
                return
            frames = max(1, round(zone.dwell_s.sample(self.rng) / self.spec.frame_interval_s))
            exit_k = enter + frames
            self.event(out, enter, track_id, chosen, "zone_enter")
            loiter_k = enter + math.ceil(self.spec.loiter_s / self.spec.frame_interval_s)
            if loiter_k <= min(exit_k - 1, last):
                self.event(out, loiter_k, track_id, chosen, "loiter")
            if exit_k > last:
                # Most visits end as the object walks out of view: the exit is on the track's
                # last frame. The rest are still inside when the track is lost: no exit event.
                if last <= enter or self.rng.random() >= P_EXIT_AT_TRACK_END:
                    return
                exit_k = last
            self.event(out, exit_k, track_id, chosen, "zone_exit")
            enter = exit_k + 1 + self.rng.randint(0, 2)

    def noise(self, out: Snapshot) -> None:
        p = self.p.false_positives_per_hour * self.spec.frame_interval_s / 3600
        lo, hi = self.spec.false_positive_confidence
        w, h = self.p.size
        for k, present in enumerate(self.present):
            if present and self.rng.random() < p:
                cls = self.rng.choice(self.spec.false_positive_classes)
                bw, bh = w * self.rng.uniform(0.02, 0.08), h * self.rng.uniform(0.04, 0.2)
                x, y = self.rng.uniform(0, w - bw), self.rng.uniform(0, h - bh)
                self.detection(out, k, None, cls, self.rng.uniform(lo, hi), (x, y, bw, bh))

    def frames(self, out: Snapshot) -> None:
        w, h = self.p.size
        for k, present in enumerate(self.present):
            if not present:
                continue
            ts = self.ts(k)
            hour = ts.astimezone(self.tz).hour
            daylight = self.p.indoor or 7 <= hour < 19
            brightness = self.rng.gauss(115, 8) if daylight else self.rng.gauss(38, 6)
            out.rows["frames"].append(
                (self.frame_id(k), self.cam.camera_id, self.spec.run_id, k, _iso(ts),
                 _iso(ts + dt.timedelta(milliseconds=40)),
                 f"frames/{self.cam.camera_id}/{self.spec.run_id}/{k:07d}.jpg",
                 self.rng.randint(40_000, 140_000), "done", w, h, round(brightness, 2),
                 round(self.rng.gauss(45, 5), 2), round(self.rng.lognormvariate(5.7, 0.3), 2),
                 self.spec.model_version, _iso(ts + dt.timedelta(milliseconds=300)))
            )  # fmt: skip


def generate(spec: SnapshotSpec, catalog: CameraCatalog) -> Snapshot:
    """All rows of the snapshot. Cameras get their own random streams and frame offsets, so
    adding a camera to a later spec does not change the others."""
    by_name = {c.name: c for c in catalog.cameras}
    out = Snapshot()
    for i, profile in enumerate(spec.cameras):
        camera = by_name.get(profile.name)
        if camera is None:
            raise ValueError(f"camera {profile.name!r} is not in the catalog")
        unknown = set(profile.zones) - {z.name for z in camera.zones}
        if unknown:
            raise ValueError(f"{profile.name}: zones not in the catalog: {sorted(unknown)}")
        _CameraGen(spec, profile, camera, offset_s=3 * i).run(out)
    return out


# --- files -----------------------------------------------------------------------------------


def _csv_value(value: Any) -> Any:
    return "" if value is None else value


def write_snapshot(snapshot: Snapshot, directory: Path) -> dict[str, str]:
    """Write `<table>.csv.gz` per table; returns {file name: sha256}. Bytes are reproducible
    (no timestamps in the gzip header, fixed row order). NULL is the empty unquoted field."""
    directory.mkdir(parents=True, exist_ok=True)
    digests = {}
    for table in TABLES:
        text = io.StringIO()
        writer = csv.writer(text, lineterminator="\n")
        writer.writerow(COLUMNS[table])
        for row in snapshot.rows[table]:
            writer.writerow([_csv_value(v) for v in row])
        data = gzip.compress(text.getvalue().encode("utf-8"), compresslevel=6, mtime=0)
        name = f"{table}.csv.gz"
        (directory / name).write_bytes(data)
        digests[name] = hashlib.sha256(data).hexdigest()
    return digests


def file_digests(directory: Path) -> dict[str, str]:
    return {
        f"{t}.csv.gz": hashlib.sha256((directory / f"{t}.csv.gz").read_bytes()).hexdigest()
        for t in TABLES
    }


class SnapshotManifest(_Model):
    """What pins a snapshot: committed to git next to the DVC-tracked directory."""

    version: int
    seed: int
    now: dt.datetime
    spec: str  # repo-relative path
    spec_sha256: str
    catalog: str
    catalog_sha256: str
    files: dict[str, str]  # name -> sha256
    rows: dict[str, int]
    synthetic: bool = True


def read_manifest(path: Path) -> SnapshotManifest:
    return SnapshotManifest.model_validate_json(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, manifest: SnapshotManifest) -> None:
    data = manifest.model_dump(mode="json")
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def verify_files(directory: Path, manifest: SnapshotManifest) -> list[str]:
    """Problems with the snapshot files against the manifest, empty if they match."""
    problems = []
    for name, digest in manifest.files.items():
        path = directory / name
        if not path.is_file():
            problems.append(f"{path} is missing")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            problems.append(f"{path} does not match the manifest's sha256")
    return problems


# --- loading ---------------------------------------------------------------------------------


def _copy_lines(path: Path) -> Iterator[bytes]:
    with gzip.open(path, "rb") as fh:
        yield from fh


def load_tables(conn: psycopg.Connection[Any], directory: Path) -> dict[str, int]:
    """COPY the snapshot files into a migrated, empty database (catalog synced separately).
    Runs in one transaction and finishes with ANALYZE, so plans see real statistics."""
    counts = {}
    with conn.transaction():
        for table in TABLES:
            cols = ", ".join(COLUMNS[table])
            copy_sql = f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv, HEADER true)"
            with conn.cursor() as cur, cur.copy(copy_sql) as copy:
                for line in _copy_lines(directory / f"{table}.csv.gz"):
                    copy.write(line)
            row = conn.execute(f"SELECT count(*) FROM {table}").fetchone()
            counts[table] = int(row[0]) if row else 0
    for table in TABLES:
        conn.execute(f"ANALYZE {table}")
    return counts


def build(spec_path: Path, out_dir: Path, repo_root: Path) -> SnapshotManifest:
    """Generate and write the snapshot for `spec_path`; returns its manifest."""
    spec = load_spec(spec_path)
    catalog_path = repo_root / spec.catalog
    snapshot = generate(spec, load_catalog(catalog_path))
    digests = write_snapshot(snapshot, out_dir)
    return SnapshotManifest(
        version=spec.version,
        seed=spec.seed,
        now=spec.now,
        spec=_relative(spec_path, repo_root),
        spec_sha256=spec_digest(spec_path),
        catalog=str(spec.catalog),
        catalog_sha256=spec_digest(catalog_path),
        files=digests,
        rows=snapshot.counts(),
    )


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)
