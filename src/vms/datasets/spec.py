"""Typed spec of the public video sources (`configs/data/sources.yaml`).

The spec is the single, versioned description of which clips exist, where their bytes come
from, the checksum that pins those bytes, the license, and hand-checked scene labels. The
fetcher turns it into normalized JPEG clips under `data/clips/`; loadgen replays those clips as
virtual cameras. Same spec + same pinned dependencies => same clips for every benchmark.
"""

from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

CLIP_ID = r"^[a-z0-9][a-z0-9-]*$"
_CHECKSUM = re.compile(r"^(sha1|sha256|sha512|md5):[0-9a-f]+$")
_HEX_LEN = {"md5": 32, "sha1": 40, "sha256": 64, "sha512": 128}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _check_checksum(value: str) -> str:
    if not _CHECKSUM.match(value):
        raise ValueError("checksum must look like '<sha1|sha256|sha512|md5>:<hex>'")
    algo, digest = value.split(":", 1)
    if len(digest) != _HEX_LEN[algo]:
        raise ValueError(f"{algo} digest must have {_HEX_LEN[algo]} hex characters")
    return value


class License(_Model):
    name: str
    url: HttpUrl
    commercial_use: bool
    share_alike: bool
    # Set for licenses that must be accepted by the user before download (e.g. VIRAT's
    # agreement). Such clips are fetched only with `--accept-license <id>`.
    requires_acceptance: bool = False
    notes: str = ""


class Location(StrEnum):
    INDOOR = "indoor"
    OUTDOOR = "outdoor"


class Lighting(StrEnum):
    DAY = "day"
    NIGHT = "night"  # dark sky; scenes may still be lit artificially
    OVERCAST = "overcast"


class Viewpoint(StrEnum):
    ELEVATED = "elevated"  # CCTV-like, looking down
    EYE_LEVEL = "eye_level"


class Density(StrEnum):
    SPARSE = "sparse"
    MODERATE = "moderate"
    CROWDED = "crowded"


class Scene(_Model):
    """Hand-checked labels (from looking at frames), used to pick clips for drift and retrieval
    scenarios. `objects` lists the classes that clearly appear, not an exhaustive census."""

    description: str
    location: Location
    lighting: Lighting
    viewpoint: Viewpoint
    density: Density
    objects: list[str] = Field(min_length=1)


class ZipSequenceSource(_Model):
    """A directory of JPEG frames inside a remote zip, read with HTTP range requests so only
    the needed members are downloaded (the MOT archives are 5 to 6 GB)."""

    kind: Literal["zip_sequence"] = "zip_sequence"
    url: HttpUrl
    archive_bytes: int = Field(gt=0)  # pins the archive version together with the checksum
    frames_prefix: str  # e.g. "MOT17/train/MOT17-02-FRCNN/img1/"
    fps: float = Field(gt=0)
    first_frame: int = Field(default=1, ge=1)
    last_frame: int | None = Field(default=None, ge=1)  # inclusive
    # sha256 over the selected members' names and bytes, in name order (see members_digest).
    checksum: Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def _check(self) -> ZipSequenceSource:
        if not self.frames_prefix.endswith("/"):
            raise ValueError("frames_prefix must end with '/'")
        if self.last_frame is not None and self.last_frame < self.first_frame:
            raise ValueError("last_frame must be >= first_frame")
        return self


class VideoFileSource(_Model):
    """A whole video file, downloaded once into the raw cache and verified before decoding."""

    kind: Literal["video_file"] = "video_file"
    url: HttpUrl
    file_bytes: int = Field(gt=0)
    checksum: str
    # Frame rate to use instead of the container's. Some files report a wrong rate (OpenCV
    # reads one VP8 WebM as 30000 fps instead of 29.97), so pin it when probing is unreliable.
    fps: float | None = Field(default=None, gt=0, le=240)
    start_s: float = Field(default=0.0, ge=0)
    duration_s: float | None = Field(default=None, gt=0)  # None: to the end

    @field_validator("checksum")
    @classmethod
    def _valid_checksum(cls, value: str) -> str:
        return _check_checksum(value)


Source = Annotated[ZipSequenceSource | VideoFileSource, Field(discriminator="kind")]


class Clip(_Model):
    id: Annotated[str, Field(pattern=CLIP_ID)]
    title: str
    license: str  # key into DataSpec.licenses
    attribution: str
    page_url: HttpUrl  # human-readable source page
    camera_motion: Literal["static"] = "static"  # moving-camera clips are not used
    scene: Scene
    source: Source


class Processing(_Model):
    """How source frames become clip frames. Changing any field changes every clip's hash."""

    max_height: int = Field(default=720, ge=16)  # downscale only, keeps aspect ratio
    max_fps: float = Field(default=30.0, gt=0)  # higher-rate sources drop frames evenly
    jpeg_quality: int = Field(default=90, ge=1, le=100)


class DataSpec(_Model):
    version: int = Field(ge=1)
    processing: Processing = Field(default_factory=Processing)
    licenses: dict[str, License]
    clips: list[Clip]
    subsets: dict[str, list[str]]

    @model_validator(mode="after")
    def _check_refs(self) -> DataSpec:
        ids = [c.id for c in self.clips]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate clip ids")
        for clip in self.clips:
            if clip.license not in self.licenses:
                raise ValueError(f"clip {clip.id}: unknown license {clip.license!r}")
        known = set(ids)
        for name, members in self.subsets.items():
            missing = [m for m in members if m not in known]
            if missing:
                raise ValueError(f"subset {name}: unknown clips {missing}")
        return self

    def clip(self, clip_id: str) -> Clip:
        for clip in self.clips:
            if clip.id == clip_id:
                return clip
        raise KeyError(clip_id)

    def subset(self, name: str) -> list[Clip]:
        if name not in self.subsets:
            raise KeyError(f"unknown subset {name!r}; known: {sorted(self.subsets)}")
        return [self.clip(i) for i in self.subsets[name]]


def load_spec(path: Path) -> DataSpec:
    with path.open(encoding="utf-8") as fh:
        return DataSpec.model_validate(yaml.safe_load(fh))


def clip_spec_hash(clip: Clip, processing: Processing) -> str:
    """Identity of a clip's output: changes whenever its source bytes, frame selection or the
    processing change, and not when labels or descriptions are edited."""
    payload = {"source": clip.source.model_dump(mode="json"), "processing": processing.model_dump()}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(blob).hexdigest()
