"""The zone editor: a local web page to draw and edit zone polygons on camera frames.

    make zones          # http://127.0.0.1:8075

It shows frames from the clip each catalog camera plays in the demo (loadgen's round-robin
assignment of `configs/loadgen.yaml` clips to `cam-001`, `cam-002`, ...), lets you draw, move,
add and delete polygon points, and saves to `configs/cameras.yaml` through the catalog's own
validation (simple polygons inside the frame, unique zone names), keeping the file's header.
It writes only the YAML: `make migrate` loads the zones into Postgres, where workers pick them
up within `zones.refresh_s`. Check the result with `python -m vms.worker render`.

A development tool for one person on localhost: no auth, binds to 127.0.0.1 by default.

    GET  /api/cameras                        cameras, their clip, frame count and zones
    GET  /api/clips/{clip_id}/frames/{i}     frame i of a clip (JPEG)
    PUT  /api/cameras/{camera_id}/zones      {"zones": [{"name", "polygon"}]} -> saved catalog
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi import Path as PathParam
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, ValidationError

from vms.common.logs import configure_logging, get_logger
from vms.common.settings import load_settings
from vms.loadgen.schedule import camera_id as loadgen_camera_id
from vms.loadgen.settings import LoadgenSettings
from vms.storage.catalog import (
    DEFAULT_CATALOG,
    CameraCatalog,
    load_catalog,
    with_zones,
    write_catalog,
)

log = get_logger(__name__)
CLIP_ID = r"^[a-z0-9][a-z0-9_-]{0,63}$"
UI_PAGE = "zones.html"


class ZoneIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    polygon: list[tuple[float, float]]


class ZonesIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    zones: list[ZoneIn]


def demo_clips(catalog: CameraCatalog, clip_ids: Sequence[str], prefix: str) -> dict[str, str]:
    """camera_id -> clip id, as loadgen assigns clips when it runs one camera per catalog entry."""
    n = len(catalog.cameras)
    by_id = {loadgen_camera_id(prefix, i, n): clip_ids[i % len(clip_ids)] for i in range(n)}
    return {c.camera_id: by_id[c.camera_id] for c in catalog.cameras if c.camera_id in by_id}


def clip_frames(data_dir: Path, clip_id: str) -> list[Path]:
    return sorted((data_dir / "clips" / clip_id / "frames").glob("*.jpg"))


def _messages(exc: Exception) -> list[str]:
    if isinstance(exc, ValidationError):
        return [
            f"{'.'.join(str(p) for p in e['loc'])}: {str(e['msg']).removeprefix('Value error, ')}"
            for e in exc.errors()
        ]
    return [str(exc)]


def create_app(
    catalog_path: Path, data_dir: Path, clips: dict[str, str], ui_dir: Path | None
) -> FastAPI:
    app = FastAPI(title="vms zone editor", docs_url=None, redoc_url=None)

    @app.get("/api/cameras")
    async def cameras() -> dict[str, Any]:
        catalog = load_catalog(catalog_path)  # re-read: the file may have been edited by hand
        out = []
        for cam in catalog.cameras:
            clip = clips.get(cam.camera_id)
            out.append(
                {
                    "camera_id": cam.camera_id,
                    "name": cam.name,
                    "clip_id": clip,
                    "frames": len(clip_frames(data_dir, clip)) if clip else 0,
                    "zones": [{"name": z.name, "polygon": z.polygon} for z in cam.zones],
                }
            )
        return {"catalog": str(catalog_path), "cameras": out}

    @app.get("/api/clips/{clip_id}/frames/{index}")
    async def frame(
        clip_id: str = PathParam(pattern=CLIP_ID), index: int = PathParam(ge=0)
    ) -> Response:
        frames = clip_frames(data_dir, clip_id)
        if not frames:
            return JSONResponse(
                {"error": f"no frames for clip {clip_id!r} in {data_dir}/clips (make data-mini)"},
                404,
            )
        return FileResponse(frames[min(index, len(frames) - 1)], media_type="image/jpeg")

    @app.put("/api/cameras/{camera_id}/zones")
    async def save(
        body: ZonesIn, camera_id: str = PathParam(pattern=r"^[A-Za-z0-9._-]{1,64}$")
    ) -> Response:
        try:
            updated = with_zones(
                load_catalog(catalog_path), camera_id, [z.model_dump() for z in body.zones]
            )
            write_catalog(catalog_path, updated)
        except (ValueError, ValidationError) as exc:
            return JSONResponse({"saved": False, "errors": _messages(exc)}, 422)
        log.info("zones saved", extra={"camera_id": camera_id, "zones": len(body.zones)})
        return JSONResponse({"saved": True, "catalog": str(catalog_path)})

    if ui_dir is not None and (ui_dir / UI_PAGE).is_file():

        @app.get("/")
        async def page() -> FileResponse:
            return FileResponse(ui_dir / UI_PAGE)

        app.mount("/", StaticFiles(directory=ui_dir), name="ui")
    return app


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vms.zones", description=__doc__)
    sub = parser.add_subparsers(dest="command")
    edit = sub.add_parser("edit", help="run the zone editor (default)")
    for p in (parser, edit):
        p.add_argument("--host", default="127.0.0.1")
        p.add_argument("--port", type=int, default=8075)
        p.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
        p.add_argument("--data-dir", type=Path, default=Path("data"))
        p.add_argument("--ui-dir", type=Path, default=Path("frontend/dist"))
        p.add_argument(
            "--clip",
            action="append",
            default=[],
            metavar="CAMERA_ID=CLIP_ID",
            help="override the clip shown for a camera",
        )
    args = parser.parse_args(argv)
    configure_logging("zones", "INFO")

    loadgen = load_settings(LoadgenSettings, "loadgen", telemetry={"enabled": False})
    catalog = load_catalog(args.catalog)
    clips = demo_clips(catalog, loadgen.clips.ids, loadgen.camera_prefix)
    for override in args.clip:
        cam, _, clip = override.partition("=")
        if not re.match(CLIP_ID, clip):
            raise SystemExit(f"--clip {override!r}: expected CAMERA_ID=CLIP_ID")
        clips[cam] = clip
    if not (args.ui_dir / UI_PAGE).is_file():
        raise SystemExit(f"{args.ui_dir / UI_PAGE} not found: build the UI first (make ui-build)")
    log.info(
        "zone editor listening",
        extra={
            "url": f"http://{args.host}:{args.port}",
            "catalog": str(args.catalog),
            "clips": clips,
        },
    )
    uvicorn.run(
        create_app(args.catalog, args.data_dir, clips, args.ui_dir),
        host=args.host,
        port=args.port,
        log_config=None,
        access_log=False,
    )
    return 0
