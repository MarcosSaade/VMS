"""`python -m vms.worker`: the inference worker and its tools.

run                         (default; make worker) lease cameras, claim frames, detect
weights [NAME ...]          (make models) fetch and verify pinned weights
bench --scenario FILE       model-only latency (bench/scenarios/worker-model-*.yaml)
render CLIP:CAMERA ... --out FILE.mp4
                            review video: detections, tracks, zones and events on clips
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import yaml

from vms.common.logs import configure_logging, get_logger

log = get_logger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m vms.worker", description=__doc__)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("run", help="run the worker (default)")
    weights = commands.add_parser("weights", help="fetch and verify pinned weights")
    weights.add_argument("names", nargs="*", help="default: every configured model")
    bench = commands.add_parser("bench", help="model-only latency benchmark")
    bench.add_argument("--scenario", type=Path, required=True)
    render = commands.add_parser("render", help="draw detections, tracks and zones on clips")
    render.add_argument(
        "segments",
        nargs="+",
        metavar="CLIP:CAMERA",
        help="clip id from data/clips and the catalog camera whose zones to use",
    )
    render.add_argument("--out", type=Path, required=True)
    render.add_argument("--seconds", type=float, help="max seconds per clip")
    render.add_argument("--weights", help="override detector.weights")
    render.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args(argv)

    if args.command in (None, "run"):
        from vms.worker.main import serve

        return serve()

    configure_logging("worker", "INFO")
    if args.command == "weights":
        from vms.worker.weights import ensure_weights, load_models_config

        models = load_models_config()
        for name in args.names or sorted(models.weights):
            path = ensure_weights(models, name)
            log.info("weights verified", extra={"name": name, "path": str(path)})
        return 0

    if args.command == "render":
        return _render(args)

    from vms.worker import bench as bench_module

    raw = yaml.safe_load(args.scenario.read_text(encoding="utf-8"))
    bench_module.run(bench_module.Scenario.model_validate(raw), raw)
    return 0


def _render(args: argparse.Namespace) -> int:
    from vms.common.settings import load_settings
    from vms.loadgen.frames import load_clip
    from vms.loadgen.settings import FrameSettings
    from vms.storage.catalog import load_catalog
    from vms.worker.detector import YoloDetector
    from vms.worker.render import Segment, render, zones_for
    from vms.worker.settings import WorkerSettings
    from vms.worker.tracking import CameraTracker
    from vms.worker.weights import ensure_weights, load_models_config

    settings = load_settings(WorkerSettings, "worker")
    if args.weights:  # only the weights; the rest of the detector section stays as configured
        detector_settings = settings.detector.model_copy(update={"weights": args.weights})
        settings = settings.model_copy(update={"detector": detector_settings})
    models = load_models_config()
    name = settings.detector.weights
    detector = YoloDetector(
        ensure_weights(models, name), models.weights[name].sha256, settings.detector
    )
    catalog = load_catalog()
    segments = []
    for spec in args.segments:
        clip_id, _, camera_name = spec.partition(":")
        if not camera_name:
            raise SystemExit(f"{spec!r}: expected CLIP:CAMERA, e.g. mot17-09:entrance")
        camera_id, zones = zones_for(catalog, camera_name)
        clip = load_clip(args.data_dir, clip_id, FrameSettings())  # as loadgen sends them
        segments.append(
            Segment(clip_id, camera_id, camera_name, clip.fps, list(clip.frames), zones)
        )
    summaries = render(
        segments,
        detector,
        CameraTracker,
        settings.tracker,
        settings.zones,
        args.out,
        fps=settings.tracker.fps,
        max_seconds=args.seconds,
    )
    for s in summaries:
        log.info(
            "segment rendered",
            extra={
                "clip": s.clip_id,
                "camera": s.camera_name,
                "frames": s.frames,
                "tracks": len(s.tracks),
                "stored_detections_per_frame": round(s.stored_detections / max(s.frames, 1), 1),
                "tracked_share": round(s.tracked_detections / max(s.stored_detections, 1), 2),
                "events": dict(s.events),
            },
        )
    log.info("video written", extra={"path": str(args.out)})
    return 0
