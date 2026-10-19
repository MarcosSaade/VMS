"""`python -m vms.worker`: the inference worker and its tools.

run                         (default; make worker) lease cameras, claim frames, detect
weights [NAME ...]          (make models) fetch and verify pinned weights
export-onnx [NAME ...]      (make onnx) export pinned weights to ONNX (models/onnx/)
onnx-parity [--weights N]   (make onnx-parity) ONNX Runtime vs PyTorch parity, results JSON
calibration build|fetch     INT8 calibration list (once per version) / its images (make
                            calibration-data)
quantize [NAME ...]         (make quantize) static INT8 quantization of the ONNX exports
ladder --scenario FILE      precision ladder: latency and throughput per backend and batch size
                            (bench/scenarios/ladder-*.yaml; each step in its own process)
bench --scenario FILE       model-only latency (bench/scenarios/worker-model-*.yaml)
sweep-plot FILE ... --out PNG
                            batch-size sweep (5.3): operating point per ladder result and the
                            throughput vs latency plot (--label per file)
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
    export = commands.add_parser("export-onnx", help="export pinned weights to ONNX")
    export.add_argument("names", nargs="*", help="default: every configured model")
    parity = commands.add_parser("onnx-parity", help="ONNX Runtime vs PyTorch parity")
    parity.add_argument("--weights", default="yolo11n")
    parity.add_argument("--eval-images", type=int, help="first N subset images (smoke only)")
    calib = commands.add_parser("calibration", help="INT8 calibration set")
    calib.add_argument("action", choices=["build", "fetch"])
    quant = commands.add_parser("quantize", help="static INT8 quantization of ONNX exports")
    quant.add_argument("names", nargs="*", help="default: every configured model")
    ladder = commands.add_parser("ladder", help="precision ladder speed benchmark")
    ladder.add_argument("--scenario", type=Path, required=True)
    step = commands.add_parser("ladder-step", help=argparse.SUPPRESS)  # one step, for `ladder`
    step.add_argument("--scenario", type=Path, required=True)
    step.add_argument("--step", required=True)
    step.add_argument("--out", type=Path, required=True)
    sweep = commands.add_parser("sweep-plot", help="batch-size sweep plot and operating point")
    sweep.add_argument("files", type=Path, nargs="+", help="ladder results of sweep scenarios")
    sweep.add_argument("--out", type=Path, required=True)
    sweep.add_argument("--label", action="append", default=[], help="one per file, in order")
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
            log.info("weights verified", extra={"weights": name, "path": str(path)})
        return 0

    if args.command == "export-onnx":
        from vms.worker.detector import DetectorSettings
        from vms.worker.onnx_detector import export_onnx
        from vms.worker.weights import ensure_weights, load_models_config

        models = load_models_config()
        imgsz = DetectorSettings().imgsz
        for name in args.names or sorted(models.weights):
            path = export_onnx(
                ensure_weights(models, name),
                models.weights[name].sha256,
                name,
                models.onnx_dir,
                imgsz,
            )
            log.info("onnx exported", extra={"weights": name, "path": str(path)})
        return 0

    if args.command == "onnx-parity":
        from vms.worker.parity import ParityConfig, run_parity

        _, ok = run_parity(ParityConfig(weights=args.weights, eval_images=args.eval_images))
        return 0 if ok else 1

    if args.command == "render":
        return _render(args)

    if args.command == "calibration":
        return _calibration(args.action)

    if args.command == "quantize":
        return _quantize(args.names)

    if args.command == "ladder":
        from vms.worker.ladder import run_ladder

        run_ladder(args.scenario)
        return 0

    if args.command == "sweep-plot":
        return _sweep_plot(args.files, args.out, args.label)

    if args.command == "ladder-step":
        from vms.worker.ladder import run_step_main

        return run_step_main(args.scenario, args.step, args.out)

    from vms.worker import bench as bench_module

    raw = yaml.safe_load(args.scenario.read_text(encoding="utf-8"))
    bench_module.run(bench_module.Scenario.model_validate(raw), raw)
    return 0


def _calibration(action: str) -> int:
    import sys

    from vms.common.settings import load_settings
    from vms.datasets.http import RetryPolicy, UrllibTransport
    from vms.worker import calibration as cal
    from vms.worker.quantize import QuantizeSettings

    p = load_settings(QuantizeSettings, "quantize").calibration
    spec = cal.load_spec(p.spec)
    if action == "build":
        images = cal.build(spec, UrllibTransport(), RetryPolicy(), p.images_dir, p.fetch_workers)
        if p.lock.exists():
            if cal.read_lock(p.lock) != images:
                sys.stdout.write(f"rebuilt calibration list differs from {p.lock}; it is frozen\n")
                return 1
            sys.stdout.write(f"rebuilt calibration list identical to {p.lock}\n")
            return 0
        digest = cal.write_lock(p.lock, images)
        sys.stdout.write(f"wrote {p.lock} ({len(images)} images, sha256 {digest})\n")
        return 0
    images = cal.read_lock(p.lock)
    n = cal.fetch(spec, images, UrllibTransport(), RetryPolicy(), p.images_dir, p.fetch_workers)
    sys.stdout.write(f"{len(images)} calibration images verified under {p.images_dir} ")
    sys.stdout.write(f"({n} downloaded)\n")
    return 0


def _sweep_plot(files: list[Path], out: Path, labels: list[str]) -> int:
    import sys

    from vms.bench.results import RunResult
    from vms.bench.settings import SERVICE as BENCH
    from vms.bench.settings import BenchSettings
    from vms.common.settings import load_settings
    from vms.worker.sweep import operating_point, plot_sweep, sweep_points

    if labels and len(labels) != len(files):
        raise SystemExit("give one --label per file, or none")
    slo_s = load_settings(BenchSettings, BENCH).slo.freshness_p95_s
    series = []
    for k, path in enumerate(files):
        result = RunResult.model_validate_json(path.read_text(encoding="utf-8"))
        points = sweep_points(result)
        label = labels[k] if labels else f"{result.hardware.cpu} ({result.hardware.logical_cpus})"
        op = operating_point(points, slo_s)
        sys.stdout.write(
            f"{label}: operating point batch {op.batch}, {op.images_per_s:.1f} images/s, "
            f"batch p95 {op.latency_ms_p95:.0f} ms\n"
        )
        for p in points:
            sys.stdout.write(
                f"  batch {p.batch:>2}: {p.images_per_s:6.1f} ({p.images_per_s_std:.1f}) img/s, "
                f"p50/p95 {p.latency_ms_p50:.0f}/{p.latency_ms_p95:.0f} ms, "
                f"{p.cpu_cores:.1f} cores\n"
            )
        series.append((label, points))
    plot_sweep(series, slo_s, out)
    sys.stdout.write(f"plot written to {out}\n")
    return 0


def _quantize(names: list[str]) -> int:
    from vms.common.settings import load_settings
    from vms.worker import calibration as cal
    from vms.worker.detector import DetectorSettings
    from vms.worker.onnx_detector import verified_export
    from vms.worker.quantize import QuantizeSettings, quantize_int8
    from vms.worker.weights import load_models_config

    settings = load_settings(QuantizeSettings, "quantize")
    p = settings.calibration
    spec = cal.load_spec(p.spec)
    images = cal.read_lock(p.lock)
    missing = [i.id for i in images if not cal.image_ok(p.images_dir, i)]
    if missing:
        raise SystemExit(f"{len(missing)} calibration images missing; run `make calibration-data`")
    models = load_models_config()
    for name in names or sorted(models.weights):
        sha = models.weights[name].sha256
        source, meta = verified_export(models.onnx_dir, name, sha)
        path = quantize_int8(
            source,
            meta.onnx_sha256,
            name,
            sha,
            cal.load_pixels(images, p.images_dir, spec),
            cal.lock_sha256(p.lock),
            settings.int8,
            models.onnx_dir,
            DetectorSettings().imgsz,
        )
        log.info("int8 model written", extra={"weights": name, "path": str(path)})
    return 0


def _render(args: argparse.Namespace) -> int:
    from vms.common.settings import load_settings
    from vms.loadgen.frames import load_clip
    from vms.loadgen.settings import FrameSettings
    from vms.storage.catalog import load_catalog
    from vms.worker.detector import build_detector
    from vms.worker.render import Segment, render, zones_for
    from vms.worker.settings import WorkerSettings
    from vms.worker.tracking import CameraTracker

    settings = load_settings(WorkerSettings, "worker")
    if args.weights:  # only the weights; the rest of the detector section stays as configured
        detector_settings = settings.detector.model_copy(update={"weights": args.weights})
        settings = settings.model_copy(update={"detector": detector_settings})
    detector = build_detector(settings.detector)
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
