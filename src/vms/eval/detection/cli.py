"""`python -m vms.eval.detection`: the detection eval subset and its scoring.

Commands:
  build   select the subset, read ground truth, fetch images, write the frozen file list (once
          per version; when the list exists, rebuild and fail on any difference)
  fetch   download the listed images that are missing or corrupt (fresh clones)
  check   offline: the spec and subset files match the manifest
  run     score a detector on the subset; writes a results JSON under eval/results/
  compare BASE OTHER
          paired bootstrap of the mAP difference between two runs, from their saved
          detections (no inference); writes a results JSON with the D32 verdict
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Sequence
from pathlib import Path

from vms.common.logs import configure_logging, get_logger
from vms.common.settings import load_settings
from vms.datasets.http import RetryPolicy, UrllibTransport
from vms.eval.detection.settings import DetectionEvalSettings
from vms.eval.detection.subset import (
    SubsetManifest,
    UnpinnedError,
    build,
    fetch,
    load_spec,
    read_subset,
    spec_sha256,
    summarize,
    write_subset,
)

log = get_logger(__name__)


def cmd_build(settings: DetectionEvalSettings) -> int:
    p = settings.paths
    spec = load_spec(p.spec)
    spec_hash = spec_sha256(p.spec)
    try:
        images, ann_hash = build(
            spec, spec_hash, UrllibTransport(), RetryPolicy(), p.images_dir, p.fetch_workers
        )
    except UnpinnedError as exc:
        sys.stdout.write(
            f"Pin the annotations in {p.spec}: annotations_sha256: {exc.sha256}\n"
            "then run build again.\n"
        )
        return 1
    if p.subset.exists():
        old = read_subset(p.subset)
        if old != sorted(images, key=lambda i: i.id):
            sys.stdout.write(f"rebuilt subset differs from {p.subset}; it is frozen\n")
            return 1
        sys.stdout.write(f"rebuilt subset identical to {p.subset}\n")
        return 0
    subset_hash = write_subset(p.subset, images)
    manifest = summarize(images, spec, spec_hash, subset_hash, ann_hash)
    p.manifest.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(f"wrote {p.subset} (sha256 {subset_hash}) and {p.manifest}\n")
    return 0


def check_problems(settings: DetectionEvalSettings) -> list[str]:
    """The spec and subset on disk are the ones the manifest recorded."""
    p = settings.paths
    if not p.manifest.exists():
        return [f"{p.manifest} missing; run `python -m vms.eval.detection build`"]
    manifest = SubsetManifest.model_validate_json(p.manifest.read_text(encoding="utf-8"))
    problems = []
    if spec_sha256(p.spec) != manifest.spec_sha256:
        problems.append(f"{p.spec} changed after the subset was built; make a new version")
    if hashlib.sha256(p.subset.read_bytes()).hexdigest() != manifest.subset_sha256:
        problems.append(f"{p.subset} does not match the manifest's sha256")
    return problems


def cmd_fetch(settings: DetectionEvalSettings) -> int:
    p = settings.paths
    problems = check_problems(settings)
    if problems:
        sys.stdout.write("\n".join(problems) + "\n")
        return 1
    images = read_subset(p.subset)
    n = fetch(
        load_spec(p.spec), images, UrllibTransport(), RetryPolicy(), p.images_dir, p.fetch_workers
    )
    sys.stdout.write(f"{len(images)} images verified under {p.images_dir} ({n} downloaded)\n")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m vms.eval.detection",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log-level", default="INFO")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="build the frozen subset (once per version)")
    commands.add_parser("fetch", help="download and verify the listed images")
    commands.add_parser("check", help="offline: spec and subset match the manifest")
    run = commands.add_parser("run", help="score a detector on the subset")
    run.add_argument("--name", default=None, help="results name (default: detection-<weights>)")
    comp = commands.add_parser("compare", help="paired mAP difference between two runs")
    comp.add_argument("base", type=Path, help="results JSON of the reference run")
    comp.add_argument("other", type=Path, help="results JSON of the run compared with it")
    comp.add_argument("--name", default=None, help="results name (default: detection-compare)")
    args = parser.parse_args(argv)
    configure_logging("eval-detection", args.log_level)
    if args.command == "compare":
        from vms.eval.detection.compare import run_compare

        path = run_compare(args.base, args.other, Path(), args.name)
        sys.stdout.write(f"results: {path}\n")
        return 0
    settings = load_settings(DetectionEvalSettings, "eval-detection")

    if args.command == "build":
        return cmd_build(settings)
    if args.command == "fetch":
        return cmd_fetch(settings)
    if args.command == "check":
        problems = check_problems(settings)
        sys.stdout.write("\n".join(problems or ["subset consistent"]) + "\n")
        return 1 if problems else 0
    from vms.eval.detection.run import run as run_eval

    path = run_eval(settings, Path(), name=args.name)
    sys.stdout.write(f"results: {path}\n")
    return 0
