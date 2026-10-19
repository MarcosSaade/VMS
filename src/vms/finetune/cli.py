"""`python -m vms.finetune` (task 9.5).

build                    resolve configs/finetune.v1.yaml once into the frozen list
                         configs/finetune.v1.lock.jsonl, fetching the images
fetch                    (make finetune-data) download and verify the listed images
run --scenario FILE      (make finetune) a `kind: finetune` scenario: train each run,
                         score base and fine-tuned models clean and drifted, summary JSON
evaluate --scenario FILE the scoring and summary only, on weights already trained (e.g.
                         copied back from a cloud session), checked against their records
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from vms.common.logs import configure_logging

SPEC = Path("configs/finetune.v1.yaml")
LOCK = Path("configs/finetune.v1.lock.jsonl")
IMAGES_DIR = Path("data/finetune/images/v1")
EVAL_SPEC = Path("configs/eval/detection.v1.yaml")
EVAL_SUBSET = Path("eval/detection/subset.v1.jsonl")
FETCH_WORKERS = 8


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vms.finetune",
        description=__doc__,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="resolve the spec into the frozen list (once)")
    commands.add_parser("fetch", help="download and verify the listed images")
    run = commands.add_parser("run", help="run a fine-tune scenario")
    run.add_argument("--scenario", type=Path, required=True)
    ev = commands.add_parser("evaluate", help="score already trained weights")
    ev.add_argument("--scenario", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    configure_logging("finetune", "INFO")
    if args.command in ("run", "evaluate"):
        from vms.finetune.run import run_finetune

        path = run_finetune(args.scenario, train=args.command == "run")
        sys.stdout.write(f"{path}\n")
        return 0

    from vms.datasets.http import RetryPolicy, UrllibTransport
    from vms.eval.detection.subset import UnpinnedError, read_subset, write_subset
    from vms.finetune import data

    spec = data.load_spec(SPEC)
    if args.command == "build":
        try:
            images, ann_hash = data.build(
                spec,
                UrllibTransport(),
                RetryPolicy(),
                IMAGES_DIR,
                EVAL_SPEC,
                EVAL_SUBSET,
                FETCH_WORKERS,
            )
        except UnpinnedError as exc:
            sys.stdout.write(f"pin coco.annotations_sha256: {exc.sha256} in {SPEC}, rerun\n")
            return 1
        if LOCK.exists():
            same = read_subset(LOCK) == images
            sys.stdout.write(f"rebuilt list {'identical to' if same else 'DIFFERS from'} {LOCK}\n")
            return 0 if same else 1
        digest = write_subset(LOCK, images)
        sys.stdout.write(f"wrote {LOCK} ({len(images)} images, sha256 {digest}; ann {ann_hash})\n")
        return 0
    images = read_subset(LOCK)
    n = data.fetch(spec, images, UrllibTransport(), RetryPolicy(), IMAGES_DIR, FETCH_WORKERS)
    sys.stdout.write(
        f"{len(images)} training images verified under {IMAGES_DIR} ({n} downloaded)\n"
    )
    return 0
