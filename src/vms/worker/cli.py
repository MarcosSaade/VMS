"""`python -m vms.worker`: the inference worker and its tools.

run                         (default; make worker) lease cameras, claim frames, detect
weights [NAME ...]          (make models) fetch and verify pinned weights
bench --scenario FILE       model-only latency (bench/scenarios/worker-model-*.yaml)
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

    from vms.worker import bench as bench_module

    raw = yaml.safe_load(args.scenario.read_text(encoding="utf-8"))
    bench_module.run(bench_module.Scenario.model_validate(raw), raw)
    return 0
