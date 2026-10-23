"""`python -m vms.bench`: run any benchmark scenario, or render results as Markdown.

    run     --scenario FILE      run a scenario; dispatches on its `kind`:
                                   e2e            ingest and query load against the running
                                                  stack (vms.bench.runner)
                                   run, selftest  loadgen alone (vms.loadgen)
                                   model          worker model-only latency (vms.worker bench)
                                   ladder         precision ladder per backend and batch size
                                                  (vms.worker.ladder)
                                   profile        one service under load with py-spy
                                                  (vms.bench.profile)
                                   finetune       detector fine-tune and before/after mAP
                                                  (vms.finetune, task 9.5)
                                   llm            LLM serving load and plan recording
                                                  (vms.serving, task 8.1)
    report  FILE [FILE ...]      Markdown tables of results files, for docs/BENCHMARKS.md

`make bench SCENARIO=bench/scenarios/<name>.yaml` runs the first. An e2e scenario needs the
stack and the application services up (`make demo`, or `make up` plus the services), with
every endpoint taken from `configs/bench.yaml`.
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from vms.bench.ingestload import LoadgenProcess
from vms.bench.prom import HttpPrometheus, http_fetch_json
from vms.bench.queryload import HttpQueryClient, QueryClient
from vms.bench.report import render
from vms.bench.results import RunResult, git_info, hardware, package_versions, write_result
from vms.bench.runner import BenchRunner, E2EMetrics
from vms.bench.scenario import KIND, BenchScenario, load_questions, read_yaml
from vms.bench.settings import SERVICE, BenchSettings
from vms.common.clock import SystemClock
from vms.common.ids import new_ulid
from vms.common.logs import configure_logging, get_logger
from vms.common.settings import load_settings

log = get_logger(__name__)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m vms.bench", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="run a benchmark scenario")
    run.add_argument("--scenario", type=Path, required=True)
    report = commands.add_parser("report", help="Markdown tables of results files")
    report.add_argument("files", type=Path, nargs="+")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "report":
        for path in args.files:
            result = RunResult.model_validate_json(path.read_text(encoding="utf-8"))
            sys.stdout.write(render(result) + f"\n\nSource: `{path}`\n\n")
        return 0

    kind = read_yaml(args.scenario).get("kind")
    if kind in ("run", "selftest"):
        from vms.loadgen.cli import main as loadgen_main

        return loadgen_main([kind, "--scenario", str(args.scenario)])
    if kind == "model":
        from vms.worker.cli import main as worker_main  # imports torch: only when needed

        return worker_main(["bench", "--scenario", str(args.scenario)])
    if kind == "ladder":
        from vms.worker.cli import main as worker_main

        return worker_main(["ladder", "--scenario", str(args.scenario)])
    if kind == "finetune":
        from vms.finetune.run import run_finetune  # imports torch: only when needed

        configure_logging(SERVICE, "INFO")
        run_finetune(args.scenario)
        return 0
    if kind == "llm":
        from vms.serving.run import run_llm

        configure_logging(SERVICE, "INFO")
        run_llm(args.scenario)
        return 0
    if kind == "profile":
        from vms.bench.profile import run_profile

        configure_logging(SERVICE, "INFO")
        run_profile(args.scenario, read_yaml(args.scenario))
        return 0
    if kind != KIND:
        raise SystemExit(f"{args.scenario}: unknown scenario kind {kind!r}")
    return run_e2e(args.scenario)


def run_e2e(path: Path) -> int:
    raw = read_yaml(path)
    scenario = BenchScenario.model_validate(raw)
    settings = load_settings(BenchSettings, SERVICE)
    configure_logging(SERVICE, settings.log.level)
    questions = load_questions(scenario.query.questions) if scenario.query else []

    def query_client(concurrency: int) -> QueryClient:
        return HttpQueryClient(
            settings.query_url, timeout_s=settings.query_timeout_s, max_connections=concurrency
        )

    runner = BenchRunner(
        scenario,
        settings,
        ingest=LoadgenProcess(timeout_margin_s=settings.loadgen_timeout_margin_s),
        query_client=query_client,
        questions=questions,
        prom=HttpPrometheus(settings.prometheus_url, http_fetch_json(settings.http_timeout_s)),
        clock=SystemClock(),
    )
    run_id = new_ulid()
    started = datetime.now(UTC)
    metrics = asyncio.run(_run(runner))
    result = RunResult(
        run_id=run_id,
        name=scenario.name,
        kind=f"bench.{KIND}",
        started_at=started,
        ended_at=datetime.now(UTC),
        seed=runner.seed,
        git=git_info(),
        hardware=hardware(),
        packages=package_versions(),
        config=settings.model_dump(mode="json"),
        scenario=raw,
        metrics=metrics.model_dump(mode="json"),
    )
    out = write_result(settings.results_dir, result)
    log.info("results written", extra={"path": str(out)})
    sys.stdout.write(render(result) + "\n")
    return 0


async def _run(runner: BenchRunner) -> E2EMetrics:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, runner.stop)
    try:
        return await runner.run()
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
