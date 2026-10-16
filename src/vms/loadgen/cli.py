"""`python -m vms.loadgen`: drive virtual cameras, measure the generator, or serve a stub.

    run       [--scenario FILE] [--cameras N] [--fps F] [--duration S] [--target-url URL]
              [--repeats R]                      paced (or unpaced) load against an ingest URL
    selftest  [--scenario FILE] [...]            unpaced, no I/O (NullTransport): the
                                                 generator's own maximum rate
    stub      [--port P] [--latency-ms MS] [--reject-ratio X] [--workers W]
                                                 stub ingest server (vms.loadgen.stub)

`run` and `selftest` write one results JSON per invocation to `results_dir`, with one entry per
repetition and an aggregate. A scenario file (`bench/scenarios/*.yaml`) supplies `name`,
`repeats` and a `config:` mapping that overrides `configs/loadgen.yaml`; command-line flags
override the scenario.
"""

from __future__ import annotations

import argparse
import asyncio
import random
import signal
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml
from prometheus_client import start_http_server
from pydantic import BaseModel, ConfigDict, Field

from vms.bench import stats
from vms.bench.results import RunResult, git_info, hardware, package_versions, write_result
from vms.common.clock import SystemClock
from vms.common.ids import new_ulid
from vms.common.observability import Observability, init_observability
from vms.common.settings import deep_merge, load_settings
from vms.loadgen.drift import schedule_drift
from vms.loadgen.frames import LoadedClip, load_clip
from vms.loadgen.recorder import Recorder
from vms.loadgen.runner import Runner
from vms.loadgen.schedule import plan_cameras
from vms.loadgen.settings import LoadgenSettings
from vms.loadgen.stub import serve
from vms.loadgen.transport import HttpTransport, NullTransport, RetryingTransport, Transport

SERVICE = "loadgen"
# Summary fields aggregated across repetitions (dotted paths into a repetition's summary).
AGGREGATED = (
    "sent_fps",
    "accepted_fps",
    "send_latency_ms.p50",
    "send_latency_ms.p95",
    "send_latency_ms.p99",
    "schedule_lag_ms.p99",
    "cpu.utilization_cores",
    "cpu.us_per_frame",
)


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""
    kind: Literal["run", "selftest"] = "run"
    repeats: int = Field(default=1, ge=1)
    config: dict[str, Any] = Field(default_factory=dict)


def load_scenario(path: Path) -> Scenario:
    return Scenario.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def resolve_settings(
    scenario_config: dict[str, Any], flags: dict[str, Any], config_dir: Path = Path("configs")
) -> LoadgenSettings:
    """configs/loadgen.yaml (+ env), then the scenario's `config:`, then command-line flags.

    Nested scenario sections are merged into the already-resolved section, so a scenario that
    sets `http: {max_retries: 2}` keeps the other `http` fields.
    """
    base = load_settings(LoadgenSettings, SERVICE, config_dir=config_dir).model_dump()
    overrides: dict[str, Any] = {}
    for key, value in scenario_config.items():
        current = base.get(key)
        overrides[key] = (
            deep_merge(current, value)
            if isinstance(current, dict) and isinstance(value, dict)
            else value
        )
    overrides.update({k: v for k, v in flags.items() if v is not None})
    return load_settings(LoadgenSettings, SERVICE, config_dir=config_dir, **overrides)


def aggregate(reps: Sequence[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    """Mean, sample standard deviation, min and max of key numbers across repetitions."""
    return stats.aggregate(reps, AGGREGATED)


def _load_clips(settings: LoadgenSettings, obs: Observability) -> dict[str, LoadedClip]:
    clips = {}
    # Clips a drift event switches cameras to are loaded too (vms.loadgen.drift).
    for clip_id in dict.fromkeys([*settings.clips.ids, *settings.drift.clips()]):
        t0 = time.perf_counter()
        clip = load_clip(
            settings.clips.data_dir, clip_id, settings.frame, settings.clips.max_frames
        )
        clips[clip_id] = clip
        obs.logger.info(
            "clip loaded",
            extra={
                "clip": clip_id,
                "frames": len(clip.frames),
                "size": f"{clip.width}x{clip.height}",
                "mib": round(clip.total_bytes / 2**20, 1),
                "seconds": round(time.perf_counter() - t0, 2),
            },
        )
    return clips


def _transport(settings: LoadgenSettings, kind: str, rng: random.Random) -> Transport:
    if kind == "selftest":
        return NullTransport()
    h = settings.http
    http = HttpTransport(
        settings.target_url,
        request_timeout_s=h.request_timeout_s,
        connect_timeout_s=h.connect_timeout_s,
        max_connections=settings.cameras * h.max_in_flight_per_camera,
    )
    if h.max_retries == 0:
        return http
    return RetryingTransport(
        http, max_retries=h.max_retries, base_s=h.retry_base_s, max_s=h.retry_max_s, rng=rng
    )


async def _repetition(
    settings: LoadgenSettings,
    kind: str,
    clips: dict[str, LoadedClip],
    obs: Observability,
    run_id: str,
    rep: int,
) -> tuple[dict[str, Any], bool]:
    """One repetition. Returns its summary and whether it was interrupted."""
    seed = settings.seed + rep
    plans = plan_cameras(
        settings.cameras,
        settings.clips.ids,
        {cid: len(c.frames) for cid, c in clips.items()},
        settings.fps,
        seed,
        assignment=settings.clips.assignment,
        prefix=settings.camera_prefix,
    )
    drift = schedule_drift(settings.drift, plans, seed)
    recorder = Recorder(obs.metrics, settings.warmup_s)
    transport = _transport(settings, kind, random.Random(seed))
    runner = Runner(
        settings,
        plans,
        clips,
        transport,
        recorder,
        clock=SystemClock(),
        run_id=f"{run_id}-r{rep}",
        tracer=obs.tracer,
        drift=drift,
    )
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, runner.stop)
    try:
        timing = await runner.run()
    finally:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.remove_signal_handler(sig)
        await transport.aclose()

    steady_s = max(0.0, min(timing.elapsed_s, settings.duration_s) - settings.warmup_s)
    paced = settings.mode == "paced" and kind == "run"
    summary = recorder.summary(steady_s, settings.cameras * settings.fps if paced else None)
    summary["seed"] = seed
    summary["started_at"] = timing.started_at  # the schedule's time zero (wall clock)
    summary["elapsed_s"] = round(timing.elapsed_s, 3)
    summary["drain_s"] = round(timing.drain_s, 3)
    summary["cpu"] = {
        "seconds": round(timing.cpu_s, 3),
        # Average busy cores over the run; ~1.0 means the single event loop is saturated.
        "utilization_cores": round(timing.cpu_s / timing.elapsed_s, 3),
        "us_per_frame": (
            round(timing.cpu_s * 1e6 / recorder.total_sent, 1) if recorder.total_sent else None
        ),
    }
    if drift:
        wall0 = datetime.fromisoformat(timing.started_at)
        summary["drift"] = {
            # Known onsets, for scoring detection delay (task 9.3).
            "events": [e.record(wall0) for events in drift.values() for e in events],
            "rendered_frames": len(runner.render_s),
            "render_ms": stats.percentiles_ms(runner.render_s),
        }
        summary["cameras"] = {p.camera_id: p.clip_id for p in plans}
    lag_p99 = summary["schedule_lag_ms"]["p99"]
    summary["generator_ok"] = lag_p99 is None or lag_p99 <= settings.max_schedule_lag_p99_ms
    if not summary["generator_ok"]:
        obs.logger.warning(
            "generator fell behind its schedule; offered load was lower than configured",
            extra={"schedule_lag_p99_ms": lag_p99, "limit_ms": settings.max_schedule_lag_p99_ms},
        )
    obs.logger.info("repetition done", extra={"rep": rep, **_headline(summary)})
    return summary, runner.stopped


def _headline(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "sent_fps": summary["sent_fps"],
        "accepted_fps": summary["accepted_fps"],
        "frames": summary["frames"],
        "send_latency_ms": summary["send_latency_ms"],
        "schedule_lag_p99_ms": summary["schedule_lag_ms"]["p99"],
        "cpu_cores": summary["cpu"]["utilization_cores"],
    }


async def _run_all(
    settings: LoadgenSettings,
    kind: str,
    name: str,
    repeats: int,
    scenario: dict[str, Any] | None,
    obs: Observability,
) -> Path:
    clips = _load_clips(settings, obs)
    run_id = new_ulid()
    started = datetime.now(UTC)
    reps: list[dict[str, Any]] = []
    for rep in range(repeats):
        summary, interrupted = await _repetition(settings, kind, clips, obs, run_id, rep)
        reps.append(summary)
        if interrupted:
            obs.logger.warning("interrupted; writing partial results", extra={"rep": rep})
            break
    result = RunResult(
        run_id=run_id,
        name=name,
        kind=f"loadgen.{kind}",
        started_at=started,
        ended_at=datetime.now(UTC),
        seed=settings.seed,
        git=git_info(),
        hardware=hardware(),
        packages=package_versions(),
        config=settings.model_dump(mode="json"),
        scenario=scenario,
        metrics={
            "transport": "null" if kind == "selftest" else "http",
            "repetitions": reps,
            "aggregate": aggregate(reps),
            "clips": {
                cid: {"frames": len(c.frames), "size": [c.width, c.height], "fps": c.fps}
                for cid, c in clips.items()
            },
        },
    )
    path = write_result(settings.results_dir, result)
    obs.logger.info("results written", extra={"path": str(path)})
    return path


def _serve_metrics(settings: LoadgenSettings, obs: Observability) -> None:
    if settings.metrics_port is None:
        return
    try:
        start_http_server(settings.metrics_port, registry=obs.metrics.registry)
    except OSError as exc:
        obs.logger.warning(
            "metrics port unavailable; continuing without /metrics",
            extra={"port": settings.metrics_port, "error": str(exc)},
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m vms.loadgen")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "selftest"):
        p = sub.add_parser(command)
        p.add_argument("--scenario", type=Path)
        p.add_argument("--cameras", type=int)
        p.add_argument("--fps", type=float)
        p.add_argument("--duration", dest="duration_s", type=float)
        p.add_argument("--warmup", dest="warmup_s", type=float)
        p.add_argument("--repeats", type=int)
        if command == "run":
            p.add_argument("--target-url")
            p.add_argument("--mode", choices=["paced", "unpaced"])
    p = sub.add_parser("stub")
    p.add_argument("--host")
    p.add_argument("--port", type=int)
    p.add_argument("--latency-ms", type=float)
    p.add_argument("--reject-ratio", type=float)
    p.add_argument("--workers", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "stub":
        base = load_settings(LoadgenSettings, SERVICE)
        stub_flags = {
            k: v
            for k, v in vars(args).items()
            if k in ("host", "port", "latency_ms", "reject_ratio", "workers") and v is not None
        }
        settings = load_settings(
            LoadgenSettings, SERVICE, stub=deep_merge(base.stub.model_dump(), stub_flags)
        )
        serve(settings.model_copy(update={"service": "loadgen-stub"}))
        return 0

    scenario = load_scenario(args.scenario) if args.scenario else None
    if scenario is not None and scenario.kind != args.command:
        raise SystemExit(f"scenario {scenario.name!r} is a {scenario.kind!r} scenario")
    flags = {
        "cameras": args.cameras,
        "fps": args.fps,
        "duration_s": args.duration_s,
        "warmup_s": args.warmup_s,
        "target_url": getattr(args, "target_url", None),
        "mode": getattr(args, "mode", None),
    }
    if args.command == "selftest":
        flags["mode"] = "unpaced"
    settings = resolve_settings(scenario.config if scenario else {}, flags)
    kind = args.command
    name = scenario.name if scenario else f"loadgen-{kind}"
    repeats = args.repeats or (scenario.repeats if scenario else 1)
    obs = init_observability(settings)
    _serve_metrics(settings, obs)
    try:
        asyncio.run(
            _run_all(
                settings,
                kind,
                name,
                repeats,
                scenario.model_dump() if scenario else None,
                obs,
            )
        )
    finally:
        obs.shutdown()
    return 0
