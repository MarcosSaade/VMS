"""Profile one service under load with py-spy (`kind: profile` scenarios, task 2.6).

The service's container is stopped and the same code runs on the host as a child of
`py-spy record` (the kernel lets a tracer attach only to its descendants, and py-spy needs no
privileges that way), with the load the scenario names: loadgen cameras for the worker, closed
-loop query clients for the query service. When the load ends the child gets SIGTERM, shuts down
gracefully, py-spy writes a speedscope profile, and the container is started again.

The profile covers the process's whole life (py-spy has no delayed start): startup (imports,
model load) is a category of its own and is left out of the shares; idle threads are not
sampled at all (py-spy's default). Samples are wall-clock time of Python threads that were
running: time a thread spends in native code (PyTorch kernels, OpenCV, libpq) counts against
the Python frame that called it. PyTorch's own intra-op threads are not Python threads and do
not appear, so the shares say where the Python threads' time goes, not how many cores burn.

Each sample is assigned to the first category whose pattern matches a frame of the stack,
looking at frames from the innermost outwards, and among those only at frames from our own
code (`src/vms`) first; then at any frame. A stack nothing matches is `other`.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import os
import random
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vms.common.logs import get_logger

log = get_logger(__name__)

KIND = "profile"
STARTUP = "startup"
OTHER = "other"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Category(_Section):
    name: str
    pattern: str  # regex on "function (file)" of one frame

    @model_validator(mode="after")
    def _compiles(self) -> Category:
        try:
            re.compile(self.pattern)
        except re.error as exc:
            raise ValueError(f"category {self.name!r}: bad pattern: {exc}") from exc
        return self


class ProfileIngest(_Section):
    cameras: int = Field(ge=1)
    fps: float = Field(default=5.0, gt=0)
    loadgen: dict[str, Any] = Field(default_factory=dict)


class ProfileQuery(_Section):
    concurrency: int = Field(ge=1)
    questions: Path = Path("configs/demo/canned_plans.v1.jsonl")


class ProfileScenario(_Section):
    name: str
    description: str = ""
    kind: Literal["profile"] = "profile"
    service: str  # the compose service stopped while its code runs on the host
    command: list[str] = Field(min_length=1)  # run under py-spy; `python` is this interpreter
    env: dict[str, str] = Field(default_factory=dict)
    ready_url: str
    ready_timeout_s: float = Field(default=180.0, gt=0)
    rate_hz: int = Field(default=100, ge=1, le=1000)
    warmup_s: float = Field(default=20.0, ge=0)
    duration_s: float = Field(default=60.0, gt=0)
    ingest: ProfileIngest | None = None
    query: ProfileQuery | None = None
    startup: str  # regex for frames of startup work (imports, model load), left out of shares
    categories: list[Category] = Field(min_length=1)
    results_dir: Path = Path("bench/results")
    profiles_dir: Path = Path("bench/profiles")

    @model_validator(mode="after")
    def _one_load(self) -> ProfileScenario:
        if (self.ingest is None) == (self.query is None):
            raise ValueError("a profile scenario needs exactly one of `ingest` or `query`")
        names = [c.name for c in self.categories]
        if len(set(names)) != len(names) or {STARTUP, OTHER} & set(names):
            raise ValueError(f"category names must be unique and not {STARTUP!r} or {OTHER!r}")
        return self


# --- Summary ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Frame:
    name: str
    file: str

    @property
    def label(self) -> str:
        return f"{self.name} ({self.file})"

    @property
    def ours(self) -> bool:
        return "/vms/" in self.file


def classify(
    stack: Sequence[Frame],
    startup: re.Pattern[str],
    categories: Sequence[tuple[str, re.Pattern[str]]],
) -> str:
    """Category of one sample; `stack` is outermost first (speedscope's order)."""
    if any(startup.search(f.label) for f in stack):
        return STARTUP
    inner_first = list(reversed(stack))
    for frames in ([f for f in inner_first if f.ours], inner_first):
        for frame in frames:
            for name, pattern in categories:
                if pattern.search(frame.label):
                    return name
    return OTHER


def summarize(
    speedscope: dict[str, Any], startup: str, categories: Sequence[Category]
) -> dict[str, Any]:
    """Seconds per thread and category, and each category's share of non-startup seconds.

    py-spy weights each sample by its interval in seconds (`unit: seconds`, 1/rate each);
    a profile without weights counts one per sample."""
    frames = [Frame(f["name"], f.get("file") or "") for f in speedscope["shared"]["frames"]]
    rules = [(c.name, re.compile(c.pattern)) for c in categories]
    startup_re = re.compile(startup)
    threads: dict[str, dict[str, float]] = {}
    total: dict[str, float] = {}
    samples = 0
    for profile in speedscope["profiles"]:
        seconds: dict[str, float] = {}
        weights = profile.get("weights") or [1.0] * len(profile["samples"])
        for sample, weight in zip(profile["samples"], weights, strict=True):
            category = classify([frames[i] for i in sample], startup_re, rules)
            seconds[category] = seconds.get(category, 0.0) + float(weight)
            samples += 1
        threads[_thread_name(profile["name"])] = seconds
        for k, v in seconds.items():
            total[k] = total.get(k, 0.0) + v
    steady = sum(v for k, v in total.items() if k != STARTUP)
    order = [c.name for c in categories] + [OTHER]
    return {
        "samples": samples,
        "seconds": total,
        "steady_seconds": steady,
        "shares": {k: (total.get(k, 0.0) / steady if steady else None) for k in order},
        "threads": threads,
    }


def _thread_name(raw: str) -> str:
    # py-spy names profiles 'Thread 123 "MainThread"'; ids change per run, names do not.
    match = re.search(r'"(.*)"', raw)
    return match.group(1) if match else raw


# --- Run ----------------------------------------------------------------------------------


def child_pids(pid: int) -> list[int]:
    path = Path(f"/proc/{pid}/task/{pid}/children")
    try:
        return [int(p) for p in path.read_text().split()]
    except (OSError, ValueError):
        return []


def wait_ready(url: str, timeout_s: float, proc: subprocess.Popen[bytes]) -> float:
    """Seconds until `url` answers 200; raises if the process dies or time runs out."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if proc.poll() is not None:
            raise RuntimeError(f"profiled process exited with {proc.returncode} before ready")
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return time.monotonic() - t0
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1.0)
    raise RuntimeError(f"{url} not ready after {timeout_s} s")


def compose_command(env_file: str = ".env") -> list[str]:
    return [
        "docker", "compose", "--env-file", env_file,
        "-f", "deploy/compose/docker-compose.yml", "--profile", "app",
    ]  # fmt: skip


async def _load(
    scenario: ProfileScenario, seed: int, ingest_url: str, query_url: str, timeout_s: float
) -> dict[str, Any]:
    seconds = scenario.warmup_s + scenario.duration_s
    if scenario.ingest is not None:
        from vms.bench.ingestload import IngestRun, LoadgenProcess

        result = await LoadgenProcess(timeout_margin_s=120.0).run(
            IngestRun(
                cameras=scenario.ingest.cameras,
                fps=scenario.ingest.fps,
                warmup_s=scenario.warmup_s,
                hold_s=scenario.duration_s,
                seed=seed,
                target_url=ingest_url,
                name=scenario.name,
                loadgen=scenario.ingest.loadgen,
            )
        )
        return {"loadgen": result.summary}
    assert scenario.query is not None
    from vms.bench.queryload import HttpQueryClient, run_clients
    from vms.bench.queryload import summarize as summarize_queries
    from vms.bench.scenario import load_questions

    q = scenario.query
    client = HttpQueryClient(query_url, timeout_s=timeout_s, max_connections=q.concurrency)
    stop = asyncio.Event()
    start = time.time()
    asyncio.get_running_loop().call_later(seconds, stop.set)
    try:
        samples = await run_clients(
            client,
            load_questions(q.questions),
            q.concurrency,
            stop,
            wall_now=time.time,
            rng=random.Random(seed),
        )
    finally:
        await client.aclose()
    window_start = start + scenario.warmup_s
    return {
        "query": summarize_queries(samples, window_start, window_start + scenario.duration_s, None)
    }


def run_profile(path: Path, raw: dict[str, Any]) -> Path:
    """Run a profile scenario end to end; returns the results file."""
    import sys

    from vms.bench.results import RunResult, git_info, hardware, package_versions, write_result
    from vms.bench.settings import SERVICE, BenchSettings
    from vms.common.ids import new_ulid
    from vms.common.settings import load_settings

    scenario = ProfileScenario.model_validate(raw)
    settings = load_settings(BenchSettings, SERVICE)
    run_id = new_ulid()
    started = datetime.now(UTC)
    day = started.strftime("%Y-%m-%d")
    scenario.profiles_dir.mkdir(parents=True, exist_ok=True)
    out = scenario.profiles_dir / f"{day}-{scenario.name}-{run_id}.speedscope.json"
    command = [sys.executable if c == "python" else c for c in scenario.command]
    pyspy = [
        str(Path(sys.executable).with_name("py-spy")), "record", "--format", "speedscope",
        "--rate", str(scenario.rate_hz), "--threads", "-o", str(out), "--", *command,
    ]  # fmt: skip
    compose = compose_command()
    subprocess.run([*compose, "stop", scenario.service], check=True)
    proc: subprocess.Popen[bytes] | None = None
    try:
        proc = subprocess.Popen(pyspy, env={**os.environ, **scenario.env})
        ready_s = wait_ready(scenario.ready_url, scenario.ready_timeout_s, proc)
        log.info("profiled process ready", extra={"ready_s": round(ready_s, 1)})
        load = asyncio.run(
            _load(
                scenario,
                settings.seed,
                settings.ingest_url,
                settings.query_url,
                settings.query_timeout_s,
            )
        )
    finally:
        if proc is not None:
            for child in child_pids(proc.pid):
                os.kill(child, signal.SIGTERM)
            proc.wait(timeout=120)
        subprocess.run([*compose, "start", scenario.service], check=True)

    speedscope = json.loads(out.read_text(encoding="utf-8"))
    summary = summarize(speedscope, scenario.startup, scenario.categories)
    gz = out.with_suffix(out.suffix + ".gz")
    gz.write_bytes(gzip.compress(out.read_bytes(), mtime=0))
    out.unlink()
    result = RunResult(
        run_id=run_id,
        name=scenario.name,
        kind=f"bench.{KIND}",
        started_at=started,
        ended_at=datetime.now(UTC),
        seed=settings.seed,
        git=git_info(),
        hardware=hardware(),
        packages=package_versions((*_packages(), "py-spy")),
        config={"bench": settings.model_dump(mode="json")},
        scenario=raw,
        metrics={"profile_file": str(gz), "ready_s": ready_s, "load": load, **summary},
        notes="py-spy wall-clock samples of running Python threads (see vms.bench.profile). "
        "Open the profile at https://www.speedscope.app (gunzip first).",
    )
    results = write_result(scenario.results_dir, result)
    log.info("profile written", extra={"path": str(results), "profile": str(gz)})
    return results


def _packages() -> tuple[str, ...]:
    from vms.bench.results import DEFAULT_PACKAGES

    return (*DEFAULT_PACKAGES, "torch", "ultralytics", "onnxruntime")
