"""The results JSON every benchmark, eval and training run writes (AGENTS.md section 7).

Results files under `bench/results/` are the source of truth for reported numbers. Each one is
self-describing: what ran (resolved config, git SHA and dirty flag, seed), where (hardware,
package versions), when, and what came out (metrics). Task 2.3 builds the scenario runner on
this envelope; for now it is written directly by the tools that produce numbers.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from vms.common.cpus import cpu_quota

SCHEMA_VERSION = 1
COMMAND_TIMEOUT_S = 5.0
DEFAULT_PACKAGES = ("aiohttp", "numpy", "opencv-python-headless", "pydantic", "prometheus-client")


class GitInfo(BaseModel):
    sha: str | None
    dirty: bool | None


class Hardware(BaseModel):
    cpu: str
    logical_cpus: int | None
    cpu_quota: int | None = None  # whole CPUs a container may use, when it has a quota
    ram_bytes: int | None
    platform: str
    gpus: list[str] = Field(default_factory=list)  # "name, driver" per GPU from nvidia-smi


class RunResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: int = SCHEMA_VERSION
    run_id: str
    name: str  # scenario name
    kind: str  # e.g. loadgen.run, loadgen.selftest
    started_at: datetime
    ended_at: datetime
    seed: int
    git: GitInfo
    hardware: Hardware
    packages: dict[str, str | None]
    config: dict[str, Any]  # fully resolved settings
    scenario: dict[str, Any] | None = None  # the scenario file as written, if any
    metrics: dict[str, Any]
    cost_usd: float = 0.0
    notes: str = ""


def _run(cmd: list[str], cwd: Path | None = None) -> str | None:
    try:
        done = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True, timeout=COMMAND_TIMEOUT_S, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip()


def git_info(repo: Path | None = None) -> GitInfo:
    sha = _run(["git", "rev-parse", "HEAD"], repo)
    # Untracked files (results JSON from earlier runs, local data) do not change what ran.
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], repo)
    return GitInfo(sha=sha, dirty=None if status is None else bool(status))


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def _ram_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return None


def _gpus() -> list[str]:
    if shutil.which("nvidia-smi") is None:
        return []
    out = _run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"])
    return [line.strip() for line in (out or "").splitlines() if line.strip()]


def hardware() -> Hardware:
    return Hardware(
        cpu=_cpu_model(),
        logical_cpus=os.cpu_count(),
        cpu_quota=cpu_quota(),
        ram_bytes=_ram_bytes(),
        platform=platform.platform(),
        gpus=_gpus(),
    )


def package_versions(names: tuple[str, ...] = DEFAULT_PACKAGES) -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def results_path(results_dir: Path, result: RunResult) -> Path:
    stamp = result.started_at.strftime("%Y-%m-%d")
    return results_dir / f"{stamp}-{result.name}-{result.run_id}.json"


def write_result(results_dir: Path, result: RunResult) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_path(results_dir, result)
    path.write_text(result.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path
