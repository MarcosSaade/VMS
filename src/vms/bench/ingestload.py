"""Ingest load for one step: loadgen in its own process.

The runner does not drive cameras itself. It writes a one-off loadgen scenario (the step's
cameras, FPS, warmup and hold on top of the scenario's `ingest.loadgen` overrides), runs
`python -m vms.loadgen run` and reads the results JSON loadgen writes. So ingest load comes from
the exact code measured in E-009 to E-011, with its open-loop pacing, its schedule-lag health
check (`generator_ok`) and its own CPU accounting, and its event loop does not share a core
with the query clients.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

import yaml

from vms.common.logs import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class IngestRun:
    cameras: int
    fps: float
    warmup_s: float
    hold_s: float
    seed: int
    target_url: str
    name: str
    loadgen: dict[str, Any]


@dataclass(frozen=True)
class IngestResult:
    started_at: datetime  # loadgen's schedule start: the step's time zero
    summary: dict[str, Any]  # loadgen's steady-state summary for its single repetition
    config: dict[str, Any]  # loadgen's fully resolved settings
    run_id: str


class IngestLoad(Protocol):
    async def run(self, spec: IngestRun) -> IngestResult: ...


class LoadgenProcess:
    def __init__(self, *, timeout_margin_s: float, python: str = sys.executable) -> None:
        self._margin = timeout_margin_s
        self._python = python

    async def run(self, spec: IngestRun) -> IngestResult:
        with tempfile.TemporaryDirectory(prefix="vms-bench-") as tmp:
            tmp_dir = Path(tmp)
            scenario = tmp_dir / "scenario.yaml"
            scenario.write_text(yaml.safe_dump(loadgen_scenario(spec, tmp_dir)), encoding="utf-8")
            cmd = [self._python, "-m", "vms.loadgen", "run", "--scenario", str(scenario)]
            proc = await asyncio.create_subprocess_exec(*cmd)
            try:
                code = await asyncio.wait_for(
                    proc.wait(), timeout=spec.warmup_s + spec.hold_s + self._margin
                )
            except TimeoutError:
                proc.kill()
                await proc.wait()
                raise RuntimeError(f"loadgen did not finish step {spec.name}") from None
            if code != 0:
                raise RuntimeError(f"loadgen exited with {code} in step {spec.name}")
            return read_loadgen_result(tmp_dir)


def loadgen_scenario(spec: IngestRun, results_dir: Path) -> dict[str, Any]:
    config = dict(spec.loadgen)
    config.update(
        {
            "mode": "paced",
            "cameras": spec.cameras,
            "fps": spec.fps,
            "duration_s": spec.warmup_s + spec.hold_s,
            "warmup_s": spec.warmup_s,
            "target_url": spec.target_url,
            "seed": spec.seed,
            "results_dir": str(results_dir),
            "metrics_port": None,  # not scraped; avoids a port clash between steps
        }
    )
    return {"name": spec.name, "kind": "run", "repeats": 1, "config": config}


def read_loadgen_result(results_dir: Path) -> IngestResult:
    files = sorted(results_dir.glob("*.json"))
    if len(files) != 1:
        raise RuntimeError(f"expected one loadgen results file in {results_dir}, got {len(files)}")
    data = json.loads(files[0].read_text(encoding="utf-8"))
    summary = data["metrics"]["repetitions"][0]
    return IngestResult(
        started_at=datetime.fromisoformat(summary.get("started_at") or data["started_at"]),
        summary=summary,
        config=data["config"],
        run_id=data["run_id"],
    )
