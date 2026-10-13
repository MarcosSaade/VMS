import json
import sys
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tests.unit.bench.conftest import CONFIGS
from vms.bench.ingestload import IngestRun, LoadgenProcess, loadgen_scenario, read_loadgen_result
from vms.loadgen.cli import Scenario, resolve_settings

SPEC = IngestRun(
    cameras=4,
    fps=5.0,
    warmup_s=10.0,
    hold_s=30.0,
    seed=1101,
    target_url="http://127.0.0.1:8080",
    name="ramp-r1-s1",
    loadgen={"clips": {"ids": ["mot17-09-mini"]}, "http": {"max_retries": 1}},
)


def test_loadgen_scenario_resolves_with_the_step_on_top(tmp_path: Path) -> None:
    raw = loadgen_scenario(SPEC, tmp_path)
    scenario = Scenario.model_validate(raw)
    assert (scenario.kind, scenario.repeats) == ("run", 1)
    s = resolve_settings(scenario.config, {}, config_dir=CONFIGS)
    assert (s.cameras, s.fps, s.mode, s.seed) == (4, 5.0, "paced", 1101)
    assert (s.duration_s, s.warmup_s) == (40.0, 10.0)
    assert s.clips.ids == ["mot17-09-mini"]
    assert s.http.max_retries == 1
    assert s.http.request_timeout_s == 5.0  # the rest of configs/loadgen.yaml is kept
    assert s.results_dir == tmp_path
    assert s.metrics_port is None


def _write_result(directory: Path, started_at: str | None) -> None:
    rep: dict[str, Any] = {"sent_fps": 20.0, "frames": {"sent": 600}}
    if started_at:
        rep["started_at"] = started_at
    directory.joinpath("x.json").write_text(
        json.dumps(
            {
                "run_id": "01LG",
                "started_at": "2026-10-13T12:00:00+00:00",
                "config": {"cameras": 4},
                "metrics": {"repetitions": [rep]},
            }
        )
    )


def test_read_loadgen_result_prefers_the_schedule_start(tmp_path: Path) -> None:
    _write_result(tmp_path, "2026-10-13T12:00:03.5+00:00")
    result = read_loadgen_result(tmp_path)
    assert result.started_at == datetime(2026, 10, 13, 12, 0, 3, 500000, tzinfo=UTC)
    assert result.summary["sent_fps"] == 20.0
    assert result.run_id == "01LG"
    assert result.config == {"cameras": 4}


def test_read_loadgen_result_needs_exactly_one_file(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="got 0"):
        read_loadgen_result(tmp_path)


def _fake_python(tmp_path: Path, body: str) -> str:
    """A stand-in interpreter: a script run as `python -m vms.loadgen run --scenario FILE`."""
    script = tmp_path / "fake_python.py"
    script.write_text(body)
    launcher = tmp_path / "python"
    launcher.write_text(f'#!/bin/sh\nexec {sys.executable} {script} "$@"\n')
    launcher.chmod(0o755)
    return str(launcher)


def test_loadgen_process_runs_loadgen_and_reads_its_result(
    tmp_path: Path, vrun_real: Callable[[Coroutine[Any, Any, Any]], Any]
) -> None:
    python = _fake_python(
        tmp_path,
        """
import json, pathlib, sys, yaml
scenario = yaml.safe_load(pathlib.Path(sys.argv[-1]).read_text())
assert sys.argv[1:4] == ["-m", "vms.loadgen", "run"], sys.argv
out = pathlib.Path(scenario["config"]["results_dir"]) / "r.json"
rep = {"started_at": "2026-10-13T12:00:01+00:00", "cameras": scenario["config"]["cameras"]}
out.write_text(json.dumps({"run_id": "R", "started_at": rep["started_at"], "config": {},
                           "metrics": {"repetitions": [rep]}}))
""",
    )
    result = vrun_real(LoadgenProcess(timeout_margin_s=30, python=python).run(SPEC))
    assert result.summary["cameras"] == 4
    assert result.run_id == "R"


def test_loadgen_process_reports_a_failed_run(
    tmp_path: Path, vrun_real: Callable[[Coroutine[Any, Any, Any]], Any]
) -> None:
    python = _fake_python(tmp_path, "raise SystemExit(3)\n")
    with pytest.raises(RuntimeError, match="exited with 3"):
        vrun_real(LoadgenProcess(timeout_margin_s=30, python=python).run(SPEC))
