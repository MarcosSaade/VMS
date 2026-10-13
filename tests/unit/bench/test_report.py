from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from tests.unit.bench.conftest import FakeIngest, FakePrometheus
from tests.unit.bench.test_runner import HEALTHY, make_runner, scenario
from vms.bench.cli import main
from vms.bench.report import render
from vms.bench.results import GitInfo, Hardware, RunResult, write_result
from vms.bench.settings import BenchSettings

REPO_ROOT = Path(__file__).resolve().parents[3]


def _result(
    kind: str, metrics: dict[str, Any], scenario_raw: dict[str, Any] | None = None
) -> RunResult:
    return RunResult(
        run_id="01RUN",
        name="t",
        kind=kind,
        started_at=datetime(2026, 10, 13, 12, tzinfo=UTC),
        ended_at=datetime(2026, 10, 13, 12, 5, tzinfo=UTC),
        seed=1,
        git=GitInfo(sha="abcdef0123", dirty=True),
        hardware=Hardware(cpu="Test CPU", logical_cpus=4, ram_bytes=1, platform="linux"),
        packages={},
        config={},
        scenario=scenario_raw,
        metrics=metrics,
    )


@pytest.fixture
def e2e(vrun: Callable[[Coroutine[Any, Any, Any]], Any], settings: BenchSettings) -> RunResult:
    ingest = FakeIngest()

    def values(name: str) -> Any:
        if name == "freshness_p95_s":
            return {1: 0.8, 2: 1.2, 4: 2.6}[ingest.runs[-1].cameras]
        return HEALTHY.get(name)

    sc = scenario()
    metrics = vrun(make_runner(sc, settings, ingest, FakePrometheus(settings, values)).run())
    return _result("bench.e2e", metrics.model_dump(mode="json"), sc.model_dump(mode="json"))


def test_e2e_table(e2e: RunResult) -> None:
    text = render(e2e)
    assert "`t` (bench.e2e), run `01RUN`, 2026-10-13, git `abcdef0+dirty`, Test CPU, no GPU" in text
    rows = [line for line in text.splitlines() if line.startswith("| ") and line[2].isdigit()]
    assert len(rows) == 3
    # cameras 2: offered 10 fps, freshness p50 / p95, SLO met in both repetitions
    assert rows[1].startswith("| 2 | 1 | 10.0 (0.0) | 10.0 (0.0) | 0 (0) | 0.40 / 1.20 |")
    assert rows[1].endswith("| 2/2 | 2/2 |")
    assert rows[2].endswith("| 0/2 | 2/2 |")
    assert "Rated capacity: 2 cameras at 5 FPS." in text


def test_generic_tables_for_other_kinds() -> None:
    spread = _result(
        "loadgen.run",
        {"aggregate": {"sent_fps": {"mean": 100.0, "std": 1.5, "min": 98.0, "max": 101.0}}},
    )
    assert "| sent_fps | 100 | 1.5 | 98 | 101 |" in render(spread)
    flat = _result("worker.model", {"aggregate": {"frames_per_s_mean": 18.776}})
    assert "| frames_per_s_mean | 18.776 |" in render(flat)
    assert "No aggregate" in render(_result("x", {}))


def test_report_command_renders_committed_results(capsys: pytest.CaptureFixture[str]) -> None:
    files = sorted((REPO_ROOT / "bench/results").glob("*.json"))
    assert main(["report", *map(str, files)]) == 0
    out = capsys.readouterr().out
    assert out.count("Source: `") == len(files)


def test_run_refuses_unknown_kinds(tmp_path: Path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text("name: s\nkind: nonsense\n")
    with pytest.raises(SystemExit, match="unknown scenario kind"):
        main(["run", "--scenario", str(path)])


def test_written_e2e_result_round_trips(e2e: RunResult, tmp_path: Path) -> None:
    path = write_result(tmp_path, e2e)
    assert RunResult.model_validate_json(path.read_text()) == e2e
