import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from vms.bench.results import (
    GitInfo,
    Hardware,
    RunResult,
    git_info,
    hardware,
    package_versions,
    results_path,
    write_result,
)
from vms.bench.runner import E2EMetrics


def test_git_info_outside_a_repo(tmp_path: Path) -> None:
    assert git_info(tmp_path) == GitInfo(sha=None, dirty=None)


def test_hardware_and_packages_are_described() -> None:
    hw = hardware()
    assert hw.cpu
    assert hw.logical_cpus is None or hw.logical_cpus >= 1
    versions = package_versions(("pydantic", "surely-not-installed-pkg"))
    assert versions["python"].startswith("3.12")
    assert versions["pydantic"]
    assert versions["surely-not-installed-pkg"] is None


def test_write_result_names_and_round_trips(tmp_path: Path) -> None:
    result = RunResult(
        run_id="01ABC",
        name="loadgen-selftest",
        kind="loadgen.selftest",
        started_at=datetime(2026, 10, 9, 12, tzinfo=UTC),
        ended_at=datetime(2026, 10, 9, 12, 1, tzinfo=UTC),
        seed=1234,
        git=GitInfo(sha="abc", dirty=False),
        hardware=Hardware(cpu="cpu", logical_cpus=4, ram_bytes=1, platform="linux"),
        packages={"python": "3.12"},
        config={"cameras": 1},
        metrics={"sent_fps": 1.0},
    )
    path = write_result(tmp_path / "results", result)
    assert path.name == "2026-10-09-loadgen-selftest-01ABC.json"
    assert RunResult.model_validate_json(path.read_text()) == result


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def test_untracked_files_do_not_make_the_tree_dirty(tmp_path: Path) -> None:
    """Results files from earlier runs are untracked; they must not mark later runs dirty."""
    _git(tmp_path, "init", "-q")
    (tmp_path / "code.py").write_text("x = 1\n")
    _git(tmp_path, "add", "code.py")
    _git(tmp_path, "commit", "-qm", "init")
    (tmp_path / "result.json").write_text("{}")
    assert git_info(tmp_path).dirty is False
    (tmp_path / "code.py").write_text("x = 2\n")
    assert git_info(tmp_path).dirty is True


RESULTS = sorted(Path(__file__).resolve().parents[3].joinpath("bench/results").glob("*.json"))


@pytest.mark.parametrize("path", RESULTS, ids=lambda p: p.stem)
def test_committed_results_match_the_schema(path: Path) -> None:
    """Every results file in the repo is a valid RunResult; e2e files also carry E2EMetrics."""
    result = RunResult.model_validate_json(path.read_text())
    assert result.git.sha
    assert result.ended_at >= result.started_at
    assert path == results_path(path.parent, result)
    if result.kind == "bench.e2e":
        E2EMetrics.model_validate(result.metrics)


def test_results_schema_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        RunResult.model_validate({"run_id": "x", "surprise": 1})
