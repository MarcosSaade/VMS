from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from vms.loadgen.cli import Scenario, aggregate, load_scenario, resolve_settings

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS = REPO_ROOT / "configs"


def test_shipped_config_is_valid() -> None:
    s = resolve_settings({}, {}, config_dir=CONFIGS)
    assert (s.cameras, s.fps, s.mode) == (20, 5.0, "paced")
    assert (s.frame.width, s.frame.height, s.frame.jpeg_quality) == (1280, 720, 85)
    assert s.service == "loadgen"


def test_scenario_merges_into_sections_and_flags_win() -> None:
    s = resolve_settings(
        {"cameras": 50, "http": {"max_retries": 2}},
        {"cameras": 7, "fps": None},
        config_dir=CONFIGS,
    )
    assert s.cameras == 7  # flag beats scenario
    assert s.fps == 5.0  # unset flag leaves the file value
    assert s.http.max_retries == 2
    assert s.http.request_timeout_s == 5.0  # other fields of the section are kept


def test_scenario_typos_fail() -> None:
    with pytest.raises(ValidationError):
        resolve_settings({"http": {"max_retry": 2}}, {}, config_dir=CONFIGS)
    with pytest.raises(ValidationError):
        Scenario.model_validate({"name": "x", "repeat": 3})


def test_warmup_must_be_shorter_than_duration() -> None:
    with pytest.raises(ValidationError, match="warmup_s"):
        resolve_settings({}, {"duration_s": 5, "warmup_s": 5}, config_dir=CONFIGS)


@pytest.mark.parametrize("path", sorted((REPO_ROOT / "bench/scenarios").glob("loadgen-*.yaml")))
def test_committed_scenarios_resolve(path: Path) -> None:
    scenario = load_scenario(path)
    assert scenario.name == path.stem
    resolve_settings(scenario.config, {}, config_dir=CONFIGS)


def test_aggregate_across_repetitions() -> None:
    reps = [
        {"sent_fps": 100.0, "send_latency_ms": {"p99": 10.0}, "cpu": {"utilization_cores": 0.2}},
        {"sent_fps": 102.0, "send_latency_ms": {"p99": 14.0}, "cpu": {"utilization_cores": 0.2}},
        {"sent_fps": 98.0, "send_latency_ms": {"p99": None}, "cpu": {"utilization_cores": 0.2}},
    ]
    agg = aggregate(reps)
    assert agg["sent_fps"] == {"mean": 100.0, "std": 2.0, "min": 98.0, "max": 102.0}
    assert agg["send_latency_ms.p99"]["mean"] == 12.0  # None samples are skipped
    assert "accepted_fps" not in agg


def test_scenario_file_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "s.yaml"
    path.write_text(yaml.safe_dump({"name": "s", "kind": "selftest", "repeats": 3}))
    assert load_scenario(path) == Scenario(name="s", kind="selftest", repeats=3)
