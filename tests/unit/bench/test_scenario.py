from pathlib import Path

import pytest
from pydantic import ValidationError

from tests.unit.bench.conftest import REPO_ROOT
from vms.bench.scenario import BenchScenario, load_questions, read_yaml

BASE = {"name": "x", "kind": "e2e", "warmup_s": 5, "hold_s": 20}


def test_steps_are_every_combination_cameras_outermost() -> None:
    sc = BenchScenario.model_validate(
        {**BASE, "ingest": {"cameras": [1, 4]}, "query": {"concurrency": [1, 2]}}
    )
    assert [(s.index, s.cameras, s.concurrency) for s in sc.steps()] == [
        (0, 1, 1),
        (1, 1, 2),
        (2, 4, 1),
        (3, 4, 2),
    ]
    assert sc.step_s == 25


def test_one_kind_of_load_is_enough() -> None:
    only_query = BenchScenario.model_validate({**BASE, "query": {"concurrency": [3]}})
    assert [(s.cameras, s.concurrency) for s in only_query.steps()] == [(0, 3)]
    only_ingest = BenchScenario.model_validate({**BASE, "ingest": {"cameras": [2]}})
    assert [(s.cameras, s.concurrency) for s in only_ingest.steps()] == [(2, 0)]


@pytest.mark.parametrize(
    "bad",
    [
        {},  # no load at all
        {"ingest": {"cameras": [0]}},
        {"ingest": {"cameras": []}},
        {"query": {"concurrency": [0]}},
        {"ingest": {"cameras": [1], "loadgen": {"cameras": 5}}},  # the runner sets it
        {"ingest": {"cameras": [1], "loadgen": {"duration_s": 5}}},
        {"ingest": {"cameras": [1]}, "repeat": 3},  # typo
        {"ingest": {"cameras": [1]}, "kind": "run"},
    ],
)
def test_invalid_scenarios_fail(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        BenchScenario.model_validate({**BASE, **bad})


def test_load_questions(tmp_path: Path) -> None:
    path = tmp_path / "q.jsonl"
    path.write_text('{"question": "a", "plan": {}}\n\n{"question": "b"}\n')
    assert load_questions(path) == ["a", "b"]
    path.write_text("\n")
    with pytest.raises(ValueError, match="no questions"):
        load_questions(path)


def test_shipped_questions_load() -> None:
    assert len(load_questions(REPO_ROOT / "configs/demo/canned_plans.v1.jsonl")) >= 5


E2E_SCENARIOS = [
    p
    for p in sorted((REPO_ROOT / "bench/scenarios").glob("*.yaml"))
    if read_yaml(p)["kind"] == "e2e"
]


def test_there_is_an_e2e_scenario() -> None:
    assert E2E_SCENARIOS


@pytest.mark.parametrize("path", E2E_SCENARIOS, ids=lambda p: p.stem)
def test_committed_e2e_scenarios_are_valid(path: Path) -> None:
    sc = BenchScenario.model_validate(read_yaml(path))
    assert sc.name == path.stem
    if sc.query:
        assert load_questions(REPO_ROOT / sc.query.questions)
