"""Profile scenarios: validation, sample classification and the per-thread summary."""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any

import pytest
from pydantic import ValidationError

from vms.bench.profile import (
    OTHER,
    STARTUP,
    Category,
    ProfileScenario,
    child_pids,
    summarize,
)

CATEGORIES = [
    Category(name="infer", pattern=r"^detect .*/vms/worker/detector\.py"),
    Category(name="decode", pattern=r"^decode_jpeg "),
    Category(name="db", pattern=r"/vms/storage/|psycopg"),
    Category(name="idle_loop", pattern=r"select\.py|selectors\.py"),
]
STARTUP_RE = r"^(build_detector|_find_and_load) "


def speedscope(
    frames: list[tuple[str, str]], profiles: dict[str, list[list[int]]]
) -> dict[str, Any]:
    return {
        "shared": {"frames": [{"name": n, "file": f} for n, f in frames]},
        "profiles": [
            {
                "name": f'Thread {i} "{name}"',
                "type": "sampled",
                "samples": s,
                "weights": [1] * len(s),
            }
            for i, (name, s) in enumerate(profiles.items())
        ],
    }


FRAMES = [
    ("run", "/app/src/vms/worker/service.py"),  # 0
    ("_analyze", "/app/src/vms/worker/service.py"),  # 1
    ("detect", "/app/src/vms/worker/detector.py"),  # 2
    ("forward", "/venv/torch/nn/modules/module.py"),  # 3
    ("decode_jpeg", "/app/src/vms/worker/service.py"),  # 4
    ("complete", "/app/src/vms/storage/work.py"),  # 5
    ("execute", "/venv/psycopg/cursor.py"),  # 6
    ("build_detector", "/app/src/vms/worker/detector.py"),  # 7
    ("select", "/usr/lib/python3.12/selectors.py"),  # 8
    ("mystery", "/venv/lib/thing.py"),  # 9
]


def test_innermost_own_frame_decides_the_category() -> None:
    doc = speedscope(
        FRAMES,
        {
            "infer": [[0, 1, 2, 3], [0, 1, 2, 3], [0, 1, 4]],
            "MainThread": [[0, 5, 6], [8], [9], [7, 2, 3]],
        },
    )
    out = summarize(doc, STARTUP_RE, CATEGORIES)
    assert out["threads"]["infer"] == {"infer": 2, "decode": 1}
    # db by our storage frame; idle_loop by a library frame (no own frame in the stack);
    # startup wins over everything; an unmatched stack is other.
    assert out["threads"]["MainThread"] == {"db": 1, "idle_loop": 1, OTHER: 1, STARTUP: 1}
    assert out["steady_samples"] == 6
    assert out["shares"]["infer"] == pytest.approx(2 / 6)
    assert list(out["shares"]) == ["infer", "decode", "db", "idle_loop", OTHER]


def test_weights_count_as_samples() -> None:
    doc = speedscope(FRAMES, {"infer": [[0, 1, 2]]})
    doc["profiles"][0]["weights"] = [5]
    assert summarize(doc, STARTUP_RE, CATEGORIES)["samples"] == {"infer": 5}


def scenario(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": "p",
        "service": "worker",
        "command": ["python", "-m", "vms.worker", "run"],
        "ready_url": "http://127.0.0.1:8090/readyz",
        "ingest": {"cameras": 2},
        "startup": STARTUP_RE,
        "categories": [c.model_dump() for c in CATEGORIES],
    }
    return base | overrides


def test_scenario_needs_exactly_one_load() -> None:
    ProfileScenario.model_validate(scenario())
    with pytest.raises(ValidationError, match="exactly one"):
        ProfileScenario.model_validate(scenario(query={"concurrency": 2}))
    with pytest.raises(ValidationError, match="exactly one"):
        ProfileScenario.model_validate(scenario(ingest=None))


def test_scenario_rejects_reserved_or_duplicate_categories() -> None:
    with pytest.raises(ValidationError, match="unique"):
        ProfileScenario.model_validate(scenario(categories=[{"name": "other", "pattern": "x"}]))
    with pytest.raises(ValidationError, match="unique"):
        ProfileScenario.model_validate(
            scenario(categories=[{"name": "a", "pattern": "x"}, {"name": "a", "pattern": "y"}])
        )
    with pytest.raises(ValidationError):
        ProfileScenario.model_validate(scenario(categories=[{"name": "a", "pattern": "("}]))


def test_child_pids_lists_direct_children() -> None:
    child = subprocess.Popen(
        [sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE
    )
    try:
        assert child.pid in child_pids(os.getpid())
    finally:
        child.communicate(b"")
    assert child.pid not in child_pids(os.getpid())
    assert child_pids(2**22 + 12345) == []  # no such process
