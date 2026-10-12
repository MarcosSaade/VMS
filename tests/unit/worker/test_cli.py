"""`python -m vms.worker` subcommands that need no model or database."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vms.worker import cli


def test_weights_command_logs_each_verified_file(monkeypatch: pytest.MonkeyPatch) -> None:
    # Regression: logging `extra={"name": ...}` raised KeyError ("name" is a LogRecord
    # attribute), so `make models` crashed after downloading the first file.
    fetched: list[str] = []

    def fake_ensure(config: Any, name: str, **_: Any) -> Path:
        fetched.append(name)
        return Path(f"models/weights/{name}.pt")

    monkeypatch.setattr("vms.worker.weights.ensure_weights", fake_ensure)
    assert cli.main(["weights", "yolo11n", "yolo11s"]) == 0
    assert fetched == ["yolo11n", "yolo11s"]
