"""Fixtures shared by every unit test."""

import os

import pytest

_ISOLATED_PREFIXES = ("VMS_", "WANDB_", "MLFLOW_", "OTEL_")


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """A developer's shell or .env must never change a unit test's result."""
    for key in list(os.environ):
        if key.startswith(_ISOLATED_PREFIXES):
            monkeypatch.delenv(key)
