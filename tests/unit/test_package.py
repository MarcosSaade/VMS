import importlib

import pytest

import vms

SUBPACKAGES = ["common", "loadgen", "ingest", "worker", "query", "drift", "storage", "bench"]


def test_version_is_set() -> None:
    assert vms.__version__ == "0.1.0"


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name: str) -> None:
    module = importlib.import_module(f"vms.{name}")
    assert module.__doc__
