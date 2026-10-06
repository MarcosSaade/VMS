from pathlib import Path

import pytest

from vms.common.settings import load_settings
from vms.query.settings import QuerySettings
from vms.query.sql_validator import SqlValidator

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def query_settings() -> QuerySettings:
    """The real configs/query.yaml, so tests exercise the shipped policy."""
    return load_settings(QuerySettings, "query", config_dir=REPO_ROOT / "configs")


@pytest.fixture(scope="session")
def validator(query_settings: QuerySettings) -> SqlValidator:
    return SqlValidator(query_settings.sql)
