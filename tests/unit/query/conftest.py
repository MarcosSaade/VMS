from pathlib import Path

import pytest

from vms.common.settings import load_settings
from vms.query.catalog import QueryCatalog, catalog_from_rows
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


@pytest.fixture(scope="session")
def catalog() -> QueryCatalog:
    """The demo catalog's names, with the loading dock moved to Tokyo so plans that span
    cameras exercise more than one timezone."""
    return catalog_from_rows(
        cameras=[
            ("cam-002", "checkout", "Europe/London"),
            ("cam-001", "entrance", "Europe/London"),
            ("cam-004", "loading_dock", "Asia/Tokyo"),
            ("cam-003", "parking_lot", "Europe/London"),
        ],
        zones=[
            ("cam-001:front_door", "cam-001", "front_door"),
            ("cam-002:register_1", "cam-002", "register_1"),
            ("cam-002:register_2", "cam-002", "register_2"),
        ],
        classes=["person", "bicycle", "car", "truck", "dog"],
    )
