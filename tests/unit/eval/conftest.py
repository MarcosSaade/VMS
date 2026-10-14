"""Fixtures for the eval tests: the shipped query policy and the eval snapshot's catalog."""

from __future__ import annotations

from pathlib import Path

import pytest

from vms.common.settings import load_settings
from vms.query.catalog import QueryCatalog, catalog_from_rows
from vms.query.settings import QuerySettings
from vms.storage.catalog import load_catalog

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def settings() -> QuerySettings:
    return load_settings(QuerySettings, "query", config_dir=REPO_ROOT / "configs")


@pytest.fixture(scope="session")
def catalog(settings: QuerySettings) -> QueryCatalog:
    """configs/eval/cameras.v1.yaml as the query service would load it from the eval DB."""
    cat = load_catalog(REPO_ROOT / "configs/eval/cameras.v1.yaml")
    return catalog_from_rows(
        [(c.camera_id, c.name, c.timezone) for c in cat.cameras],
        [(c.zone_id(z), c.camera_id, z.name) for c in cat.cameras for z in c.zones],
        settings.classes,
    )
