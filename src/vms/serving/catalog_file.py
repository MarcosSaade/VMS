"""The eval catalog as a file, so the planner can run without the eval database (D40).

`python -m vms.eval catalog` writes `eval/structured/catalog.v1.json` from the loaded eval
database: cameras and zones in the order the query service reads them (the same SQL, so the
same collation order and the same prompt), the class list, the eval clock and the snapshot
manifest it came from. A pod builds the planner's exact system prompt from it.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from vms.query.catalog import QueryCatalog, catalog_from_rows

CATALOG_FILE = Path("eval/structured/catalog.v1.json")


def catalog_json(
    cameras: Iterable[tuple[Any, ...]],
    zones: Iterable[tuple[Any, ...]],
    classes: Iterable[str],
    now: dt.datetime,
    snapshot_manifest_sha256: str,
) -> dict[str, Any]:
    return {
        "cameras": [[str(c) for c in row] for row in cameras],
        "zones": [[str(c) for c in row] for row in zones],
        "classes": list(classes),
        "now": now.isoformat(),
        "snapshot_manifest_sha256": snapshot_manifest_sha256,
    }


def write_catalog(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def load_catalog_file(path: Path) -> tuple[QueryCatalog, dt.datetime]:
    data = json.loads(path.read_text(encoding="utf-8"))
    catalog = catalog_from_rows(data["cameras"], data["zones"], data["classes"])
    return catalog, dt.datetime.fromisoformat(data["now"])
