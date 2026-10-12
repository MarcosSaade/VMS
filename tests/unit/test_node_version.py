"""The host UI build must refuse an unsupported Node instead of failing obscurely.

Vite 8 loads a native rolldown binding that is an optional dependency with
`engines: node ^20.19.0 || >=22.12.0`. On an older Node (20.18 in the report), `npm ci`
silently skips that optional package and `vite build` then dies with "Cannot find
module '@rolldown/binding-linux-x64-gnu'", so `make zones` and `make ui-build` broke
with no hint about the cause. `engines` plus `engine-strict` turns this into an
EBADENGINE error naming the required version, and `.nvmrc` pins the same Node that CI
and the app image use.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
FRONTEND = REPO_ROOT / "frontend"


def _image_node_version(path: Path) -> str:
    match = re.search(r"node:(\d+\.\d+\.\d+)-alpine", path.read_text(encoding="utf-8"))
    assert match, f"no pinned node image in {path}"
    return match.group(1)


def test_npm_enforces_the_node_engine() -> None:
    package = json.loads((FRONTEND / "package.json").read_text(encoding="utf-8"))
    assert package["engines"]["node"] == ">=22.12.0"
    npmrc = (FRONTEND / ".npmrc").read_text(encoding="utf-8").splitlines()
    assert "engine-strict=true" in npmrc


def test_nvmrc_matches_ci_and_app_image() -> None:
    nvmrc = (FRONTEND / ".nvmrc").read_text(encoding="utf-8").strip()
    assert nvmrc == _image_node_version(REPO_ROOT / "deploy/docker/Dockerfile")
    assert nvmrc == _image_node_version(REPO_ROOT / ".github/workflows/ci.yml")
