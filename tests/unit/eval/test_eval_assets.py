"""The committed structured-query eval set (task 3.1) is consistent, covers what the plan
promises, stays clear of the few-shot bank, and is pinned where results will cite it."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pytest

from vms.common.settings import load_settings
from vms.eval.cli import check_dataset, query_settings
from vms.eval.dataset import Category, EvalItem, load_items, read_dataset_manifest
from vms.eval.settings import EvalSettings
from vms.eval.snapshot import read_manifest
from vms.query.catalog import QueryCatalog
from vms.query.compiler import PostgresCompiler
from vms.query.plan import Intent
from vms.query.settings import QuerySettings
from vms.query.sql_validator import SqlValidator

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def eval_settings() -> EvalSettings:
    return load_settings(EvalSettings, "eval", config_dir=REPO_ROOT / "configs")


@pytest.fixture(scope="module")
def items(eval_settings: EvalSettings) -> list[EvalItem]:
    return load_items(REPO_ROOT / eval_settings.structured.items)


def test_dataset_is_consistent(eval_settings: EvalSettings) -> None:
    # Items valid, no few-shot leakage or undeclared paraphrases, split files and manifest
    # in step with the items, the seed and the snapshot.
    assert check_dataset(eval_settings, REPO_ROOT) == []


def test_snapshot_manifest_pins_the_spec_and_catalog(eval_settings: EvalSettings) -> None:
    manifest = read_manifest(REPO_ROOT / eval_settings.snapshot.manifest)
    for path, digest in ((manifest.spec, manifest.spec_sha256),
                         (manifest.catalog, manifest.catalog_sha256)):  # fmt: skip
        assert hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest() == digest, (
            f"{path} changed after snapshot v{manifest.version} was frozen: make a new version"
        )
    assert manifest.synthetic


def test_coverage(items: list[EvalItem]) -> None:
    assert 110 <= len(items) <= 140
    per_category = Counter(i.category for i in items)
    assert set(per_category) == set(Category)
    assert min(per_category.values()) >= 6, per_category
    intents = {i.plan.intent for i in items}
    assert intents == set(Intent)
    kinds = {i.plan.filters.time_range.kind for i in items if i.plan.filters.time_range}
    assert kinds == {"absolute", "last", "named"}
    tags = Counter(t for i in items for t in i.tags)
    for tag in ("tz", "dst", "outage", "paraphrase"):
        assert tags[tag] >= 2, tag
    redteam = {t.split(":")[1] for t in tags if t.startswith("redteam:")}
    assert {"ddl", "dml", "injection", "exfiltration", "exhaustion"} <= redteam
    assert sum(i.ordered for i in items) >= 15


def test_every_gold_sql_passes_the_validator(
    items: list[EvalItem],
    settings: QuerySettings,
) -> None:
    validator = SqlValidator(settings.sql)
    for i in items:
        if i.sql is not None:
            validator.validate(i.sql)


def test_every_gold_plan_compiles_on_the_eval_catalog(
    items: list[EvalItem],
    settings: QuerySettings,
    catalog: QueryCatalog,
) -> None:
    compiler = PostgresCompiler(catalog, default_limit=100, max_limit=1000)
    validator = SqlValidator(settings.sql)
    now = read_manifest(REPO_ROOT / "eval/structured/snapshot.v1.json").now
    for i in items:
        if i.plan.intent is not Intent.ABSTAIN:
            validator.validate(compiler.compile(i.plan, now).sql)


def test_test_split_hash_is_recorded_in_benchmarks(eval_settings: EvalSettings) -> None:
    manifest = read_dataset_manifest(REPO_ROOT / eval_settings.structured.manifest)
    assert manifest.test_sha256 in (REPO_ROOT / "docs/BENCHMARKS.md").read_text()


def test_query_settings_come_from_the_repo() -> None:
    assert query_settings(REPO_ROOT).assets.fewshots == Path("eval/fewshot/structured.v1.jsonl")
