"""The versioned query assets referenced from configs/query.yaml are present and consistent."""

import json
from pathlib import Path

import pytest

from tests.unit.query.conftest import REPO_ROOT
from vms.query.fewshots import Category, FewShotExample, load_fewshots, normalize_question
from vms.query.plan import Intent
from vms.query.prompts import PLANNER_FIELDS, SQL_FIELDS, PromptTemplate
from vms.query.settings import QuerySettings
from vms.query.sql_validator import SqlValidator


@pytest.fixture(scope="module")
def fewshots(query_settings: QuerySettings) -> list[FewShotExample]:
    return load_fewshots(REPO_ROOT / query_settings.assets.fewshots)


def test_config_references_existing_assets(query_settings: QuerySettings) -> None:
    for path in query_settings.assets.model_dump().values():
        assert (REPO_ROOT / path).is_file(), path


def test_every_fewshot_plan_validates(fewshots: list[FewShotExample]) -> None:
    # load_fewshots validates each plan against QueryPlan; this guards the bank's coverage.
    assert len(fewshots) >= 25
    assert {e.category for e in fewshots} == set(Category)
    assert {e.plan.intent for e in fewshots} == set(Intent)
    kinds = {e.plan.filters.time_range.kind for e in fewshots if e.plan.filters.time_range}
    assert kinds == {"absolute", "last", "named"}


def test_every_fewshot_sql_passes_the_validator(
    fewshots: list[FewShotExample], validator: SqlValidator
) -> None:
    with_sql = [e for e in fewshots if e.sql is not None]
    assert with_sql
    for example in with_sql:
        assert example.sql is not None
        validator.validate(example.sql)  # raises with the rejection code on failure


def test_fewshot_bank_has_unsupported_examples_with_sql(fewshots: list[FewShotExample]) -> None:
    unsupported = [e for e in fewshots if not e.expressible_by_plan]
    assert len(unsupported) >= 3
    assert all(e.sql for e in unsupported)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


COUNT_ROW: dict[str, object] = {
    "id": "fs-001",
    "category": "count",
    "question": "How many cars today?",
    "plan": {"intent": "count", "target": "tracks"},
    "expressible_by_plan": True,
    "sql": "SELECT count(*) FROM tracks",
}


@pytest.mark.parametrize(
    ("change", "fragment"),
    [
        ({"plan": {"intent": "count"}}, "requires target"),
        ({"category": "ambiguous"}, "needs abstain_reason"),
        ({"plan": {"intent": "abstain", "abstain_reason": "ambiguous"}}, "must not abstain"),
        ({"expressible_by_plan": False}, "expressible_by_plan"),
        ({"sql": None}, "sql is required"),
        ({"category": "hybrid"}, "visual_query"),
        ({"id": "example-1"}, "id"),
        ({"answer": 3}, "Extra inputs"),
    ],
)
def test_loader_rejects_inconsistent_examples(
    tmp_path: Path, change: dict[str, object], fragment: str
) -> None:
    path = write_jsonl(tmp_path / "bank.jsonl", [{**COUNT_ROW, **change}])
    with pytest.raises(ValueError, match=fragment):
        load_fewshots(path)


def test_loader_reports_the_line(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "bank.jsonl", [COUNT_ROW, {**COUNT_ROW, "id": "bad"}])
    with pytest.raises(ValueError, match=r"bank\.jsonl:2"):
        load_fewshots(path)


def test_loader_rejects_duplicates(tmp_path: Path) -> None:
    dup_id = write_jsonl(tmp_path / "a.jsonl", [COUNT_ROW, {**COUNT_ROW, "question": "Other?"}])
    with pytest.raises(ValueError, match="duplicate ids"):
        load_fewshots(dup_id)
    dup_q = write_jsonl(
        tmp_path / "b.jsonl",
        [COUNT_ROW, {**COUNT_ROW, "id": "fs-002", "question": "how many  CARS today"}],
    )
    with pytest.raises(ValueError, match="duplicate questions"):
        load_fewshots(dup_q)


def test_normalize_question() -> None:
    assert normalize_question("  How many CARS today? ") == "how many cars today"


@pytest.mark.parametrize(
    ("attr", "fields"), [("planner_prompt", PLANNER_FIELDS), ("sql_prompt", SQL_FIELDS)]
)
def test_prompt_templates_have_expected_placeholders(
    query_settings: QuerySettings, attr: str, fields: frozenset[str]
) -> None:
    prompt = PromptTemplate.load(REPO_ROOT / getattr(query_settings.assets, attr), fields)
    rendered = prompt.render({name: f"<{name}>" for name in fields})
    assert "$" not in rendered
    assert all(f"<{name}>" in rendered for name in fields)


def test_template_render_is_strict(tmp_path: Path) -> None:
    path = tmp_path / "t.v1.txt"
    path.write_text("Hello $who, costs $$5", encoding="utf-8")
    prompt = PromptTemplate.load(path, frozenset({"who"}))
    assert prompt.render({"who": "you"}) == "Hello you, costs $5"
    with pytest.raises(ValueError, match="missing"):
        prompt.render({})
    with pytest.raises(ValueError, match="unknown"):
        prompt.render({"who": "you", "extra": 1})
    with pytest.raises(ValueError, match="placeholders"):
        PromptTemplate.load(path, frozenset({"who", "when"}))


def test_template_rejects_bad_placeholder_syntax(tmp_path: Path) -> None:
    path = tmp_path / "t.v1.txt"
    path.write_text("costs $5", encoding="utf-8")
    with pytest.raises(ValueError, match="escape"):
        PromptTemplate.load(path)
