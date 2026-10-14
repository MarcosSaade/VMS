"""Eval items: validation rules per category, seeded splits that keep paraphrase groups
together, and the leakage checks against the few-shot bank."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from vms.eval.compare import ResultSet
from vms.eval.dataset import (
    EvalItem,
    Split,
    assign_splits,
    built_problems,
    leakage_problems,
    load_pinned_splits,
    near_duplicates,
    split_counts,
    structure_problems,
    write_pinned_splits,
)

COUNT_PLAN = {"intent": "count", "target": "tracks", "filters": {"cameras": ["lobby"]}}


def item(n: int, **fields: Any) -> EvalItem:
    data: dict[str, Any] = {
        "id": f"sq-{n:03d}",
        "category": "count",
        "question": f"How many people were in the lobby on day {n}?",
        "plan": COUNT_PLAN,
        "expressible_by_plan": True,
        "sql": "SELECT count(*) AS n FROM tracks t",
    }
    data.update(fields)
    return EvalItem.model_validate(data)


def abstaining(n: int, category: str, reason: str, **fields: Any) -> EvalItem:
    return item(
        n,
        category=category,
        plan={"intent": "abstain", "abstain_reason": reason},
        expressible_by_plan=category != "unsupported",
        sql="SELECT 1 AS x FROM tracks t" if category == "unsupported" else None,
        **fields,
    )


def test_valid_items_of_each_kind() -> None:
    item(1)
    abstaining(2, "unsupported", "unsupported")
    abstaining(3, "ambiguous", "ambiguous")
    abstaining(4, "unanswerable", "unanswerable")
    abstaining(5, "adversarial", "disallowed", tags=["redteam:ddl"])


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"category": "adversarial"}, "needs a plan abstaining with disallowed"),
        (
            {"plan": {"intent": "abstain", "abstain_reason": "unsupported"}},
            "must not abstain",
        ),
        ({"expressible_by_plan": False}, "expressible_by_plan must be false exactly"),
        ({"sql": None}, "sql is required exactly"),
        ({"id": "fs-001"}, "String should match pattern"),
        ({"group": "Paraphrase 1"}, "String should match pattern"),
        (
            {
                "category": "frames",
                "plan": {"intent": "frames", "target": "frames", "visual_query": "a red car"},
            },
            "visual_query needs CLIP",
        ),
    ],
)
def test_invalid_items(fields: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        item(1, **fields)


def test_no_answer_categories_take_no_sql() -> None:
    with pytest.raises(ValidationError, match="sql is required exactly"):
        item(
            1,
            category="ambiguous",
            plan={"intent": "abstain", "abstain_reason": "ambiguous"},
            sql="SELECT 1 AS x FROM tracks t",
        )


def test_json_round_trip_keeps_flags_explicit() -> None:
    original = item(1, ordered=True, tags=["tz"], expected=ResultSet(columns=["n"], rows=[[3]]))
    text = original.to_json()
    assert '"expressible_by_plan": true' in text
    assert '"ordered": true' in text
    assert EvalItem.model_validate_json(text) == original
    assert '"ordered": false' in item(2).to_json()


def test_structure_problems() -> None:
    items = [
        item(1),
        item(1, question="Another question?"),
        item(2, question="How many people were in the lobby on day 1"),
        item(3, group="g-x"),
        abstaining(4, "unsupported", "unsupported", group="g-x"),
    ]
    assert structure_problems(items) == [
        "sq-001: duplicate id",
        "sq-002: same question as sq-001",
        "group g-x mixes categories ['count', 'unsupported']",
    ]


def many(categories: dict[str, int]) -> list[EvalItem]:
    items, n = [], 1
    for category, count in categories.items():
        for _ in range(count):
            if category == "count":
                items.append(item(n, question=f"Count question number {n}?"))
            else:
                items.append(
                    abstaining(n, "ambiguous", "ambiguous", question=f"Vague question {n}?")
                )
            n += 1
    return items


def test_splits_are_stratified_seeded_and_reproducible() -> None:
    items = many({"count": 20, "ambiguous": 7})
    splits = assign_splits(items, seed=31, test_fraction=0.5)
    assert splits == assign_splits(items, seed=31, test_fraction=0.5)
    assert splits != assign_splits(items, seed=32, test_fraction=0.5)
    by_category = Counter((i.category.value, splits[i.id]) for i in items)
    assert by_category[("count", Split.TEST)] == 10
    assert by_category[("ambiguous", Split.TEST)] in (3, 4)
    built = [i.model_copy(update={"split": splits[i.id]}) for i in items]
    counts = split_counts(built)
    assert counts["test"]["count"] + counts["dev"]["count"] == 20


def test_paraphrase_groups_stay_on_one_side() -> None:
    items = [item(n, question=f"Question {n}?", group=f"g-{n // 3}") for n in range(1, 31)]
    for seed in range(20):
        splits = assign_splits(items, seed=seed, test_fraction=0.5)
        sides: dict[str, set[Split]] = {}
        for i in items:
            sides.setdefault(i.group_key, set()).add(splits[i.id])
        assert all(len(s) == 1 for s in sides.values())


def test_single_item_category_goes_to_test() -> None:
    items = [item(1)]
    assert assign_splits(items, seed=1, test_fraction=0.5) == {"sq-001": Split.TEST}


def test_near_duplicates_by_ratio_or_word_overlap() -> None:
    pairs = near_duplicates(
        [("a", "How many people were seen at the entrance today?")],
        [
            ("fs-1", "how many people were seen at the entrance today"),
            ("fs-2", "Today, at the entrance, how many people were seen?"),
            ("fs-3", "Which zone had the most loitering events last week?"),
        ],
        min_ratio=0.85,
        min_jaccard=0.8,
    )
    assert [(p.a, p.b) for p in pairs] == [("a", "fs-1"), ("a", "fs-2")]
    assert pairs[0].ratio == 1.0
    assert pairs[1].jaccard == 1.0


def test_leakage_into_the_fewshot_bank_and_undeclared_paraphrases() -> None:
    fewshots = [("fs-001", "How many people were seen at the entrance today?")]
    items = [
        item(1, question="How many people were seen at the entrance today?!"),
        item(2, question="How many trucks reached the loading dock yesterday?"),
        item(3, question="How many trucks reached the loading dock yesterday evening?"),
        item(4, question="How many cars left the lot?", group="g-cars"),
        item(5, question="How many cars left the lot today?", group="g-cars"),
    ]
    problems = leakage_problems(items, fewshots, min_ratio=0.85, min_jaccard=0.8)
    assert len(problems) == 2
    assert problems[0].startswith("sq-001 is a near duplicate of few-shot fs-001")
    assert problems[1].startswith("sq-002 and sq-003 are near duplicates in different groups")


def test_built_problems_find_drift_from_the_items_file() -> None:
    items = [item(1), item(2, question="Other question?")]
    splits = {"sq-001": Split.TEST, "sq-002": Split.DEV}
    expected = ResultSet(columns=["n"], rows=[[1]])
    good = [
        items[0].model_copy(update={"split": Split.TEST, "expected": expected}),
        items[1].model_copy(update={"split": Split.DEV, "expected": expected}),
    ]
    assert built_problems(items, good, splits) == []
    edited = [good[0].model_copy(update={"question": "Edited?"}), good[1]]
    assert built_problems(items, edited, splits) == [
        "sq-001: built entry differs from items file; rebuild"
    ]
    moved = [good[0].model_copy(update={"split": Split.DEV}), good[1]]
    assert built_problems(items, moved, splits) == [
        "sq-001: split dev but the seeded split says test"
    ]
    no_rows = [good[0].model_copy(update={"expected": None}), good[1]]
    assert built_problems(items, no_rows, splits) == [
        "sq-001: expected result missing or unexpected"
    ]
    assert built_problems(items, good[:1], splits) == [
        "built files differ from items: missing ['sq-002'], extra []"
    ]


def test_pinned_items_keep_their_side_and_new_items_fill_the_gap() -> None:
    items = [item(n, question=f"Question {n}?") for n in range(1, 11)]
    first = assign_splits(items, seed=31, test_fraction=0.5)
    # A changed seed would reshuffle everything; pins keep every earlier assignment.
    assert assign_splits(items, seed=99, test_fraction=0.5, pinned=first) == first
    more = [*items, *(item(n, question=f"Question {n}?") for n in range(11, 15))]
    grown = assign_splits(more, seed=31, test_fraction=0.5, pinned=first)
    assert {k: grown[k] for k in first} == first
    assert Counter(grown.values())[Split.TEST] == 7  # the new items balance the category


def test_a_new_paraphrase_joins_its_pinned_group() -> None:
    items = [item(1, question="Q one?", group="g-a"), item(2, question="Q two?")]
    pinned = {"sq-001": Split.DEV, "sq-002": Split.TEST}
    grown = [*items, item(3, question="Q three?", group="g-a")]
    assert assign_splits(grown, seed=1, test_fraction=0.5, pinned=pinned)["sq-003"] is Split.DEV
    with pytest.raises(ValueError, match="pinned to both splits"):
        assign_splits(grown, seed=1, test_fraction=0.5, pinned={**pinned, "sq-003": Split.TEST})


def test_pinned_splits_file_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "splits.json"
    assert load_pinned_splits(path) == {}
    write_pinned_splits(path, {"sq-002": Split.TEST, "sq-001": Split.DEV})
    assert load_pinned_splits(path) == {"sq-001": Split.DEV, "sq-002": Split.TEST}
    assert path.read_text().index("sq-001") < path.read_text().index("sq-002")
