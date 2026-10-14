"""The structured-query eval set: items, splits and leakage checks (task 3.1).

Authoring and building are separate. `eval/structured/items.v<N>.jsonl` is written by hand
(drafted by a model, reviewed by the user): each item has a question, its category, a gold
`QueryPlan` (an abstaining plan where the answer is to refuse or ask), an independent gold SQL
query where an answer exists, and whether row order matters. `python -m vms.eval build` then
executes both golds on the frozen snapshot, requires them to agree, stores the result as
`expected`, assigns the dev/test split and writes `dev.v<N>.jsonl` and `test.v<N>.jsonl`.

Gold SQL is written against the schema directly, not derived from the plan, so the agreement
check catches errors in either (and in the compiler). It binds one parameter, `%(now)s`, the
eval clock from the snapshot manifest, so relative periods give the same rows on every run.

Splits are assigned by paraphrase group (`group`, else the item's id), stratified by category,
with a fixed seed: a paraphrase never sits on both sides, so tuning on dev cannot see a test
question reworded. Leakage checks reject any eval question that is an exact or near duplicate
of a few-shot example (prompt material), and any near-duplicate pair that is not declared as
one group.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import random
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from vms.eval.compare import ResultSet
from vms.query.fewshots import normalize_question
from vms.query.plan import AbstainReason, Intent, QueryPlan


class Category(StrEnum):
    COUNT = "count"
    AGGREGATE = "aggregate"
    LIST = "list"
    TIME_WINDOW = "time_window"  # absolute ranges, named periods, camera timezones, DST
    JOIN = "join"  # across tracks, events and zones (events by class, tracks by zone, ...)
    TREND = "trend"
    TOP_K = "top_k"
    DWELL = "dwell"
    FRAMES = "frames"
    AMBIGUOUS = "ambiguous"
    UNANSWERABLE = "unanswerable"
    UNSUPPORTED = "unsupported"  # answerable with SQL, outside the plan schema
    ADVERSARIAL = "adversarial"  # must be refused


class Split(StrEnum):
    DEV = "dev"
    TEST = "test"


ABSTAIN_CATEGORIES: dict[Category, AbstainReason] = {
    Category.AMBIGUOUS: AbstainReason.AMBIGUOUS,
    Category.UNANSWERABLE: AbstainReason.UNANSWERABLE,
    Category.UNSUPPORTED: AbstainReason.UNSUPPORTED,
    Category.ADVERSARIAL: AbstainReason.DISALLOWED,
}
# Abstentions with no answer to compute: nothing to execute, no expected rows.
NO_ANSWER_CATEGORIES = frozenset({Category.AMBIGUOUS, Category.UNANSWERABLE, Category.ADVERSARIAL})


class EvalItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^sq-\d{3}$")
    category: Category
    group: str | None = Field(default=None, pattern=r"^g-[a-z0-9-]+$")  # paraphrase group
    question: str = Field(min_length=3)
    plan: QueryPlan
    expressible_by_plan: bool
    sql: str | None = None  # gold SQL; binds %(now)s
    ordered: bool = False  # row order is part of the answer
    tags: tuple[str, ...] = ()  # e.g. redteam:injection, tz, dst, paraphrase
    notes: str | None = None
    # Filled by the build, absent in items.v<N>.jsonl.
    expected: ResultSet | None = None
    split: Split | None = None

    @model_validator(mode="after")
    def _check(self) -> EvalItem:
        reason = ABSTAIN_CATEGORIES.get(self.category)
        if reason is not None and self.plan.abstain_reason is not reason:
            raise ValueError(f"category {self.category} needs a plan abstaining with {reason}")
        if reason is None and self.plan.intent is Intent.ABSTAIN:
            raise ValueError(f"category {self.category} must not abstain")
        if self.expressible_by_plan == (self.category is Category.UNSUPPORTED):
            raise ValueError("expressible_by_plan must be false exactly for unsupported items")
        if (self.sql is None) != (self.category in NO_ANSWER_CATEGORIES):
            raise ValueError("sql is required exactly when the question has an answer")
        if self.plan.visual_query is not None:
            raise ValueError("visual_query needs CLIP (deferred): not in this eval set")
        return self

    @property
    def group_key(self) -> str:
        return self.group or self.id

    @property
    def has_answer(self) -> bool:
        return self.sql is not None

    def authored(self) -> EvalItem:
        """The item as written in the items file (build outputs removed)."""
        return self.model_copy(update={"expected": None, "split": None})

    def to_json(self) -> str:
        data = self.model_dump(mode="json", exclude_none=True, exclude_defaults=True)
        data["plan"] = self.plan.compact()
        # Keep the flags explicit: a reader should not need to know the defaults.
        data["expressible_by_plan"] = self.expressible_by_plan
        data["ordered"] = self.ordered
        return json.dumps(data, ensure_ascii=False)


def load_items(path: Path) -> list[EvalItem]:
    items: list[EvalItem] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                items.append(EvalItem.model_validate(json.loads(line)))
            except ValueError as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    return items


def write_items(path: Path, items: Iterable[EvalItem]) -> str:
    """Write JSONL; returns the file's sha256."""
    data = "".join(item.to_json() + "\n" for item in items).encode("utf-8")
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- consistency -----------------------------------------------------------------------------


def structure_problems(items: Sequence[EvalItem]) -> list[str]:
    """Duplicate ids or questions, and groups that mix categories."""
    problems = []
    seen_ids: set[str] = set()
    seen_questions: dict[str, str] = {}
    for item in items:
        if item.id in seen_ids:
            problems.append(f"{item.id}: duplicate id")
        seen_ids.add(item.id)
        key = normalize_question(item.question)
        if key in seen_questions:
            problems.append(f"{item.id}: same question as {seen_questions[key]}")
        seen_questions.setdefault(key, item.id)
    categories: dict[str, set[Category]] = defaultdict(set)
    for item in items:
        categories[item.group_key].add(item.category)
    for group, cats in sorted(categories.items()):
        if len(cats) > 1:
            problems.append(f"group {group} mixes categories {sorted(c.value for c in cats)}")
    return problems


# --- leakage ---------------------------------------------------------------------------------


_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class NearDuplicate:
    a: str  # id or question
    b: str
    ratio: float  # difflib similarity of the normalized questions
    jaccard: float  # word-set overlap


def similarity(a: str, b: str) -> tuple[float, float]:
    na, nb = normalize_question(a), normalize_question(b)
    ratio = difflib.SequenceMatcher(None, na, nb, autojunk=False).ratio()
    wa, wb = set(_WORD.findall(na)), set(_WORD.findall(nb))
    jaccard = len(wa & wb) / len(wa | wb) if wa | wb else 1.0
    return ratio, jaccard


def near_duplicates(
    left: Sequence[tuple[str, str]],
    right: Sequence[tuple[str, str]],
    *,
    min_ratio: float,
    min_jaccard: float,
) -> list[NearDuplicate]:
    """Pairs of (label, question) whose questions are near duplicates by either measure.
    With `left is right`, each unordered pair is checked once."""
    same = left is right
    found = []
    for i, (la, qa) in enumerate(left):
        for j, (lb, qb) in enumerate(right):
            if same and j <= i:
                continue
            ratio, jaccard = similarity(qa, qb)
            if ratio >= min_ratio or jaccard >= min_jaccard:
                found.append(NearDuplicate(la, lb, round(ratio, 3), round(jaccard, 3)))
    return found


def leakage_problems(
    items: Sequence[EvalItem],
    fewshot_questions: Sequence[tuple[str, str]],
    *,
    min_ratio: float,
    min_jaccard: float,
) -> list[str]:
    """Eval questions too close to a few-shot example, and undeclared paraphrase pairs."""
    problems = []
    labelled = [(item.id, item.question) for item in items]
    for d in near_duplicates(
        labelled, fewshot_questions, min_ratio=min_ratio, min_jaccard=min_jaccard
    ):
        problems.append(
            f"{d.a} is a near duplicate of few-shot {d.b} (ratio {d.ratio}, jaccard {d.jaccard})"
        )
    groups = {item.id: item.group_key for item in items}
    for d in near_duplicates(labelled, labelled, min_ratio=min_ratio, min_jaccard=min_jaccard):
        if groups[d.a] != groups[d.b]:
            problems.append(
                f"{d.a} and {d.b} are near duplicates in different groups "
                f"(ratio {d.ratio}, jaccard {d.jaccard}); give them one group"
            )
    return problems


# --- splits ----------------------------------------------------------------------------------


def assign_splits(
    items: Sequence[EvalItem], *, seed: int, test_fraction: float
) -> dict[str, Split]:
    """item id -> split. Whole groups go to one side; within each category, groups are
    shuffled with a seeded stream and each goes to the side furthest below its target size,
    so every category is split close to `test_fraction` and the result is reproducible."""
    by_category: dict[Category, dict[str, list[str]]] = defaultdict(dict)
    for item in items:
        by_category[item.category].setdefault(item.group_key, []).append(item.id)
    splits: dict[str, Split] = {}
    for category in sorted(by_category):
        groups = by_category[category]
        keys = sorted(groups)
        random.Random(f"{seed}:{category}").shuffle(keys)
        total = sum(len(ids) for ids in groups.values())
        target = {Split.TEST: total * test_fraction, Split.DEV: total * (1 - test_fraction)}
        size = {Split.TEST: 0, Split.DEV: 0}
        for key in keys:
            # Deficit relative to target; ties go to test, so a 1-item category is tested.
            side = max((Split.TEST, Split.DEV), key=lambda s: target[s] - size[s])
            for item_id in groups[key]:
                splits[item_id] = side
            size[side] += len(groups[key])
    return splits


def split_counts(items: Iterable[EvalItem]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for item in items:
        if item.split is not None:
            counts[item.split.value][item.category.value] += 1
    return {s: dict(sorted(c.items())) for s, c in sorted(counts.items())}


class DatasetManifest(BaseModel):
    """Pins a built dataset: committed next to the split files. The test hash is also
    recorded in BENCHMARKS.md."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    version: int
    items: str
    items_sha256: str
    snapshot_manifest: str
    snapshot_files: dict[str, str]
    now: str  # the eval clock (UTC ISO)
    split_seed: int
    test_fraction: float
    dev_sha256: str
    test_sha256: str
    counts: dict[str, dict[str, int]]
    fewshots: str
    fewshots_sha256: str


def read_dataset_manifest(path: Path) -> DatasetManifest:
    return DatasetManifest.model_validate_json(path.read_text(encoding="utf-8"))


def write_dataset_manifest(path: Path, manifest: DatasetManifest) -> None:
    path.write_text(json.dumps(manifest.model_dump(), indent=2) + "\n", encoding="utf-8")


def built_problems(
    items: Sequence[EvalItem], built: Sequence[EvalItem], splits: dict[str, Split]
) -> list[str]:
    """Differences between the authored items and the built split files (other than the
    expected rows, which need the database to check)."""
    problems = []
    authored = {i.id: i for i in items}
    built_by_id = {i.id: i for i in built}
    if set(authored) != set(built_by_id):
        missing = sorted(set(authored) - set(built_by_id))
        extra = sorted(set(built_by_id) - set(authored))
        problems.append(f"built files differ from items: missing {missing}, extra {extra}")
    for item_id in sorted(set(authored) & set(built_by_id)):
        b = built_by_id[item_id]
        if b.authored() != authored[item_id]:
            problems.append(f"{item_id}: built entry differs from items file; rebuild")
        if b.split is not splits.get(item_id):
            problems.append(
                f"{item_id}: split {b.split} but the seeded split says {splits[item_id]}"
            )
        if (b.expected is None) == authored[item_id].has_answer:
            problems.append(f"{item_id}: expected result missing or unexpected")
    return problems
