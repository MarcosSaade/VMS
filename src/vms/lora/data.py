"""Training examples for the planner adapters (task 8.3, D42).

Each pair of the 8.2 dataset becomes one chat example: the planner's system prompt rendered
exactly as the planner renders it (the adapters' prompt template, the pair's training state
catalog and clock), the question as the user turn, and the plan as the target, in the compact
JSON the few-shot bank uses. The split is seeded: a held-out validation slice for eval loss,
then one shuffled order of the rest, so every data-size run trains on a prefix of it (500 is
inside 1000 is inside all).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vms.eval.harness import eval_planner
from vms.query.catalog import QueryCatalog, catalog_from_rows
from vms.query.llm import LlmReply
from vms.query.plan import QueryPlan
from vms.query.settings import QuerySettings


@dataclass(frozen=True)
class Example:
    id: str
    state: str
    system: str
    question: str
    completion: str

    def record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "state": self.state,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.question},
            ],
            "completion": self.completion,
        }


class _NoLlm:
    name = "none"

    async def complete(self, messages: Any, schema: Any) -> LlmReply:
        raise AssertionError("prompt-only planner")

    async def aclose(self) -> None:
        return None


def load_catalogs(path: Path) -> dict[str, tuple[QueryCatalog, dt.datetime]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        name: (
            catalog_from_rows(c["cameras"], c["zones"], c["classes"]),
            dt.datetime.fromisoformat(c["now"]),
        )
        for name, c in data.items()
    }


def with_prompt(qs: QuerySettings, prompt: Path) -> QuerySettings:
    assets = qs.assets.model_copy(update={"planner_prompt": prompt})
    return qs.model_copy(update={"assets": assets})


def render_examples(
    pairs: Iterable[dict[str, Any]],
    catalogs: dict[str, tuple[QueryCatalog, dt.datetime]],
    qs: QuerySettings,
    root: Path,
) -> list[Example]:
    """`qs` carries the prompt template (`with_prompt`); one system prompt per state."""
    systems: dict[str, str] = {}
    out = []
    for pair in pairs:
        state = pair["state"]
        catalog, now = catalogs[state]
        if dt.datetime.fromisoformat(pair["now"]) != now:
            raise ValueError(f"{pair['id']}: clock {pair['now']} != state {state}'s {now}")
        if state not in systems:
            systems[state] = eval_planner(_NoLlm(), qs, now, root).system_prompt(catalog, now)
        plan = QueryPlan.model_validate(pair["plan"])
        out.append(
            Example(pair["id"], state, systems[state], pair["question"], plan.compact_json())
        )
    return out


def split_examples(
    examples: Sequence[Example], val_fraction: float, seed: int
) -> tuple[list[Example], list[Example]]:
    """(train in its seeded order, validation)."""
    order = sorted(examples, key=lambda e: e.id)
    random.Random(seed).shuffle(order)
    n_val = max(1, round(len(order) * val_fraction))
    return order[n_val:], order[:n_val]


def write_jsonl(path: Path, examples: Iterable[Example]) -> str:
    """Write the examples; returns the file's sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(e.record(), ensure_ascii=False) + "\n" for e in examples)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_pairs(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
