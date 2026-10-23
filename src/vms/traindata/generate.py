"""Generate candidate (question, QueryPlan) pairs (task 8.2): two independent generations.

Generation A: one call per batch writes `questions_per_call` questions with plans. Its prompt
is the planner's own system prompt for one training state (schema, catalog, local time, plan
schema, few-shot bank) followed by the training-data instruction, so it shares the planner's
cached prefix. Generation B: the planner itself (service prompt, temperature 0, one repair) on
each question alone, without seeing A's plan. Candidates are appended to a JSONL file as they
arrive, so an interrupted run resumes; the cost so far is re-read from it.

Every call goes through a `Budget`: before a call, its worst case (the whole prompt at the
cache-miss price, `max_tokens` of output) is reserved; after it, the actual cost from the
usage (cache hits at the hit price) replaces the reservation. No call starts that could take
the total past the cap.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from vms.common.logs import get_logger
from vms.eval.harness import eval_planner
from vms.eval.settings import Price
from vms.query.llm import (
    ChatMessage,
    LlmClient,
    LlmEndpointSettings,
    LlmError,
    LlmReply,
    LlmTuning,
    OpenAICompatibleClient,
)
from vms.query.plan import plan_json_schema
from vms.query.planner import Planner, PlannerUnavailableError, PlanningFailedError
from vms.query.prompts import PromptTemplate
from vms.query.settings import QuerySettings
from vms.traindata.spec import TrainDataSpec
from vms.traindata.states import LoadedState

log = get_logger(__name__)

PROMPT_FIELDS = frozenset({"n", "focus", "style"})
PROMPT_TOKENS_ESTIMATE = 6000  # the planner prompt is ~4.7k tokens; reserved at the miss price


class BudgetExhaustedError(Exception):
    pass


@dataclass
class Budget:
    price: Price
    cap_usd: float
    spent_usd: float = 0.0
    reserved_usd: float = 0.0
    calls: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0

    def worst_case(self, max_tokens: int) -> float:
        return self.price.cost_usd(PROMPT_TOKENS_ESTIMATE, max_tokens)

    def reserve(self, max_tokens: int) -> float:
        amount = self.worst_case(max_tokens)
        if self.spent_usd + self.reserved_usd + amount > self.cap_usd:
            raise BudgetExhaustedError(f"${self.spent_usd:.3f} spent of ${self.cap_usd:.2f}")
        self.reserved_usd += amount
        return amount

    def settle(self, reserved: float, reply: LlmReply | None) -> None:
        self.reserved_usd -= reserved
        if reply is None:
            # A failed call may still be billed; count its worst case.
            self.spent_usd += reserved
            return
        self.calls += 1
        self.prompt_tokens += reply.prompt_tokens
        self.cached_tokens += reply.cached_prompt_tokens
        self.completion_tokens += reply.completion_tokens
        self.spent_usd += self.price.cost_usd(
            reply.prompt_tokens, reply.completion_tokens, reply.cached_prompt_tokens
        )

    def summary(self) -> dict[str, Any]:
        return {
            "spent_usd": round(self.spent_usd, 4),
            "cap_usd": self.cap_usd,
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "cached_prompt_tokens": self.cached_tokens,
            "completion_tokens": self.completion_tokens,
            "price": self.price.model_dump(),
        }


class BudgetedLlm:
    """Every call reserves its worst case first; the actual cost settles it."""

    def __init__(self, inner: LlmClient, budget: Budget, max_tokens: int) -> None:
        self._inner = inner
        self._budget = budget
        self._max_tokens = max_tokens

    @property
    def name(self) -> str:
        return self._inner.name

    async def complete(self, messages: Sequence[ChatMessage], schema: dict[str, Any]) -> LlmReply:
        reserved = self._budget.reserve(self._max_tokens)
        reply: LlmReply | None = None
        try:
            reply = await self._inner.complete(messages, schema)
            return reply
        finally:
            self._budget.settle(reserved, reply)

    async def aclose(self) -> None:
        await self._inner.aclose()


@dataclass
class Batch:
    call: int
    state: str
    focus: str
    style: str


def plan_batches(spec: TrainDataSpec, start: int, count: int) -> list[Batch]:
    """Seeded batch assignments; batch k is the same whichever run reaches it (resumable)."""
    out = []
    for k in range(start, start + count):
        rng = random.Random(f"{spec.seed}:{k}")
        out.append(
            Batch(
                call=k,
                state=spec.states[k % len(spec.states)].name,
                focus=rng.choice(spec.generator.focuses),
                style=rng.choice(spec.generator.styles),
            )
        )
    return out


def parse_pairs(text: str) -> tuple[list[dict[str, Any]], str | None]:
    """The `pairs` of a generation-A reply: dicts with a non-empty question and a plan."""
    try:
        data = json.loads(text)
    except ValueError as exc:
        return [], f"not JSON: {exc}"
    pairs = data.get("pairs") if isinstance(data, dict) else None
    if not isinstance(pairs, list):
        return [], "no pairs list"
    out = [
        {"question": p["question"].strip(), "plan": p.get("plan")}
        for p in pairs
        if isinstance(p, dict) and isinstance(p.get("question"), str) and p["question"].strip()
    ]
    return out, None


@dataclass
class Generator:
    spec: TrainDataSpec
    qs: QuerySettings
    states: dict[str, LoadedState]
    budget: Budget
    root: Path
    endpoint: LlmEndpointSettings
    out: Path
    template: PromptTemplate = field(init=False)
    written: int = 0

    def __post_init__(self) -> None:
        self.template = PromptTemplate.load(self.root / self.spec.generator.prompt, PROMPT_FIELDS)

    def _client(self, *, temperature: float, max_tokens: int, fmt: str) -> LlmClient:
        tuning: LlmTuning = self.qs.llm.model_copy(
            update={
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": fmt,
                "timeout_s": self.spec.generator.timeout_s,
            }
        )
        inner = OpenAICompatibleClient(self.endpoint, tuning)
        return BudgetedLlm(inner, self.budget, max_tokens)

    def planner(self, state: LoadedState) -> tuple[Planner, LlmClient]:
        client = self._client(
            temperature=self.qs.llm.temperature,
            max_tokens=self.qs.llm.max_tokens,
            fmt=self.qs.llm.response_format,
        )
        return eval_planner(client, self.qs, state.now, self.root), client

    async def batch(self, batch: Batch, limit: asyncio.Semaphore) -> list[dict[str, Any]]:
        g = self.spec.generator
        state = self.states[batch.state]
        planner, planner_client = self.planner(state)
        system = planner.system_prompt(state.catalog, state.now)
        user = self.template.render(
            {"n": g.questions_per_call, "focus": batch.focus, "style": batch.style}
        )
        writer = self._client(
            temperature=g.temperature, max_tokens=g.max_tokens, fmt=g.response_format
        )
        messages = [ChatMessage("system", system), ChatMessage("user", user)]
        try:
            reply = await writer.complete(messages, plan_json_schema())
        except LlmError as exc:
            await planner_client.aclose()
            return [{"call": batch.call, "state": batch.state, "error": str(exc)}]
        finally:
            await writer.aclose()
        pairs, error = parse_pairs(reply.text)
        if error:
            await planner_client.aclose()
            return [{"call": batch.call, "state": batch.state, "error": error}]

        async def plan_b(i: int, pair: dict[str, Any]) -> dict[str, Any]:
            row: dict[str, Any] = {
                "id": f"c{batch.call:05d}-{i:02d}",
                "call": batch.call,
                "state": batch.state,
                "focus": batch.focus,
                "style": batch.style,
                "question": pair["question"],
                "plan_a": pair["plan"],
            }
            async with limit:
                try:
                    result = await planner.plan(pair["question"], state.catalog)
                    row["plan_b"] = json.loads(result.plan.compact_json())
                    row["b_status"] = "planned"
                    row["b_attempts"] = len(result.attempts)
                except PlanningFailedError as exc:
                    row["plan_b"], row["b_status"] = None, "failed_validation"
                    row["b_errors"] = exc.errors[:5]
                except PlannerUnavailableError as exc:
                    row["plan_b"], row["b_status"] = None, "unavailable"
                    row["b_errors"] = [str(exc)[:300]]
                except BudgetExhaustedError:
                    row["plan_b"], row["b_status"] = None, "budget"
            return row

        try:
            return list(await asyncio.gather(*(plan_b(i, p) for i, p in enumerate(pairs))))
        finally:
            await planner_client.aclose()

    async def run(self, start_call: int) -> int:
        """Generate until the candidate target or the budget; returns the next call index."""
        spec = self.spec
        limit = asyncio.Semaphore(spec.planner_concurrency)
        calls = asyncio.Semaphore(spec.budget.call_concurrency)
        call = start_call
        stop = False

        async def one(batch: Batch) -> None:
            nonlocal stop
            async with calls:
                if stop:
                    return
                try:
                    rows = await self.batch(batch, limit)
                except BudgetExhaustedError:
                    stop = True
                    return
                self.append(rows)
                self.append([{"call": batch.call, "budget": self.budget.summary()}])
                if self.written >= spec.budget.max_candidates or any(
                    r.get("b_status") == "budget" for r in rows
                ):
                    stop = True

        while not stop:
            wave = plan_batches(spec, call, spec.budget.call_concurrency)
            await asyncio.gather(*(one(b) for b in wave))
            call += len(wave)
            log.info(
                "generation progress",
                extra={"calls": call, "candidates": self.written, **self.budget.summary()},
            )
        return call

    def append(self, rows: list[dict[str, Any]]) -> None:
        self.out.parent.mkdir(parents=True, exist_ok=True)
        with self.out.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                if "question" in row:
                    self.written += 1


def read_candidates(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
