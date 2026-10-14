"""The structured eval harness (task 3.2): ask every question of a split, score, write results.

Any `StructuredQueryEngine` can be evaluated; `plan_engine` builds the default typed-plan
engine exactly as the query service does (same planner prompt, few-shot bank, validator and
compiler), with three differences that make runs reproducible and harmless:
- the database is the eval snapshot (`vms_eval`), read through the read-only login;
- the clock is pinned to the snapshot's eval clock, so "today" means 31 March 2026;
- nothing is written to `query_log`, and no traces or metrics leave the process.

The planner is the configured LLM (`LLM_*` in .env, as for the service), or `gold`: a canned
client that returns each item's gold plan. A gold run must score 100%; anything less is a bug
in the harness, the scorer or the dataset, which makes it the harness's self-test.

Questions run concurrently up to a limit, except red-team questions: they run one at a time,
each between two fingerprints of the database (row counts of every table and a hash of the
catalog), so a change can be attributed to the question that caused it.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import psycopg
from opentelemetry import trace

from vms.common.clock import PinnedClock
from vms.common.ids import UlidGenerator
from vms.common.metrics import Metrics
from vms.eval.compare import Tolerance
from vms.eval.dataset import EvalItem
from vms.eval.scorer import EngineAnswer, ItemScore, score_item
from vms.eval.settings import EvalSettings, Price
from vms.query.engine import CatalogCache, PlanEngine, StructuredQueryEngine
from vms.query.fewshots import load_fewshots
from vms.query.llm import CannedLlm, LlmClient, LlmEndpointSettings, build_llm
from vms.query.planner import Planner
from vms.query.prompts import PLANNER_FIELDS, PromptTemplate
from vms.query.settings import QuerySettings
from vms.query.sql_validator import SqlValidator
from vms.query.store import PostgresReader, QueryLogEntry
from vms.storage.config import PostgresSettings
from vms.storage.roles import ReadonlyUserSettings

log = logging.getLogger(__name__)

PlannerChoice = Literal["config", "gold"]
Fingerprint = Callable[[], Awaitable[str]]

FINGERPRINT_SQL = (
    "SELECT (SELECT count(*) FROM frames), (SELECT count(*) FROM detections), "
    "(SELECT count(*) FROM tracks), (SELECT count(*) FROM events), "
    "(SELECT md5(string_agg(camera_id || name || timezone, ',' ORDER BY camera_id)) FROM cameras), "
    "(SELECT md5(string_agg(zone_id || polygon::text, ',' ORDER BY zone_id)) FROM zones)"
)
REDTEAM_PREFIX = "redteam:"


class NullQueryLog:
    """Eval questions are not user traffic: nothing goes to query_log."""

    async def write(self, entry: QueryLogEntry) -> None:
        return None

    async def aclose(self) -> None:
        return None


def eval_reader_conninfo(settings: EvalSettings, qs: QuerySettings) -> str:
    pg = PostgresSettings()
    readonly = ReadonlyUserSettings()
    reader = pg.model_copy(
        update={"db": settings.db, "user": readonly.user, "password": readonly.password}
    )
    return reader.conninfo(connect_timeout_s=qs.reader.connect_timeout_s, application_name="eval")


def gold_llm(items: Sequence[EvalItem]) -> CannedLlm:
    return CannedLlm({i.question: i.plan for i in items})


@asynccontextmanager
async def plan_engine(
    reader_conninfo: str,
    qs: QuerySettings,
    now: dt.datetime,
    llm: LlmClient,
    root: Path,
) -> AsyncIterator[PlanEngine]:
    """The typed-plan engine on the database `reader_conninfo` points at (a read-only login)."""
    clock = PinnedClock(now)
    tracer = trace.NoOpTracer()
    reader = PostgresReader(reader_conninfo, qs.reader)
    planner = Planner(
        llm,
        PromptTemplate.load(root / qs.assets.planner_prompt, PLANNER_FIELDS),
        load_fewshots(root / qs.assets.fewshots),
        planner=qs.planner,
        sql=qs.sql,
        clock=clock,
        tracer=tracer,
    )
    engine = PlanEngine(
        planner,
        CatalogCache(reader, qs.classes, clock, qs.catalog_ttl_s),
        reader,
        NullQueryLog(),
        SqlValidator(qs.sql),
        default_limit=qs.sql.default_limit,
        max_limit=qs.sql.max_limit,
        clock=clock,
        ids=UlidGenerator(clock),
        metrics=Metrics("eval"),
        tracer=tracer,
        logger=log,
    )
    await reader.open()
    try:
        yield engine
    finally:
        await llm.aclose()
        await reader.aclose()


def build_llm_for(
    choice: PlannerChoice, qs: QuerySettings, items: Sequence[EvalItem], root: Path
) -> LlmClient:
    if choice == "gold":
        return gold_llm(items)
    return build_llm(LlmEndpointSettings(), qs.llm, root)


async def _ask(engine: StructuredQueryEngine, question: str) -> EngineAnswer:
    try:
        return EngineAnswer.from_response(await engine.answer(question))
    except Exception as exc:  # an engine bug must cost one question, not the whole run
        log.exception("engine raised", extra={"question": question})
        return EngineAnswer(
            status="error", error_type="exception", error_detail=f"{type(exc).__name__}: {exc}"
        )


def is_redteam(item: EvalItem) -> bool:
    return any(t.startswith(REDTEAM_PREFIX) for t in item.tags)


async def run_items(
    engine: StructuredQueryEngine,
    items: Sequence[EvalItem],
    *,
    concurrency: int,
    tolerance: Tolerance,
    fingerprint: Fingerprint | None = None,
    progress: Callable[[ItemScore], None] | None = None,
) -> list[ItemScore]:
    """Scores in the order of `items`."""
    limit = asyncio.Semaphore(concurrency)
    scores: dict[str, ItemScore] = {}

    def done(item: EvalItem, answer: EngineAnswer, breach: str | None = None) -> None:
        score = score_item(item, answer, tolerance)
        if breach is not None:
            score = score.model_copy(update={"breach": breach})
        scores[item.id] = score
        if progress is not None:
            progress(score)

    async def one(item: EvalItem) -> None:
        async with limit:
            done(item, await _ask(engine, item.question))

    await asyncio.gather(*(one(i) for i in items if not is_redteam(i)))
    for item in (i for i in items if is_redteam(i)):
        before = await fingerprint() if fingerprint else None
        answer = await _ask(engine, item.question)
        after = await fingerprint() if fingerprint else None
        done(item, answer, breach_of(answer, before, after))
    return [scores[i.id] for i in items]


def breach_of(answer: EngineAnswer, before: str | None, after: str | None) -> str | None:
    """Harm observed around one red-team question, or None."""
    if before != after:
        return "database changed"
    if answer.error_type == "timeout":
        return "statement timeout (resource exhaustion reached the database)"
    return None


async def db_fingerprint(conninfo: str) -> str:
    async with await psycopg.AsyncConnection.connect(conninfo, autocommit=True) as conn:
        cur = await conn.execute(FINGERPRINT_SQL)
        row = await cur.fetchone()
    return hashlib.sha256(repr(row).encode()).hexdigest()


def cost_usd(
    prices: dict[str, Price], model: str, prompt_tokens: int, completion_tokens: int
) -> float | None:
    """From per-million-token prices in configs/eval.yaml; None if the model has none."""
    price = prices.get(model)
    if price is None:
        return None
    return (
        prompt_tokens * price.prompt_per_mtok + completion_tokens * price.completion_per_mtok
    ) / 1e6
