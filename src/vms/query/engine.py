"""Structured query engines. The typed-plan engine is the only one (D11).

`StructuredQueryEngine` is the seam for comparing engines on one eval set (task 3.x, D11): a
question in, a `QueryResponse` out. `PlanEngine` runs the pipeline:

    catalog -> planner (LLM, validation, one repair) -> abstain? -> compile (parameterized SQL)
    -> sqlglot validator (regenerated SQL) -> read-only execution (timeout, row cap)
    -> templated answer -> response; every question is written to query_log

Each failure has its own error type, so the user (and the eval taxonomy) can tell them apart:

    planner_unavailable  the LLM could not be reached               HTTP 503
    invalid_plan         no valid plan after the repair             HTTP 422
    compile_error        a valid plan the compiler cannot handle    HTTP 500 (a bug)
    sql_rejected         the validator refused the compiled SQL     HTTP 500 (a bug, but blocked)
    timeout              the statement timeout cancelled the query  HTTP 504
    execution_error      any other database error                   HTTP 500

An abstention is a successful answer (`status: abstained`) with a reason and a message.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any, Literal, Protocol, TypeVar

from opentelemetry import trace
from pydantic import BaseModel, Field

from vms.common.clock import Clock
from vms.common.ids import UlidGenerator
from vms.common.metrics import Metrics
from vms.query.answer import render_answer
from vms.query.catalog import QueryCatalog, catalog_from_rows
from vms.query.compiler import CompileError, PostgresCompiler
from vms.query.plan import AbstainReason, ResultKind
from vms.query.planner import (
    PlanAttempt,
    Planner,
    PlannerUnavailableError,
    PlanningFailedError,
    PlanResult,
    abstention,
)
from vms.query.sql_validator import SqlValidationError, SqlValidator
from vms.query.store import (
    ExecutionError,
    ExecutionTimeoutError,
    QueryLog,
    QueryLogEntry,
    Reader,
    jsonable,
)

T = TypeVar("T")

STAGE_BUCKETS_S = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)


def frame_image_path(frame_id: str) -> str:
    return f"/v1/frames/{frame_id}/image"


class QueryStatus(StrEnum):
    OK = "ok"
    ABSTAINED = "abstained"
    ERROR = "error"


class ErrorType(StrEnum):
    PLANNER_UNAVAILABLE = "planner_unavailable"
    INVALID_PLAN = "invalid_plan"
    COMPILE_ERROR = "compile_error"
    SQL_REJECTED = "sql_rejected"
    TIMEOUT = "timeout"
    EXECUTION_ERROR = "execution_error"


HTTP_STATUS = {
    ErrorType.PLANNER_UNAVAILABLE: 503,
    ErrorType.INVALID_PLAN: 422,
    ErrorType.COMPILE_ERROR: 500,
    ErrorType.SQL_REJECTED: 500,
    ErrorType.TIMEOUT: 504,
    ErrorType.EXECUTION_ERROR: 500,
}


class QueryError(BaseModel):
    type: ErrorType
    detail: str
    errors: list[str] = Field(default_factory=list)  # plan validation errors, if any
    # For invalid_plan: whether the last reply was not a schema-valid plan ("schema") or a
    # valid plan that failed the checks against the catalog and policy ("semantic").
    plan_check: Literal["schema", "semantic"] | None = None


class QueryResponse(BaseModel):
    query_id: str
    question: str
    status: QueryStatus
    answer: str
    result_kind: ResultKind | None = None
    columns: list[str] = Field(default_factory=list)
    rows: list[list[Any]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    plan: dict[str, Any] | None = None
    compiled_sql: str | None = None  # the validated SQL that ran, with %(pN)s placeholders
    params: dict[str, Any] = Field(default_factory=dict)
    abstain_reason: AbstainReason | None = None
    error: QueryError | None = None
    llm: str
    repaired: bool = False
    tokens: int = 0
    prompt_tokens: int = 0  # tokens apart: prompt and completion are priced differently
    completion_tokens: int = 0
    timings_ms: dict[str, float] = Field(default_factory=dict)

    @property
    def http_status(self) -> int:
        return 200 if self.error is None else HTTP_STATUS[self.error.type]


class StructuredQueryEngine(Protocol):
    name: str

    async def answer(self, question: str) -> QueryResponse: ...


class CatalogCache:
    """Cameras and zones from the database, refreshed at most every `ttl_s`."""

    def __init__(self, reader: Reader, classes: list[str], clock: Clock, ttl_s: float) -> None:
        self._reader = reader
        self._classes = classes
        self._clock = clock
        self._ttl_s = ttl_s
        self._catalog: QueryCatalog | None = None
        self._loaded_at = 0.0

    async def get(self) -> QueryCatalog:
        now = self._clock.monotonic()
        if self._catalog is None or now - self._loaded_at >= self._ttl_s:
            cameras, zones = await self._reader.catalog_rows()
            self._catalog = catalog_from_rows(cameras, zones, self._classes)
            self._loaded_at = now
        return self._catalog


class PlanEngine:
    name = "plan"

    def __init__(
        self,
        planner: Planner,
        catalog: CatalogCache,
        reader: Reader,
        query_log: QueryLog,
        validator: SqlValidator,
        *,
        default_limit: int,
        max_limit: int,
        clock: Clock,
        ids: UlidGenerator,
        metrics: Metrics,
        tracer: trace.Tracer,
        logger: logging.Logger,
    ) -> None:
        self._planner = planner
        self._catalog = catalog
        self._reader = reader
        self._log = query_log
        self._validator = validator
        self._default_limit = default_limit
        self._max_limit = max_limit
        self._clock = clock
        self._ids = ids
        self._tracer = tracer
        self._logger = logger
        self._requests = metrics.counter(
            "requests", "total", "Structured queries by outcome", ["status", "error_type"]
        )
        self._stages = metrics.histogram(
            "stage_duration",
            "seconds",
            "Query pipeline stage latency",
            ["stage"],
            STAGE_BUCKETS_S,
        )
        # Prompt and completion apart: they are priced differently, and the prompt (~4.5k
        # tokens, mostly few-shots; E-017) is what prefix caching would save.
        self._tokens = metrics.counter(
            "llm_tokens", "total", "LLM tokens used by the planner", ["kind"]
        )
        self._repairs = metrics.counter("plan_repairs", "total", "Plans that needed a repair")
        self._log_failures = metrics.counter(
            "query_log_failures", "total", "query_log writes that failed"
        )

    async def _stage(self, name: str, timings: dict[str, float], work: Awaitable[T]) -> T:
        t0 = self._clock.monotonic()
        with self._tracer.start_as_current_span(f"query.{name}"):
            try:
                return await work
            finally:
                elapsed = self._clock.monotonic() - t0
                timings[name] = round(elapsed * 1000, 3)
                self._stages.labels(stage=name).observe(elapsed)

    def _timed(self, name: str, timings: dict[str, float], work: Callable[[], T]) -> T:
        """`_stage` for synchronous, CPU-only steps (compile, validate: well under 1 ms)."""
        t0 = self._clock.monotonic()
        with self._tracer.start_as_current_span(f"query.{name}"):
            try:
                return work()
            finally:
                elapsed = self._clock.monotonic() - t0
                timings[name] = round(elapsed * 1000, 3)
                self._stages.labels(stage=name).observe(elapsed)

    async def answer(self, question: str) -> QueryResponse:
        with self._tracer.start_as_current_span("query") as span:
            t0 = self._clock.monotonic()
            started = self._clock.now()
            query_id = self._ids.new()
            span.set_attribute("vms.query_id", query_id)
            timings: dict[str, float] = {}
            response = await self._run(query_id, question, timings)
            response.timings_ms = {
                **timings,
                "total": round((self._clock.monotonic() - t0) * 1000, 3),
            }
            error_type = response.error.type.value if response.error else ""
            span.set_attribute("vms.query_status", response.status.value)
            self._requests.labels(status=response.status.value, error_type=error_type).inc()
            await self._write_log(response, started)
            return response

    async def _run(self, query_id: str, question: str, timings: dict[str, float]) -> QueryResponse:
        base: dict[str, Any] = {
            "query_id": query_id,
            "question": question,
            "llm": self._planner.llm_name,
        }

        def failed(kind: ErrorType, detail: str, **extra: Any) -> QueryResponse:
            errors = extra.pop("errors", [])
            plan_check = extra.pop("plan_check", None)
            return QueryResponse(
                **base,
                **extra,
                status=QueryStatus.ERROR,
                answer=_ERROR_ANSWERS[kind],
                error=QueryError(type=kind, detail=detail, errors=errors, plan_check=plan_check),
            )

        try:
            catalog = await self._stage("catalog", timings, self._catalog.get())
        except ExecutionError as exc:
            return failed(ErrorType.EXECUTION_ERROR, f"catalog: {exc}")

        try:
            result: PlanResult = await self._stage(
                "plan", timings, self._planner.plan(question, catalog)
            )
        except PlannerUnavailableError as exc:
            return failed(ErrorType.PLANNER_UNAVAILABLE, str(exc))
        except PlanningFailedError as exc:
            tokens = sum(a.tokens for a in exc.attempts)
            self._count_tokens(exc.attempts)
            self._repairs.inc(1 if len(exc.attempts) > 1 else 0)
            last = exc.attempts[-1] if exc.attempts else None
            return failed(
                ErrorType.INVALID_PLAN,
                "the planner did not produce a valid plan",
                errors=exc.errors,
                plan_check="semantic" if last is not None and last.parsed else "schema",
                tokens=tokens,
                prompt_tokens=sum(a.prompt_tokens for a in exc.attempts),
                completion_tokens=sum(a.completion_tokens for a in exc.attempts),
                repaired=len(exc.attempts) > 1,
            )
        self._count_tokens(result.attempts)
        if result.repaired:
            self._repairs.inc()
        plan = result.plan
        base.update(
            plan=plan.compact(),
            tokens=result.tokens,
            prompt_tokens=sum(a.prompt_tokens for a in result.attempts),
            completion_tokens=sum(a.completion_tokens for a in result.attempts),
            repaired=result.repaired,
        )

        refusal = abstention(plan, result.model)
        if refusal is not None:
            return QueryResponse(
                **base,
                status=QueryStatus.ABSTAINED,
                answer=refusal.message,
                result_kind=ResultKind.ABSTENTION,
                abstain_reason=refusal.reason,
            )

        compiler = PostgresCompiler(
            catalog, default_limit=self._default_limit, max_limit=self._max_limit
        )
        try:
            compiled = self._timed(
                "compile", timings, lambda: compiler.compile(plan, self._clock.now())
            )
        except CompileError as exc:
            return failed(ErrorType.COMPILE_ERROR, str(exc))
        base["params"] = {k: jsonable(v) for k, v in compiled.params.items()}
        base["result_kind"] = compiled.result_kind

        try:
            checked = self._timed(
                "validate", timings, lambda: self._validator.validate(compiled.sql)
            )
        except SqlValidationError as exc:
            self._logger.error(
                "compiled SQL rejected by the validator",
                extra={"query_id": query_id, "code": exc.code.value, "sql": compiled.sql},
            )
            return failed(ErrorType.SQL_REJECTED, str(exc), compiled_sql=compiled.sql)
        base["compiled_sql"] = checked.sql

        try:
            rows = await self._stage(
                "execute",
                timings,
                self._reader.execute(checked.sql, compiled.params, checked.limit),
            )
        except ExecutionTimeoutError as exc:
            return failed(ErrorType.TIMEOUT, str(exc))
        except ExecutionError as exc:
            return failed(ErrorType.EXECUTION_ERROR, str(exc))

        # Hitting a limit the user did not ask for means there may be more rows.
        truncated = rows.truncated or (
            plan.limit is None
            and compiled.result_kind is not ResultKind.SCALAR
            and len(rows.rows) == checked.limit
        )
        columns, out = rows.columns, [[jsonable(v) for v in row] for row in rows.rows]
        if compiled.result_kind is ResultKind.FRAMES:
            # Images are served by this service (GET /v1/frames/{id}/image), so the browser
            # never needs object-store credentials or a route to the store.
            at = columns.index("frame_id")
            columns = [*columns, "image_url"]
            out = [[*row, frame_image_path(str(row[at]))] for row in out]
        return QueryResponse(
            **base,
            status=QueryStatus.OK,
            answer=render_answer(
                plan, compiled.result_kind, rows.columns, rows.rows, truncated=truncated
            ),
            columns=columns,
            rows=out,
            row_count=len(rows.rows),
            truncated=truncated,
        )

    def _count_tokens(self, attempts: list[PlanAttempt]) -> None:
        self._tokens.labels(kind="prompt").inc(sum(a.prompt_tokens for a in attempts))
        self._tokens.labels(kind="completion").inc(sum(a.completion_tokens for a in attempts))

    async def _write_log(self, response: QueryResponse, started: Any) -> None:
        entry = QueryLogEntry(
            query_id=response.query_id,
            ts=started,
            question=response.question,
            query_plan=response.plan,
            compiled_sql=response.compiled_sql,
            latency_ms=response.timings_ms.get("total", 0.0),
            tokens=response.tokens,
            success=response.error is None,
            error_type=response.error.type.value if response.error else None,
            timings=response.timings_ms,
            row_count=response.row_count if response.status is QueryStatus.OK else None,
            llm=response.llm,
        )
        try:
            await self._log.write(entry)
        except Exception as exc:  # never fail an answer because the log is down
            self._log_failures.inc()
            self._logger.warning(
                "query_log write failed",
                extra={"query_id": response.query_id, "error": f"{type(exc).__name__}: {exc}"},
            )


_ERROR_ANSWERS = {
    ErrorType.PLANNER_UNAVAILABLE: "The question planner (LLM) is not reachable right now.",
    ErrorType.INVALID_PLAN: "The question could not be turned into a valid query. "
    "Try naming the camera, zone, object type or time.",
    ErrorType.COMPILE_ERROR: "Internal error: the plan could not be compiled.",
    ErrorType.SQL_REJECTED: "Internal error: the generated SQL was blocked by the safety check.",
    ErrorType.TIMEOUT: "The query took too long and was cancelled. Try a shorter time range.",
    ErrorType.EXECUTION_ERROR: "The database could not run the query.",
}
