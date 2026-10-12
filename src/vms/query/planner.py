"""The LLM planner: question -> validated `QueryPlan`, with one repair attempt.

1. Render the versioned planner prompt (`configs/prompts/planner.v1.txt`) with the schema
   summary, the cameras with their timezones and zones, the known classes, the current local
   time per timezone, the plan JSON schema and the static few-shot examples.
2. Ask the LLM (structured output where the provider supports it).
3. Parse the reply as JSON, validate it as a `QueryPlan` (structure and field combinations),
   then semantically (names exist, sane time range, limit within the cap).
4. On any error, send the errors back once and ask for a corrected plan (`repair_attempts`).
5. Return the plan, or `PlanningFailedError` with every error from the last attempt.

An abstaining plan is a valid outcome, not an error. The planner never sees SQL.
"""

from __future__ import annotations

import datetime as dt
import json
import re
from dataclasses import dataclass, field
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import ValidationError

from vms.common.clock import Clock
from vms.query.catalog import QueryCatalog
from vms.query.fewshots import FewShotExample
from vms.query.llm import ChatMessage, LlmClient, LlmError
from vms.query.plan import AbstainReason, Intent, QueryPlan, plan_json_schema
from vms.query.prompts import PromptTemplate
from vms.query.semantic import validate_plan
from vms.query.settings import PlannerSettings, SqlValidatorSettings

MAX_QUESTION_CHARS = 1000
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.DOTALL)

REPAIR_TEMPLATE = (
    "That QueryPlan is not valid:\n{errors}\n"
    "Return a corrected QueryPlan as a single JSON object, or abstain if the question cannot "
    "be expressed."
)


class PlannerUnavailableError(Exception):
    """The LLM could not be reached (after the client's retries)."""


class PlanningFailedError(Exception):
    """No valid plan after the repair attempts."""

    def __init__(self, errors: list[str], attempts: list[PlanAttempt]) -> None:
        super().__init__("; ".join(errors))
        self.errors = errors
        self.attempts = attempts


@dataclass(frozen=True)
class PlanAttempt:
    raw: str
    errors: list[str]
    tokens: int


@dataclass
class PlanResult:
    plan: QueryPlan
    attempts: list[PlanAttempt] = field(default_factory=list)

    @property
    def repaired(self) -> bool:
        return len(self.attempts) > 1

    @property
    def tokens(self) -> int:
        return sum(a.tokens for a in self.attempts)


ABSTAIN_MESSAGES = {
    AbstainReason.AMBIGUOUS: "The question can be read in several ways with different answers. "
    "Please say which camera, zone, object type or time you mean.",
    AbstainReason.UNANSWERABLE: "The system does not record what this question asks about "
    "(for example plates, faces, ages or names).",
    AbstainReason.UNSUPPORTED: "This question needs a query shape the planner does not support "
    "yet (for example comparisons of two periods, funnels or ratios).",
    AbstainReason.DISALLOWED: "This request is not allowed: the query service only reads "
    "analytics data.",
}
VISUAL_SEARCH_MESSAGE = (
    "Visual search (describing what things look like) needs CLIP, which is not available yet."
)


@dataclass(frozen=True)
class Abstention:
    reason: AbstainReason
    message: str


def abstention(plan: QueryPlan) -> Abstention | None:
    """Why the service will not run this plan, or None if it will.

    A plan with `visual_query` is valid but needs CLIP ranking, which is deferred, so it is
    answered as an `unsupported` abstention rather than silently dropping the visual part.
    """
    if plan.intent is Intent.ABSTAIN:
        assert plan.abstain_reason is not None
        return Abstention(plan.abstain_reason, ABSTAIN_MESSAGES[plan.abstain_reason])
    if plan.visual_query is not None:
        return Abstention(AbstainReason.UNSUPPORTED, VISUAL_SEARCH_MESSAGE)
    return None


def parse_plan(text: str) -> tuple[QueryPlan | None, list[str]]:
    """Parse model output into a plan, or the list of errors to send back."""
    match = _FENCE.match(text)
    body = match.group(1) if match else text.strip()
    try:
        data = json.loads(body)
    except json.JSONDecodeError as exc:
        return None, [f"output is not valid JSON ({exc.msg} at char {exc.pos})"]
    if not isinstance(data, dict):
        return None, ["output must be one JSON object"]
    try:
        return QueryPlan.model_validate(data), []
    except ValidationError as exc:
        return None, [_format_error(e) for e in exc.errors()]


def _format_error(error: Any) -> str:
    loc = ".".join(str(p) for p in error.get("loc", ()))
    msg = str(error.get("msg", "invalid"))
    msg = msg.removeprefix("Value error, ")
    return f"{loc}: {msg}" if loc else msg


def schema_summary(sql: SqlValidatorSettings) -> str:
    return "\n".join(f"- {table}: {', '.join(cols)}" for table, cols in sql.tables.items())


def cameras_summary(catalog: QueryCatalog) -> str:
    lines = []
    for cam in catalog.cameras:
        zones = ", ".join(z.name for z in catalog.zones_of(cam.camera_id)) or "none"
        lines.append(f"- {cam.name} ({cam.timezone}); zones: {zones}")
    return "\n".join(lines) or "- (no cameras)"


def local_now(catalog: QueryCatalog, now: dt.datetime) -> str:
    zones = sorted({c.timezone for c in catalog.cameras}) or ["UTC"]
    parts = []
    for name in zones:
        local = now.astimezone(ZoneInfo(name))
        parts.append(f"{local:%Y-%m-%d %H:%M} {local:%A} ({name})")
    return "; ".join(parts)


def examples_block(examples: list[FewShotExample]) -> str:
    return "\n".join(f"Q: {e.question}\nA: {e.plan.compact_json()}" for e in examples)


class Planner:
    def __init__(
        self,
        llm: LlmClient,
        prompt: PromptTemplate,
        examples: list[FewShotExample],
        *,
        planner: PlannerSettings,
        sql: SqlValidatorSettings,
        clock: Clock,
    ) -> None:
        self._llm = llm
        self._prompt = prompt
        self._examples = examples_block(examples)
        self._schema = plan_json_schema()
        self._schema_text = json.dumps(self._schema, separators=(",", ":"))
        self._settings = planner
        self._sql = sql
        self._summary = schema_summary(sql)
        self._clock = clock

    @property
    def llm_name(self) -> str:
        return self._llm.name

    def system_prompt(self, catalog: QueryCatalog, now: dt.datetime) -> str:
        return self._prompt.render(
            {
                "schema_summary": self._summary,
                "cameras": cameras_summary(catalog),
                "classes": ", ".join(catalog.classes),
                "now": local_now(catalog, now),
                "plan_schema": self._schema_text,
                "examples": self._examples,
            }
        )

    async def plan(self, question: str, catalog: QueryCatalog) -> PlanResult:
        now = self._clock.now()
        messages = [
            ChatMessage("system", self.system_prompt(catalog, now)),
            ChatMessage("user", question),
        ]
        attempts: list[PlanAttempt] = []
        for _ in range(self._settings.repair_attempts + 1):
            try:
                reply = await self._llm.complete(messages, self._schema)
            except LlmError as exc:
                raise PlannerUnavailableError(str(exc)) from exc
            plan, errors = parse_plan(reply.text)
            if plan is not None:
                errors = validate_plan(
                    plan, catalog, now, self._settings, max_limit=self._sql.max_limit
                )
            attempts.append(PlanAttempt(reply.text, errors, reply.tokens))
            if plan is not None and not errors:
                return PlanResult(plan, attempts)
            messages += [
                ChatMessage("assistant", reply.text),
                ChatMessage(
                    "user", REPAIR_TEMPLATE.format(errors="\n".join(f"- {e}" for e in errors))
                ),
            ]
        raise PlanningFailedError(attempts[-1].errors, attempts)
