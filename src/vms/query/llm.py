"""LLM clients behind one small interface, so the planner never knows which one it talks to.

- `OpenAICompatibleClient`: POST `{base_url}/chat/completions` (Ollama, vLLM, OpenAI, DeepSeek
  and most hosted APIs speak it). Structured output is configurable because providers differ:
  `json_schema` constrains decoding to the `QueryPlan` schema (Ollama, vLLM, OpenAI);
  `json_object` only guarantees some JSON object (DeepSeek); `none` relies on the prompt. Every
  call has a timeout; connection errors, timeouts, 429 and 5xx are retried a bounded number of
  times with full-jitter backoff.
- `CannedLlm`: no model at all. It answers questions it has a stored plan for and abstains on
  anything else. The demo and CI use it so they run without an LLM; its output is what a perfect
  planner would say for those questions, so it proves the pipeline, not planning quality.

Endpoint identity comes from `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY` (`.env`), like the
POSTGRES_* variables; behavior (format, timeouts, retries) comes from `configs/query.yaml`.
"""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import aiohttp
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from vms.query.fewshots import normalize_question
from vms.query.plan import QueryPlan

Role = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class ChatMessage:
    role: Role
    content: str


@dataclass(frozen=True)
class LlmReply:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = ""

    @property
    def tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LlmError(Exception):
    """The model could not be reached or returned something unusable (after retries)."""


class LlmClient(Protocol):
    @property
    def name(self) -> str:
        """Provider and model, for logs and query_log."""
        ...

    async def complete(
        self, messages: Sequence[ChatMessage], schema: dict[str, Any]
    ) -> LlmReply: ...

    async def aclose(self) -> None: ...


class LlmEndpointSettings(BaseSettings):
    """Which model to call. Empty `model` means no LLM is configured."""

    model_config = SettingsConfigDict(env_prefix="LLM_", env_file=".env", extra="ignore")

    base_url: str = "http://localhost:11434/v1"
    model: str = ""
    api_key: SecretStr | None = None

    @field_validator("api_key", mode="before")
    @classmethod
    def _empty_is_none(cls, value: object) -> object:
        return None if value == "" else value


class LlmTuning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # auto: the OpenAI-compatible client when LLM_MODEL is set, else the canned one.
    provider: Literal["auto", "openai", "canned"] = "auto"
    response_format: Literal["json_schema", "json_object", "none"] = "json_schema"
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=16)
    timeout_s: float = Field(default=30.0, gt=0)
    max_retries: int = Field(default=2, ge=0, le=5)
    retry_base_s: float = Field(default=0.5, gt=0)
    retry_max_s: float = Field(default=4.0, gt=0)
    # Question -> plan files for the canned client (JSONL rows with "question" and "plan").
    canned_plans: list[Path] = Field(default_factory=list)


# --- OpenAI-compatible -------------------------------------------------------------------------

# (status, parsed JSON body or None). Injected so tests need no sockets.
PostFn = Callable[[str, dict[str, Any], dict[str, str], float], Awaitable[tuple[int, Any]]]
SleepFn = Callable[[float], Awaitable[None]]

_RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504})


class _RetryableError(Exception):
    pass


class OpenAICompatibleClient:
    def __init__(
        self,
        endpoint: LlmEndpointSettings,
        tuning: LlmTuning,
        *,
        post: PostFn | None = None,
        sleep: SleepFn = asyncio.sleep,
        rng: random.Random | None = None,
    ) -> None:
        if not endpoint.model:
            raise ValueError("LLM_MODEL is empty: set it, or use the canned provider")
        self._endpoint = endpoint
        self._tuning = tuning
        self._url = endpoint.base_url.rstrip("/") + "/chat/completions"
        self._session: aiohttp.ClientSession | None = None
        self._post = post or self._aiohttp_post
        self._sleep = sleep
        self._rng = rng or random.Random()

    @property
    def name(self) -> str:
        return f"openai:{self._endpoint.model}"

    def request_body(
        self, messages: Sequence[ChatMessage], schema: dict[str, Any]
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self._endpoint.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self._tuning.temperature,
            "max_tokens": self._tuning.max_tokens,
            "stream": False,
        }
        if self._tuning.response_format == "json_schema":
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "QueryPlan", "schema": schema, "strict": False},
            }
        elif self._tuning.response_format == "json_object":
            body["response_format"] = {"type": "json_object"}
        return body

    async def complete(self, messages: Sequence[ChatMessage], schema: dict[str, Any]) -> LlmReply:
        body = self.request_body(messages, schema)
        headers = {"Content-Type": "application/json"}
        if self._endpoint.api_key is not None:
            headers["Authorization"] = f"Bearer {self._endpoint.api_key.get_secret_value()}"
        attempts = self._tuning.max_retries + 1
        for attempt in range(1, attempts + 1):
            try:
                return await self._once(body, headers)
            except _RetryableError as exc:
                if attempt == attempts:
                    raise LlmError(f"{self.name}: failed after {attempt} attempts: {exc}") from exc
                cap = min(self._tuning.retry_max_s, self._tuning.retry_base_s * 2 ** (attempt - 1))
                await self._sleep(self._rng.uniform(0, cap))  # full jitter
        raise AssertionError("unreachable")  # pragma: no cover

    async def _once(self, body: dict[str, Any], headers: dict[str, str]) -> LlmReply:
        try:
            status, data = await self._post(self._url, body, headers, self._tuning.timeout_s)
        except (TimeoutError, aiohttp.ClientError, OSError) as exc:
            raise _RetryableError(f"{type(exc).__name__}: {exc}") from exc
        if status in _RETRY_STATUSES:
            raise _RetryableError(f"HTTP {status}")
        if status != 200:
            raise LlmError(f"{self.name}: HTTP {status}: {_error_text(data)}")
        try:
            message = data["choices"][0]["message"]
            text = message.get("content") or ""
            usage = data.get("usage") or {}
            return LlmReply(
                text=str(text),
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                model=str(data.get("model") or self._endpoint.model),
            )
        except (KeyError, IndexError, TypeError, AttributeError, ValueError) as exc:
            raise LlmError(f"{self.name}: unexpected response shape: {exc!r}") from exc

    async def _aiohttp_post(
        self, url: str, body: dict[str, Any], headers: dict[str, str], timeout_s: float
    ) -> tuple[int, Any]:
        if self._session is None:
            # trust_env: honor HTTPS_PROXY and friends, as other HTTP clients here do.
            self._session = aiohttp.ClientSession(trust_env=True)
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with self._session.post(url, json=body, headers=headers, timeout=timeout) as resp:
            try:
                data = await resp.json(content_type=None)
            except (json.JSONDecodeError, aiohttp.ContentTypeError, ValueError):
                data = None
            return resp.status, data

    async def aclose(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None


def _error_text(data: Any) -> str:
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            return str(error.get("message", error))[:300]
        if error is not None:
            return str(error)[:300]
    return str(data)[:300]


# --- Canned ------------------------------------------------------------------------------------

UNKNOWN_QUESTION_PLAN = QueryPlan.model_validate(
    {"intent": "abstain", "abstain_reason": "unsupported"}
)
# Reply `model` when the canned client has no stored plan, so callers can tell "no LLM to ask"
# apart from a stored plan that abstains.
CANNED_NO_MATCH_MODEL = "canned:no-match"


class CannedLlm:
    """Answers from a fixed question -> plan table (exact match after normalization).

    The question is the first user message (the planner sends it verbatim). Repair turns get
    the same answer again, so a canned plan that fails validation fails twice, visibly.
    """

    def __init__(self, plans: dict[str, QueryPlan]) -> None:
        self._plans = {normalize_question(q): p for q, p in plans.items()}
        self.questions_as_written = list(plans)  # file order, original spelling

    @classmethod
    def from_files(cls, paths: Sequence[Path]) -> CannedLlm:
        plans: dict[str, QueryPlan] = {}
        for path in paths:
            with path.open(encoding="utf-8") as fh:
                for lineno, line in enumerate(fh, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        plans[str(row["question"])] = QueryPlan.model_validate(row["plan"])
                    except (ValueError, KeyError, TypeError) as exc:
                        raise ValueError(f"{path}:{lineno}: {exc}") from exc
        return cls(plans)

    @property
    def name(self) -> str:
        return "canned"

    @property
    def questions(self) -> list[str]:
        return sorted(self._plans)

    async def complete(self, messages: Sequence[ChatMessage], schema: dict[str, Any]) -> LlmReply:
        question = next((m.content for m in messages if m.role == "user"), "")
        plan = self._plans.get(normalize_question(question))
        if plan is None:
            return LlmReply(text=UNKNOWN_QUESTION_PLAN.compact_json(), model=CANNED_NO_MATCH_MODEL)
        return LlmReply(text=plan.compact_json(), model="canned")

    async def aclose(self) -> None:
        return None


def build_llm(endpoint: LlmEndpointSettings, tuning: LlmTuning, repo_root: Path) -> LlmClient:
    provider = tuning.provider
    if provider == "auto":
        provider = "openai" if endpoint.model else "canned"
    if provider == "openai":
        return OpenAICompatibleClient(endpoint, tuning)
    return CannedLlm.from_files([repo_root / p for p in tuning.canned_plans])
