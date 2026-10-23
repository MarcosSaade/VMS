"""LLM clients: the OpenAI-compatible request/response contract, retries, and the canned client.

The HTTP call is injected, so no sockets are opened.
"""

from __future__ import annotations

import asyncio
import json
import random
from pathlib import Path
from typing import Any

import aiohttp
import pytest
from pydantic import SecretStr

from tests.unit.query.conftest import REPO_ROOT
from vms.query.llm import (
    CANNED_NO_MATCH_MODEL,
    CannedLlm,
    ChatMessage,
    LlmEndpointSettings,
    LlmError,
    LlmTuning,
    OpenAICompatibleClient,
    build_llm,
)
from vms.query.plan import QueryPlan

MESSAGES = [ChatMessage("system", "be a planner"), ChatMessage("user", "how many cars?")]
SCHEMA = {"type": "object"}
OK_BODY = {
    "model": "qwen",
    "choices": [{"message": {"role": "assistant", "content": '{"intent": "count"}'}}],
    "usage": {"prompt_tokens": 120, "completion_tokens": 8},
}


def endpoint(**kw: Any) -> LlmEndpointSettings:
    values: dict[str, Any] = {"base_url": "http://llm.test/v1/", "model": "qwen", **kw}
    return LlmEndpointSettings(_env_file=None, **values)  # type: ignore[call-arg]


class FakeHttp:
    def __init__(self, *responses: tuple[int, Any] | Exception) -> None:
        self.responses = list(responses)
        self.requests: list[tuple[str, dict[str, Any], dict[str, str], float]] = []
        self.sleeps: list[float] = []

    async def post(
        self, url: str, body: dict[str, Any], headers: dict[str, str], timeout_s: float
    ) -> tuple[int, Any]:
        self.requests.append((url, body, headers, timeout_s))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)


def client(http: FakeHttp, tuning: LlmTuning | None = None, **kw: Any) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        endpoint(**kw),
        tuning or LlmTuning(),
        post=http.post,
        sleep=http.sleep,
        rng=random.Random(0),
    )


def complete(c: OpenAICompatibleClient) -> Any:
    return asyncio.run(c.complete(MESSAGES, SCHEMA))


def test_request_uses_json_schema_and_returns_text_and_tokens() -> None:
    http = FakeHttp((200, OK_BODY))
    reply = complete(client(http))
    url, body, headers, timeout_s = http.requests[0]
    assert url == "http://llm.test/v1/chat/completions"
    assert body["model"] == "qwen"
    assert body["temperature"] == 0.0
    assert body["messages"][1] == {"role": "user", "content": "how many cars?"}
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == SCHEMA
    assert "Authorization" not in headers  # no key configured (Ollama)
    assert timeout_s == 30.0
    assert reply.text == '{"intent": "count"}'
    assert (reply.prompt_tokens, reply.completion_tokens, reply.tokens) == (120, 8, 128)


@pytest.mark.parametrize(
    ("fmt", "expected"),
    [("json_object", {"type": "json_object"}), ("none", None)],
)
def test_response_format_is_configurable(fmt: str, expected: dict[str, str] | None) -> None:
    http = FakeHttp((200, OK_BODY))
    complete(client(http, LlmTuning(response_format=fmt)))  # type: ignore[arg-type]
    assert http.requests[0][1].get("response_format") == expected


def test_extra_body_is_merged_into_the_request() -> None:
    http = FakeHttp((200, OK_BODY))
    extra = {"chat_template_kwargs": {"enable_thinking": False}}
    complete(client(http, LlmTuning(extra_body=extra)))
    body = http.requests[0][1]
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["response_format"]["type"] == "json_schema"  # the rest is unchanged


@pytest.mark.parametrize(
    "usage",
    [
        {"prompt_tokens": 120, "completion_tokens": 8, "prompt_cache_hit_tokens": 100},
        {
            "prompt_tokens": 120,
            "completion_tokens": 8,
            "prompt_tokens_details": {"cached_tokens": 100},
        },
    ],
)
def test_cached_prompt_tokens_are_read_from_either_usage_shape(usage: dict[str, Any]) -> None:
    http = FakeHttp((200, {**OK_BODY, "usage": usage}))
    assert complete(client(http)).cached_prompt_tokens == 100


def test_api_key_is_sent_as_bearer() -> None:
    http = FakeHttp((200, OK_BODY))
    complete(client(http, api_key=SecretStr("sk-test")))
    assert http.requests[0][2]["Authorization"] == "Bearer sk-test"


def test_retries_transient_failures_with_bounded_jittered_backoff() -> None:
    http = FakeHttp(
        (503, None), aiohttp.ClientConnectionError("refused"), TimeoutError(), (200, OK_BODY)
    )
    tuning = LlmTuning(max_retries=3, retry_base_s=0.5, retry_max_s=1.0)
    reply = complete(client(http, tuning))
    assert reply.text
    assert len(http.requests) == 4
    caps = [0.5, 1.0, 1.0]  # base * 2^n, capped at retry_max_s
    assert len(http.sleeps) == 3
    assert all(0 <= s <= cap for s, cap in zip(http.sleeps, caps, strict=True))


def test_gives_up_after_max_retries() -> None:
    http = FakeHttp((429, None), (429, None), (429, None))
    with pytest.raises(LlmError, match="after 3 attempts"):
        complete(client(http, LlmTuning(max_retries=2)))


def test_client_errors_are_not_retried() -> None:
    http = FakeHttp((400, {"error": {"message": "bad response_format"}}))
    with pytest.raises(LlmError, match="bad response_format"):
        complete(client(http))
    assert len(http.requests) == 1


@pytest.mark.parametrize("body", [None, {"choices": []}, {"choices": [{}]}, "text"])
def test_malformed_response_is_an_error(body: Any) -> None:
    with pytest.raises(LlmError, match="unexpected response"):
        complete(client(FakeHttp((200, body))))


def test_model_is_required() -> None:
    with pytest.raises(ValueError, match="LLM_MODEL"):
        OpenAICompatibleClient(endpoint(model=""), LlmTuning())


# --- canned ----------------------------------------------------------------------------------


def test_canned_answers_known_questions_and_abstains_otherwise(tmp_path: Path) -> None:
    plan = {"intent": "count", "target": "tracks"}
    path = tmp_path / "plans.jsonl"
    path.write_text(json.dumps({"question": "How many cars?", "plan": plan}) + "\n")
    llm = CannedLlm.from_files([path])
    known = asyncio.run(llm.complete([ChatMessage("user", "how many  CARS")], SCHEMA))
    assert QueryPlan.model_validate_json(known.text) == QueryPlan.model_validate(plan)
    unknown = asyncio.run(llm.complete([ChatMessage("user", "something else")], SCHEMA))
    assert json.loads(unknown.text) == {"intent": "abstain", "abstain_reason": "unsupported"}
    assert (known.model, unknown.model) == ("canned", CANNED_NO_MATCH_MODEL)


def test_canned_file_errors_name_the_line(tmp_path: Path) -> None:
    path = tmp_path / "plans.jsonl"
    path.write_text('{"question": "q", "plan": {"intent": "count"}}\n')
    with pytest.raises(ValueError, match=r"plans\.jsonl:1"):
        CannedLlm.from_files([path])


def test_shipped_canned_plans_load() -> None:
    llm = CannedLlm.from_files([REPO_ROOT / "configs/demo/canned_plans.v1.jsonl"])
    assert len(llm.questions) >= 8


def test_build_llm_picks_provider() -> None:
    tuning = LlmTuning(canned_plans=[Path("configs/demo/canned_plans.v1.jsonl")])
    assert build_llm(endpoint(model=""), tuning, REPO_ROOT).name == "canned"
    assert build_llm(endpoint(model="qwen2.5"), tuning, REPO_ROOT).name == "openai:qwen2.5"
    forced = tuning.model_copy(update={"provider": "canned"})
    assert build_llm(endpoint(model="qwen2.5"), forced, REPO_ROOT).name == "canned"
