"""Fakes for query-service unit tests: no network, no real LLM, no database."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from vms.query.llm import ChatMessage, LlmError, LlmReply


class ScriptedLlm:
    """Returns the scripted replies in order and records every request.

    A reply may be a dict (sent as JSON), a string (sent as is) or an exception (raised).
    """

    def __init__(self, *replies: dict[str, Any] | str | Exception, tokens: int = 10) -> None:
        self._replies = list(replies)
        self._tokens = tokens
        self.calls: list[list[ChatMessage]] = []
        self.schemas: list[dict[str, Any]] = []
        self.closed = False

    @property
    def name(self) -> str:
        return "scripted"

    async def complete(self, messages: Sequence[ChatMessage], schema: dict[str, Any]) -> LlmReply:
        self.calls.append(list(messages))
        self.schemas.append(schema)
        if not self._replies:
            raise LlmError("script exhausted")
        reply = self._replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        text = reply if isinstance(reply, str) else json.dumps(reply)
        return LlmReply(text=text, prompt_tokens=self._tokens, completion_tokens=self._tokens)

    async def aclose(self) -> None:
        self.closed = True
