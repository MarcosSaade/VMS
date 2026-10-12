"""Fakes for query-service unit tests: no network, no real LLM, no database."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from vms.query.llm import ChatMessage, LlmError, LlmReply
from vms.query.store import QueryLogEntry, QueryRows


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


class FakeReader:
    """Read-only executor fake: scripted results per call, records the SQL it was given."""

    def __init__(
        self,
        cameras: list[tuple[Any, ...]],
        zones: list[tuple[Any, ...]],
        results: list[QueryRows | Exception] | None = None,
    ) -> None:
        self.cameras = cameras
        self.zones = zones
        self.results = list(results or [])
        self.executed: list[tuple[str, dict[str, Any], int]] = []
        self.fetched: list[tuple[str, dict[str, Any] | None]] = []
        self.fixed: dict[str, QueryRows] = {}
        self.catalog_loads = 0
        self.is_ready = True

    async def execute(self, sql: str, params: dict[str, Any], max_rows: int) -> QueryRows:
        self.executed.append((sql, params, max_rows))
        result = self.results.pop(0) if self.results else QueryRows(["n"], [(0,)], False)
        if isinstance(result, Exception):
            raise result
        return result

    async def fetch(self, sql: str, params: dict[str, Any] | None = None) -> QueryRows:
        self.fetched.append((sql, params))
        for marker, rows in self.fixed.items():
            if marker in sql:
                return rows
        return QueryRows([], [], False)

    async def catalog_rows(self) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
        self.catalog_loads += 1
        return self.cameras, self.zones

    async def ready(self) -> bool:
        return self.is_ready

    async def aclose(self) -> None:
        return None


class FakeQueryLog:
    def __init__(self, fail: bool = False) -> None:
        self.entries: list[QueryLogEntry] = []
        self.fail = fail

    async def write(self, entry: QueryLogEntry) -> None:
        if self.fail:
            raise OSError("log database down")
        self.entries.append(entry)

    async def aclose(self) -> None:
        return None


class FakeFrames:
    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects = objects or {}

    async def get(self, key: str) -> bytes:
        if key not in self.objects:
            raise KeyError(key)
        return self.objects[key]

    async def ready(self) -> bool:
        return True
