"""Record LLM replies on one machine and replay them on another (D40).

The structured eval needs the eval database (the compose stack); a Runpod pod has no Docker
(D37). So the pod runs only the planner, which needs no database, against the served model, and
records every reply under a key of the exact messages sent; the laptop then runs the unchanged
eval harness with `ReplayLlm`. The planner is deterministic given its replies, so a replayed run
asks the same messages in the same order and scores exactly what the served model produced. A
message that was never recorded (a different prompt, catalog or clock) is a replay miss: an
`LlmError`, counted by the harness like any planner failure and reported, never a silent pass.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

from vms.query.llm import ChatMessage, LlmClient, LlmError, LlmReply


def messages_key(messages: Sequence[ChatMessage]) -> str:
    """sha256 of the messages as sent (role and content, in order)."""
    payload = json.dumps([[m.role, m.content] for m in messages], ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Recorded:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    latency_ms: float


class RecordingLlm:
    """Passes calls through and keeps each reply under its messages' key."""

    def __init__(self, inner: LlmClient, clock: Any = time.perf_counter) -> None:
        self._inner = inner
        self._clock = clock
        self.records: dict[str, Recorded] = {}

    @property
    def name(self) -> str:
        return self._inner.name

    async def complete(self, messages: Sequence[ChatMessage], schema: dict[str, Any]) -> LlmReply:
        t0 = self._clock()
        reply = await self._inner.complete(messages, schema)
        self.records[messages_key(messages)] = Recorded(
            reply.text,
            reply.prompt_tokens,
            reply.completion_tokens,
            reply.model,
            (self._clock() - t0) * 1000.0,
        )
        return reply

    def dump(self) -> dict[str, dict[str, Any]]:
        return {k: asdict(v) for k, v in self.records.items()}

    async def aclose(self) -> None:
        await self._inner.aclose()


class ReplayLlm:
    """Answers from recorded replies; a message never recorded raises `LlmError`."""

    def __init__(self, records: Mapping[str, Mapping[str, Any]], model: str) -> None:
        self._records = {k: Recorded(**dict(v)) for k, v in records.items()}
        self._model = model
        self.misses = 0

    @property
    def name(self) -> str:
        return f"replay:{self._model}"

    async def complete(self, messages: Sequence[ChatMessage], schema: dict[str, Any]) -> LlmReply:
        record = self._records.get(messages_key(messages))
        if record is None:
            self.misses += 1
            raise LlmError(f"{self.name}: replay miss (messages never recorded)")
        return LlmReply(
            text=record.text,
            prompt_tokens=record.prompt_tokens,
            completion_tokens=record.completion_tokens,
            model=record.model,
        )

    async def aclose(self) -> None:
        return None
