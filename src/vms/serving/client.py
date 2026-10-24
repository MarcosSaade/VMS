"""Streaming chat requests and closed-loop load (task 8.1).

One request is an OpenAI-compatible `/chat/completions` call with `stream: true` and usage in
the last chunk, so time to first token (TTFT) is the arrival of the first non-empty content
delta and decode speed is the rest of the reply over the rest of the time. The transport is
injected (an async iterator of SSE lines), so tests need no sockets; the clock too.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import aiohttp

# (url, body, timeout_s) -> the response's lines; raises on a non-200 status.
StreamFn = Callable[[str, dict[str, Any], float], AsyncIterator[bytes]]
Clock = Callable[[], float]


class HttpStatusError(Exception):
    pass


@dataclass
class Sample:
    """One request's outcome; times in seconds from its start."""

    ok: bool
    latency_s: float
    ttft_s: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int | None = None
    text: str = ""
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def decode_tokens_per_s(self) -> float | None:
        """Tokens after the first over the time after the first (the decode rate)."""
        if self.ttft_s is None or self.completion_tokens < 2:
            return None
        decode_s = self.latency_s - self.ttft_s
        return (self.completion_tokens - 1) / decode_s if decode_s > 0 else None


def aiohttp_stream(
    session: aiohttp.ClientSession, headers: dict[str, str] | None = None
) -> StreamFn:
    async def stream(url: str, body: dict[str, Any], timeout_s: float) -> AsyncIterator[bytes]:
        timeout = aiohttp.ClientTimeout(total=timeout_s)
        async with session.post(url, json=body, headers=headers, timeout=timeout) as resp:
            if resp.status != 200:
                text = (await resp.text())[:300]
                raise HttpStatusError(f"HTTP {resp.status}: {text}")
            async for line in resp.content:
                yield line

    return stream


def _usage(chunk: dict[str, Any], sample: Sample) -> None:
    usage = chunk.get("usage")
    if not isinstance(usage, dict):
        return
    sample.prompt_tokens = int(usage.get("prompt_tokens") or 0)
    sample.completion_tokens = int(usage.get("completion_tokens") or 0)
    details = usage.get("prompt_tokens_details")
    if isinstance(details, dict) and details.get("cached_tokens") is not None:
        sample.cached_tokens = int(details["cached_tokens"])
    elif usage.get("prompt_cache_hit_tokens") is not None:  # DeepSeek's name for the same
        sample.cached_tokens = int(usage["prompt_cache_hit_tokens"])


async def stream_chat(
    stream: StreamFn, url: str, body: dict[str, Any], timeout_s: float, clock: Clock
) -> Sample:
    """Send one streamed request; never raises (errors are samples with ok=False)."""
    body = {**body, "stream": True, "stream_options": {"include_usage": True}}
    t0 = clock()
    sample = Sample(ok=False, latency_s=0.0)
    parts: list[str] = []
    try:
        async with asyncio.timeout(timeout_s):
            async for raw in stream(url, body, timeout_s):
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                _usage(chunk, sample)
                for choice in chunk.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        if sample.ttft_s is None:
                            sample.ttft_s = clock() - t0
                        parts.append(delta)
        sample.ok = True
    except (TimeoutError, aiohttp.ClientError, OSError, HttpStatusError, ValueError) as exc:
        sample.error = f"{type(exc).__name__}: {exc}"[:300]
    sample.latency_s = clock() - t0
    sample.text = "".join(parts)
    if sample.ok and sample.ttft_s is None:
        sample.ok, sample.error = False, "empty reply"
    return sample


async def closed_loop(
    send: Callable[[dict[str, Any]], Any],
    bodies: Sequence[dict[str, Any]],
    concurrency: int,
    clock: Clock = time.perf_counter,
) -> tuple[list[Sample], float]:
    """Every body once with at most `concurrency` in flight; (samples in order, wall seconds)."""
    limit = asyncio.Semaphore(concurrency)

    async def one(body: dict[str, Any]) -> Sample:
        async with limit:
            result: Sample = await send(body)
            return result

    t0 = clock()
    samples = await asyncio.gather(*(one(b) for b in bodies))
    return list(samples), clock() - t0
