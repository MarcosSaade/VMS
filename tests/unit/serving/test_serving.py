"""LLM serving benchmark (task 8.1, D40): SSE parsing and timing, closed-loop load, per-rep
summaries, Prometheus counters, record and replay, the committed scenario, and a server that
fails to start. No sockets and no engines: the transport, clock and server are injected."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from vms.query.llm import ChatMessage, LlmError, LlmReply
from vms.serving import run as serving_run
from vms.serving.catalog_file import catalog_json, load_catalog_file, write_catalog
from vms.serving.client import HttpStatusError, Sample, closed_loop, stream_chat
from vms.serving.replay import RecordingLlm, ReplayLlm, messages_key
from vms.serving.scenario import LlmScenario
from vms.serving.servers import ServerStartError, prometheus_counters

REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIO = REPO_ROOT / "bench/scenarios/llm-serve-a40.yaml"


class StepClock:
    """Advances by `step` seconds on every read."""

    def __init__(self, step: float = 0.1) -> None:
        self.t = 0.0
        self.step = step

    def __call__(self) -> float:
        self.t += self.step
        return self.t


def sse(*chunks: dict[str, Any], done: bool = True) -> list[bytes]:
    lines = [f"data: {json.dumps(c)}\n".encode() for c in chunks]
    lines.insert(0, b": keep-alive\n")
    if done:
        lines.append(b"data: [DONE]\n")
    return lines


def fake_stream(lines: list[bytes] | Exception) -> Any:
    sent: list[dict[str, Any]] = []

    async def stream(url: str, body: dict[str, Any], timeout_s: float) -> AsyncIterator[bytes]:
        sent.append(body)
        if isinstance(lines, Exception):
            raise lines
        for line in lines:
            yield line

    stream.sent = sent  # type: ignore[attr-defined]
    return stream


def delta(text: str) -> dict[str, Any]:
    return {"choices": [{"delta": {"content": text}}]}


def test_stream_chat_times_first_token_and_reads_usage() -> None:
    usage = {
        "choices": [],
        "usage": {
            "prompt_tokens": 4700,
            "completion_tokens": 21,
            "prompt_tokens_details": {"cached_tokens": 4608},
        },
    }
    lines = sse(
        {"choices": [{"delta": {"role": "assistant"}}]}, delta('{"in'), delta('tent"}'), usage
    )
    stream = fake_stream(lines)
    sample = asyncio.run(stream_chat(stream, "u", {"model": "m"}, 10.0, StepClock(0.1)))
    assert sample.ok
    assert sample.text == '{"intent"}'
    # t0 = 0.1; the first content arrives at the next read (0.2); the end is read at 0.3.
    assert sample.ttft_s == pytest.approx(0.1)
    assert sample.latency_s == pytest.approx(0.2)
    assert (sample.prompt_tokens, sample.completion_tokens, sample.cached_tokens) == (
        4700,
        21,
        4608,
    )
    assert sample.decode_tokens_per_s == pytest.approx(20 / 0.1)
    assert stream.sent[0]["stream"] is True
    assert stream.sent[0]["stream_options"] == {"include_usage": True}


@pytest.mark.parametrize(
    ("lines", "error"),
    [
        (HttpStatusError("HTTP 400: bad schema"), "HTTP 400"),
        (sse(delta(""), done=True), "empty reply"),
        ([b"data: {not json\n"], "JSONDecodeError"),
    ],
)
def test_stream_chat_turns_failures_into_samples(lines: Any, error: str) -> None:
    sample = asyncio.run(stream_chat(fake_stream(lines), "u", {}, 10.0, StepClock()))
    assert not sample.ok
    assert sample.error is not None
    assert error in sample.error


def test_closed_loop_caps_requests_in_flight_and_keeps_order() -> None:
    in_flight, peak = 0, 0

    async def send(body: dict[str, Any]) -> Sample:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0)  # yield so others can start
        in_flight -= 1
        return Sample(ok=True, latency_s=0.0, text=str(body["i"]))

    samples, wall = asyncio.run(closed_loop(send, [{"i": i} for i in range(10)], 3, StepClock()))
    assert [s.text for s in samples] == [str(i) for i in range(10)]
    assert peak == 3
    assert wall == pytest.approx(0.1)


def test_summarize_rep_rates_and_prefix_hit_rate() -> None:
    ok = Sample(ok=True, latency_s=2.0, ttft_s=0.5, prompt_tokens=4700, completion_tokens=31)
    bad = Sample(ok=False, latency_s=1.0, error="HTTP 500")
    cache = {"vllm:prefix_cache_queries_total": 9400.0, "vllm:prefix_cache_hits_total": 8000.0}
    rep = serving_run.summarize_rep([ok, ok, bad, ok], 4.0, [True, False, False, True], cache)
    assert rep["error_rate"] == 0.25
    assert rep["valid_rate"] == 0.5  # of all requests sent, errors count as invalid
    assert rep["requests_per_s"] == 0.75
    assert rep["output_tokens_per_s"] == pytest.approx(93 / 4.0)
    assert rep["decode_tokens_per_s_p50"] == pytest.approx(30 / 1.5)
    assert rep["prefix_cache_hit_rate"] == pytest.approx(8000 / 9400)
    assert rep["latency_ms"]["p50"] == pytest.approx(2000.0)
    assert rep["first_errors"] == ["HTTP 500"]


def test_prometheus_counters_sum_labels_and_skip_comments() -> None:
    text = (
        "# HELP vllm:prefix_cache_hits_total hits\n"
        'vllm:prefix_cache_hits_total{engine="0",model_name="m"} 120.0\n'
        'vllm:prefix_cache_hits_total{engine="1",model_name="m"} 30.0\n'
        "vllm:prefix_cache_queries_total 200\n"
        "vllm:num_requests_running 3\n"
    )
    assert prometheus_counters(text, "prefix_cache") == {
        "vllm:prefix_cache_hits_total": 150.0,
        "vllm:prefix_cache_queries_total": 200.0,
    }
    assert prometheus_counters(None, "prefix_cache") == {}


class ScriptedLlm:
    name = "openai:m"

    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)

    async def complete(self, messages: Any, schema: Any) -> LlmReply:
        return LlmReply(text=self.texts.pop(0), prompt_tokens=10, completion_tokens=3, model="m")

    async def aclose(self) -> None:
        return None


def test_recorded_replies_replay_exactly_and_misses_raise() -> None:
    first = [ChatMessage("system", "s"), ChatMessage("user", "how many cars?")]
    repair = [*first, ChatMessage("assistant", "{bad"), ChatMessage("user", "fix it")]
    recorder = RecordingLlm(ScriptedLlm("{bad", '{"intent": "count"}'), clock=StepClock())
    asyncio.run(recorder.complete(first, {}))
    asyncio.run(recorder.complete(repair, {}))
    dumped = json.loads(json.dumps(recorder.dump()))  # through a results file
    replay = ReplayLlm(dumped, "qwen")
    assert asyncio.run(replay.complete(first, {})).text == "{bad"
    reply = asyncio.run(replay.complete(repair, {}))
    assert (reply.text, reply.prompt_tokens, reply.model) == ('{"intent": "count"}', 10, "m")
    with pytest.raises(LlmError, match="replay miss"):
        asyncio.run(replay.complete([ChatMessage("system", "s2"), first[1]], {}))
    assert replay.misses == 1
    assert replay.name == "replay:qwen"


def test_messages_key_depends_on_role_content_and_order() -> None:
    a, b = ChatMessage("system", "x"), ChatMessage("user", "y")
    keys = {
        messages_key([a, b]),
        messages_key([b, a]),
        messages_key([ChatMessage("user", "x"), b]),
        messages_key([a, ChatMessage("user", "y ")]),
    }
    assert len(keys) == 4


def test_catalog_file_round_trips(tmp_path: Path) -> None:
    import datetime as dt

    now = dt.datetime(2026, 3, 31, 14, 30, tzinfo=dt.UTC)
    data = catalog_json(
        [("cam-1", "entrance", "Europe/London")],
        [("cam-1:door", "cam-1", "door")],
        ["person"],
        now,
        "abc",
    )
    write_catalog(tmp_path / "c.json", data)
    catalog, loaded_now = load_catalog_file(tmp_path / "c.json")
    assert loaded_now == now
    assert catalog.camera("entrance") is not None
    assert catalog.zone("door") is not None
    assert catalog.classes == ("person",)


def test_committed_eval_catalog_matches_the_snapshot_manifest() -> None:
    import hashlib

    data = json.loads((REPO_ROOT / "eval/structured/catalog.v1.json").read_text())
    manifest = (REPO_ROOT / "eval/structured/snapshot.v1.json").read_bytes()
    assert data["snapshot_manifest_sha256"] == hashlib.sha256(manifest).hexdigest()
    assert data["now"] == json.loads(manifest)["now"].replace("Z", "+00:00")


def test_committed_scenario_changes_one_variable_per_comparison() -> None:
    scenario, _ = serving_run.load_scenario(SCENARIO)
    servers = {s.name: s for s in scenario.servers}
    for base in ("qwen3.5-9b", "qwen2.5-7b"):
        on, off = servers[base], servers[f"{base}-noprefix"]
        assert on.model == off.model
        assert on.revision == off.revision
        flip = {"--enable-prefix-caching": "--no-enable-prefix-caching"}
        assert [flip.get(a, a) for a in on.args] == off.args
        assert {v.response_format for v in on.variants} == {"json_schema", "none"}
    assert all(s.revision for s in scenario.servers if s.engine == "vllm")
    assert scenario.split == "dev"
    assert scenario.load.repetitions >= 3


def test_scenario_rejects_unpinned_vllm_models() -> None:
    raw = {
        "name": "x",
        "seed": 1,
        "catalog": "c.json",
        "load": {"concurrency": [1]},
        "servers": [
            {"name": "a", "engine": "vllm", "model": "Qwen/Q", "variants": [{"name": "v"}]}
        ],
    }
    with pytest.raises(ValidationError, match="revision"):
        LlmScenario.model_validate(raw)


class BrokenServer:
    base_url = "http://127.0.0.1:1/v1"
    stopped = 0

    def start(self) -> dict[str, Any]:
        raise ServerStartError("exited with 1:\nCUDA out of memory")

    def metrics_text(self) -> str | None:
        return None

    def stop(self) -> None:
        BrokenServer.stopped += 1


def test_a_server_that_fails_to_start_is_recorded_and_the_rest_still_run(tmp_path: Path) -> None:
    scenario = tmp_path / "s.yaml"
    raw = {
        "name": "llm-broken",
        "kind": "llm",
        "seed": 1,
        "catalog": str(REPO_ROOT / "eval/structured/catalog.v1.json"),
        "results_dir": str(tmp_path / "results"),
        "load": {"concurrency": [1]},
        "servers": [
            {
                "name": n,
                "engine": "external",
                "model": "m",
                "base_url": "http://x/v1",
                "variants": [{"name": "v"}],
            }
            for n in ("a", "b")
        ],
    }
    scenario.write_text(json.dumps(raw))
    path = serving_run.run_llm(scenario, REPO_ROOT, factory=lambda *a: BrokenServer())
    result = json.loads(path.read_text())
    servers = result["metrics"]["servers"]
    assert [servers[n]["ok"] for n in ("a", "b")] == [False, False]
    assert "CUDA out of memory" in servers["a"]["error"]
    assert BrokenServer.stopped == 2  # stopped even when it failed
