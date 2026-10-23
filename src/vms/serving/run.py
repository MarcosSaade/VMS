"""Run a `kind: llm` scenario (task 8.1, D40): for each server, start it, load test and record
each variant, stop it; one results JSON (kind `bench.llm`) for the whole scenario.

Requests are the planner's own first-attempt messages (the long system prompt with schema,
catalog and few-shot bank, then one dev question), built exactly as the query service builds
them, so prefix caching sees the real shared prefix. Plan validity is checked the way the
planner checks it (parse, then validate against the catalog and policy), first attempt only.
Recording runs the planner itself, with its repair turn, and keeps every reply for the replayed
eval (`python -m vms.eval run --planner replay`). A server that fails to start is recorded with
its log tail and the scenario moves on to the next one.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiohttp
import yaml
from pydantic import SecretStr

from vms.bench.results import RunResult, git_info, hardware, package_versions, write_result
from vms.bench.stats import aggregate, percentiles_ms
from vms.common.ids import new_ulid
from vms.common.logs import get_logger
from vms.common.settings import load_settings
from vms.eval.dataset import EvalItem, load_items
from vms.eval.harness import eval_planner
from vms.eval.settings import EvalSettings
from vms.query.catalog import QueryCatalog
from vms.query.llm import (
    ChatMessage,
    LlmEndpointSettings,
    LlmTuning,
    OpenAICompatibleClient,
)
from vms.query.plan import plan_json_schema
from vms.query.planner import PlannerUnavailableError, PlanningFailedError, parse_plan
from vms.query.semantic import validate_plan
from vms.query.settings import QuerySettings
from vms.serving.catalog_file import load_catalog_file
from vms.serving.client import Sample, aiohttp_stream, closed_loop, stream_chat
from vms.serving.replay import RecordingLlm
from vms.serving.scenario import LlmScenario, Server, Variant
from vms.serving.servers import (
    RunningServer,
    ServerStartError,
    build_server,
    gpu_memory_used_mib,
    prometheus_counters,
)
from vms.serving.settings import ServingSettings

log = get_logger(__name__)

KIND = "bench.llm"
PACKAGES = ("aiohttp", "pydantic")  # engine versions come from each server (`info`)
REP_KEYS = (
    "requests_per_s",
    "output_tokens_per_s",
    "error_rate",
    "valid_rate",
    "latency_ms.p50",
    "latency_ms.p95",
    "latency_ms.p99",
    "ttft_ms.p50",
    "ttft_ms.p95",
    "decode_tokens_per_s_p50",
    "prefix_cache_hit_rate",
)


def load_scenario(path: Path) -> tuple[LlmScenario, dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return LlmScenario.model_validate(raw), raw


class Context:
    """What every server and variant shares: the planner, its prompt inputs and the items."""

    def __init__(self, scenario: LlmScenario, root: Path) -> None:
        self.scenario = scenario
        self.root = root
        self.qs = load_settings(QuerySettings, "query", config_dir=root / "configs")
        eval_settings = load_settings(EvalSettings, "eval", config_dir=root / "configs")
        self.items: list[EvalItem] = load_items(root / eval_settings.structured.dev)[
            : scenario.max_items
        ]
        self.catalog: QueryCatalog
        self.catalog, self.now = load_catalog_file(root / scenario.catalog)
        self.schema = plan_json_schema()
        # The planner for prompts only; recording builds one per variant around its client.
        planner = eval_planner(_NoLlm(), self.qs, self.now, root)
        self.system_prompt = planner.system_prompt(self.catalog, self.now)

    def tuning(self, server: Server, variant: Variant) -> LlmTuning:
        return self.qs.llm.model_copy(
            update={
                "response_format": variant.response_format,
                "max_tokens": self.scenario.max_tokens,
                "temperature": self.scenario.temperature,
                "extra_body": server.request_extra,
                "timeout_s": self.scenario.load.timeout_s,
                "max_retries": 0,  # a failed request is a measured error, not retried
            }
        )

    def client(self, server: Server, variant: Variant, base_url: str) -> OpenAICompatibleClient:
        key = api_key(server)
        endpoint = LlmEndpointSettings(  # type: ignore[call-arg]
            _env_file=None,
            base_url=base_url,
            model=server.request_model,
            api_key=SecretStr(key) if key else None,
        )
        return OpenAICompatibleClient(endpoint, self.tuning(server, variant))

    def body(self, client: OpenAICompatibleClient, item: EvalItem) -> dict[str, Any]:
        messages = [ChatMessage("system", self.system_prompt), ChatMessage("user", item.question)]
        return client.request_body(messages, self.schema)

    def plan_valid(self, text: str) -> bool:
        plan, errors = parse_plan(text)
        if plan is None:
            return False
        errors = validate_plan(
            plan, self.catalog, self.now, self.qs.planner, max_limit=self.qs.sql.max_limit
        )
        return not errors


def api_key(server: Server) -> str | None:
    """An external server's key: LLM_API_KEY from the environment or `.env` (never written to
    results)."""
    if not server.use_llm_api_key:
        return None
    key = LlmEndpointSettings().api_key
    return key.get_secret_value() if key is not None else None


class _NoLlm:
    name = "none"

    async def complete(self, messages: Any, schema: Any) -> Any:
        raise AssertionError("prompt-only planner")

    async def aclose(self) -> None:
        return None


def summarize_rep(
    samples: Sequence[Sample], wall_s: float, valid: Sequence[bool], cache: dict[str, float]
) -> dict[str, Any]:
    ok = [s for s in samples if s.ok]
    tokens = sum(s.completion_tokens for s in ok)
    decode = sorted(r for s in ok if (r := s.decode_tokens_per_s) is not None)
    queries = sum(v for k, v in cache.items() if k.startswith("vllm:prefix_cache_queries"))
    hits = sum(v for k, v in cache.items() if k.startswith("vllm:prefix_cache_hits"))
    return {
        "requests": len(samples),
        "errors": len(samples) - len(ok),
        "error_rate": (len(samples) - len(ok)) / len(samples) if samples else None,
        "valid_rate": sum(valid) / len(samples) if samples else None,
        "wall_s": round(wall_s, 3),
        "requests_per_s": len(ok) / wall_s if wall_s > 0 else None,
        "output_tokens_per_s": tokens / wall_s if wall_s > 0 else None,
        "prompt_tokens_mean": (sum(s.prompt_tokens for s in ok) / len(ok)) if ok else None,
        "completion_tokens_mean": tokens / len(ok) if ok else None,
        "latency_ms": percentiles_ms([s.latency_s for s in ok]),
        "ttft_ms": percentiles_ms([s.ttft_s for s in ok if s.ttft_s is not None]),
        "decode_tokens_per_s_p50": decode[len(decode) // 2] if decode else None,
        "prefix_cache_hit_rate": hits / queries if queries else None,
        "prefix_cache_counters": cache,
        "first_errors": sorted({s.error for s in samples if s.error})[:3],
    }


def _delta(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {k: v - before.get(k, 0.0) for k, v in after.items()}


async def load_variant(
    ctx: Context,
    server: Server,
    running: RunningServer,
    variant: Variant,
    session: aiohttp.ClientSession,
) -> dict[str, Any]:
    spec = ctx.scenario.load
    client = ctx.client(server, variant, running.base_url)
    url = running.base_url.rstrip("/") + "/chat/completions"
    key = api_key(server)
    stream = aiohttp_stream(session, {"Authorization": f"Bearer {key}"} if key else None)
    order = list(ctx.items)
    random.Random(ctx.scenario.seed).shuffle(order)
    bodies = [ctx.body(client, item) for item in order]

    async def send(body: dict[str, Any]) -> Sample:
        return await stream_chat(stream, url, body, spec.timeout_s, time.perf_counter)

    cursor = 0

    def take(n: int) -> list[dict[str, Any]]:
        nonlocal cursor
        out = [bodies[(cursor + i) % len(bodies)] for i in range(n)]
        cursor += n
        return out

    levels: dict[str, Any] = {}
    for concurrency in spec.concurrency:
        for body in take(spec.warmup):
            await send(body)
        reps = []
        cpu0 = time.process_time()
        for _ in range(spec.repetitions):
            before = prometheus_counters(running.metrics_text(), "prefix_cache")
            samples, wall = await closed_loop(send, take(spec.requests(concurrency)), concurrency)
            after = prometheus_counters(running.metrics_text(), "prefix_cache")
            valid = [s.ok and ctx.plan_valid(s.text) for s in samples]
            reps.append(summarize_rep(samples, wall, valid, _delta(after, before)))
        levels[str(concurrency)] = {
            "reps": reps,
            "spread": aggregate(reps, REP_KEYS),
            # The load client's own CPU over the level: it must stay far below one core.
            "client_cpu_s": round(time.process_time() - cpu0, 3),
            "client_wall_s": round(sum(r["wall_s"] for r in reps), 3),
            "gpu_memory_used_mib": gpu_memory_used_mib(),
        }
        log.info(
            "load level done",
            extra={"server": server.name, "variant": variant.name, "concurrency": concurrency},
        )
    return levels


async def record_variant(
    ctx: Context, server: Server, running: RunningServer, variant: Variant
) -> dict[str, Any]:
    """Plan every dev question with the planner (repair included); keep every reply."""
    recorder = RecordingLlm(ctx.client(server, variant, running.base_url))
    planner = eval_planner(recorder, ctx.qs, ctx.now, ctx.root)
    limit = asyncio.Semaphore(ctx.scenario.record.concurrency)
    outcomes = {"planned": 0, "failed_validation": 0, "unavailable": 0}

    async def one(item: EvalItem) -> None:
        async with limit:
            try:
                await planner.plan(item.question, ctx.catalog)
                outcomes["planned"] += 1
            except PlanningFailedError:
                outcomes["failed_validation"] += 1
            except PlannerUnavailableError:
                outcomes["unavailable"] += 1

    t0 = time.perf_counter()
    try:
        await asyncio.gather(*(one(i) for i in ctx.items))
    finally:
        await recorder.aclose()
    latencies = [r.latency_ms / 1000.0 for r in recorder.records.values()]
    return {
        "items": len(ctx.items),
        "outcomes": outcomes,
        "wall_s": round(time.perf_counter() - t0, 3),
        "call_latency_ms": percentiles_ms(latencies),
        "model": server.request_model,
        "system_prompt_chars": len(ctx.system_prompt),
        "replies": recorder.dump(),
    }


async def run_server(ctx: Context, server: Server, running: RunningServer) -> dict[str, Any]:
    out: dict[str, Any] = {"variants": {}}
    async with aiohttp.ClientSession() as session:
        for variant in server.variants:
            v: dict[str, Any] = {"response_format": variant.response_format}
            if variant.load:
                v["load"] = await load_variant(ctx, server, running, variant, session)
            if variant.record:
                v["record"] = await record_variant(ctx, server, running, variant)
            out["variants"][variant.name] = v
    return out


def run_llm(
    scenario_path: Path,
    root: Path = Path(),
    *,
    factory: Any = build_server,
) -> Path:
    scenario, raw = load_scenario(scenario_path)
    settings = load_settings(ServingSettings, "serving", config_dir=root / "configs")
    ctx = Context(scenario, root)
    started = datetime.now(UTC)
    servers: dict[str, Any] = {}
    for server in scenario.servers:
        running = factory(server, settings, root / scenario.logs_dir, scenario.seed)
        entry: dict[str, Any] = {
            "engine": server.engine,
            "model": server.model,
            "revision": server.revision,
            "args": server.args,
            "env": server.env,
            "request_extra": server.request_extra,
        }
        try:
            entry["info"] = running.start()
            entry["gpu_memory_used_mib_idle"] = gpu_memory_used_mib()
            entry.update(asyncio.run(run_server(ctx, server, running)))
            entry["ok"] = True
        except ServerStartError as exc:
            entry["ok"], entry["error"] = False, str(exc)
            log.error("server failed to start", extra={"server": server.name})
        except Exception as exc:  # one broken server must not cost the rest of the session
            entry["ok"], entry["error"] = False, f"{type(exc).__name__}: {exc}"
            log.exception("server run failed", extra={"server": server.name})
        finally:
            running.stop()
        servers[server.name] = entry
    result = RunResult(
        run_id=new_ulid(),
        name=scenario.name,
        kind=KIND,
        started_at=started,
        ended_at=datetime.now(UTC),
        seed=scenario.seed,
        git=git_info(root),
        hardware=hardware(),
        packages=package_versions(PACKAGES),
        config={
            "serving": settings.model_dump(mode="json"),
            "llm": ctx.qs.llm.model_dump(mode="json"),
            "assets": ctx.qs.assets.model_dump(mode="json"),
            "eval_clock": ctx.now.isoformat(),
            "items": len(ctx.items),
        },
        scenario=raw,
        metrics={"servers": servers, "system_prompt_chars": len(ctx.system_prompt)},
        notes="Closed-loop load on the planner's real first-attempt prompts (dev questions), "
        "warmup excluded; times are measured on the pod (client and server on one host). "
        "valid_rate = first-attempt replies that parse and pass the planner's validation. "
        "Recorded replies feed `python -m vms.eval run --planner replay` (D40).",
    )
    path = write_result(root / scenario.results_dir, result)
    log.info("llm scenario written", extra={"path": str(path)})
    return path
