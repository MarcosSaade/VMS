"""Quantized and merged serving (task 8.4): merged models built before the servers, their
servers' commands, vLLM's memory facts from its log, hosted cache-hit usage, token totals, and
the committed scenarios. No GPU, no merge script and no engine: the script runner and the
servers are injected."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from vms.common.settings import load_settings
from vms.lora import run as lora_run
from vms.serving import run as serving_run
from vms.serving.client import Sample, stream_chat
from vms.serving.scenario import LlmScenario, Server
from vms.serving.servers import ProcessServer, build_server, vllm_log_facts
from vms.serving.settings import ServingSettings

REPO_ROOT = Path(__file__).resolve().parents[3]
DEV = REPO_ROOT / "bench/scenarios/llm-quant-a40.yaml"
TEST = REPO_ROOT / "bench/scenarios/llm-quant-test-a40.yaml"
HOSTED = REPO_ROOT / "bench/scenarios/llm-load-hosted.yaml"
BASE = {"model": "Qwen/Q", "revision": "abc"}


def scenario_raw(
    tmp_path: Path, merges: list[dict[str, Any]], servers: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "name": "quant",
        "kind": "llm",
        "seed": 1,
        "catalog": str(REPO_ROOT / "eval/structured/catalog.v1.json"),
        "results_dir": str(tmp_path / "results"),
        "logs_dir": str(tmp_path / "logs"),
        "max_items": 1,
        "load": {"concurrency": [1]},
        "merges": merges,
        "servers": servers,
    }


def merged_server(name: str, merge: str, **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "engine": "vllm",
        **BASE,
        "merged": merge,
        "variants": [{"name": "v", "load": False}],
        **extra,
    }


def test_a_merged_server_must_name_a_merge_of_its_own_base(tmp_path: Path) -> None:
    merge = {"name": "m", **BASE, "adapter": "a.tar"}
    ok = scenario_raw(tmp_path, [merge], [merged_server("s", "m")])
    assert LlmScenario.model_validate(ok).servers[0].merged == "m"
    with pytest.raises(ValidationError, match="unknown merge"):
        LlmScenario.model_validate(scenario_raw(tmp_path, [merge], [merged_server("s", "x")]))
    other = merged_server("s", "m", revision="def")
    with pytest.raises(ValidationError, match="base"):
        LlmScenario.model_validate(scenario_raw(tmp_path, [merge], [other]))
    with pytest.raises(ValidationError, match="unique"):
        LlmScenario.model_validate(
            scenario_raw(tmp_path, [merge, merge], [merged_server("s", "m")])
        )
    with pytest.raises(ValidationError, match="vLLM only"):
        Server.model_validate(
            {
                "name": "s",
                "engine": "ollama",
                "model": "q",
                "merged": "m",
                "variants": [{"name": "v"}],
            }
        )


def test_a_local_model_is_served_without_a_revision(tmp_path: Path) -> None:
    settings = load_settings(ServingSettings, "serving", config_dir=REPO_ROOT / "configs")
    server = Server(name="m", engine="vllm", variants=[{"name": "v"}], **BASE)
    local = server.model_copy(update={"model": "/models/m", "revision": None})
    cmd = build_server(local, settings, tmp_path, seed=1)
    assert isinstance(cmd, ProcessServer)
    assert cmd.command[1:3] == ["serve", "/models/m"]
    assert "--revision" not in cmd.command
    pinned = build_server(server, settings, tmp_path, seed=1)
    assert isinstance(pinned, ProcessServer)
    assert pinned.command[pinned.command.index("--revision") + 1] == "abc"


class Runner:
    """Stands in for scripts/lora_merge.py: writes a model directory or fails."""

    def __init__(self, fail: set[str] | None = None) -> None:
        self.fail = fail or set()
        self.specs: list[dict[str, Any]] = []

    def __call__(
        self, python: str, script: Path, spec_path: Path, log_path: Path, timeout_s: float
    ) -> dict[str, Any]:
        spec = json.loads(spec_path.read_text())
        self.specs.append(spec)
        if Path(spec["out_dir"]).name in self.fail:
            raise RuntimeError("lora_merge.py exited with 1:\nCUDA out of memory")
        Path(spec["out_dir"]).mkdir(parents=True)
        assert script.name == "lora_merge.py"
        return {"merge_s": 1.0, "weights_bytes": 8}


class Recorder:
    """A server factory that remembers what it was asked to serve and never starts."""

    def __init__(self) -> None:
        self.models: dict[str, str] = {}

    def __call__(self, server: Server, *args: Any) -> Any:
        self.models[server.name] = server.model
        recorder = self

        class _Running:
            base_url = "http://127.0.0.1:1/v1"

            def start(self) -> dict[str, Any]:
                from vms.serving.servers import ServerStartError

                raise ServerStartError(f"{server.name}: not started in tests")

            def metrics_text(self) -> str | None:
                return None

            def stop(self) -> None:
                recorder.models.setdefault("stopped", "")

        return _Running()


def test_merges_are_built_first_and_a_failed_one_fails_only_its_server(tmp_path: Path) -> None:
    lora_run.pack_adapter(fake_adapter(tmp_path / "ad"), tmp_path / "ad.tar")
    merges = [
        {"name": "good", **BASE, "adapter": "ad.tar"},
        {"name": "bad", **BASE, "adapter": "ad.tar"},
    ]
    servers = [
        merged_server("s-good", "good"),
        merged_server("s-bad", "bad"),
        {"name": "plain", "engine": "vllm", **BASE, "variants": [{"name": "v", "load": False}]},
    ]
    (tmp_path / "s.yaml").write_text(json.dumps(scenario_raw(tmp_path, merges, servers)))
    (tmp_path / "configs").symlink_to(REPO_ROOT / "configs")
    (tmp_path / "eval").symlink_to(REPO_ROOT / "eval")
    runner, factory = Runner(fail={"bad"}), Recorder()
    path = serving_run.run_llm(tmp_path / "s.yaml", tmp_path, factory=factory, script_runner=runner)
    metrics = json.loads(path.read_text())["metrics"]
    assert metrics["merges"]["good"]["ok"] is True
    assert metrics["merges"]["good"]["adapter_sha256"] is not None
    assert "CUDA out of memory" in metrics["merges"]["bad"]["error"]
    assert [Path(s["out_dir"]).name for s in runner.specs] == ["good", "bad"]
    assert runner.specs[0]["revision"] == "abc"
    # The merged server serves the built directory; the failed merge's server never starts.
    assert factory.models["s-good"] == str((tmp_path / "data/serving/merged/good").resolve())
    assert "s-bad" not in factory.models
    assert metrics["servers"]["s-bad"] == {
        **metrics["servers"]["s-bad"],
        "ok": False,
        "error": "merge 'bad' was not built",
        "merged": "bad",
    }
    assert factory.models["plain"] == "Qwen/Q"
    assert metrics["servers"]["s-good"]["model"] == "Qwen/Q"  # recorded as the merge's base


def test_the_pinned_adapter_is_merged_only_at_its_hash(tmp_path: Path) -> None:
    settings = load_settings(ServingSettings, "serving", config_dir=REPO_ROOT / "configs")
    pin = settings.lora.planner
    assert pin is not None
    target = tmp_path / pin.path
    target.parent.mkdir(parents=True)
    lora_run.pack_adapter(fake_adapter(tmp_path / "ad"), target)  # not the pinned bytes
    raw = scenario_raw(
        tmp_path,
        [{"name": "m", **BASE, "adapter": str(pin.path)}],
        [merged_server("s", "m")],
    )
    (tmp_path / "s.yaml").write_text(json.dumps(raw))
    (tmp_path / "configs").symlink_to(REPO_ROOT / "configs")
    (tmp_path / "eval").symlink_to(REPO_ROOT / "eval")
    runner = Runner()
    path = serving_run.run_llm(
        tmp_path / "s.yaml", tmp_path, factory=Recorder(), script_runner=runner
    )
    merge = json.loads(path.read_text())["metrics"]["merges"]["m"]
    assert not merge["ok"]
    assert "pinned" in merge["error"]
    assert runner.specs == []


def fake_adapter(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(json.dumps({"r": 4}))
    (path / "adapter_model.safetensors").write_bytes(b"\0" * 32)
    return path


def test_vllm_memory_facts_are_read_from_its_startup_log() -> None:
    log = "\n".join(
        [
            "INFO [gpu_model_runner.py:5355] Model loading took 8.12 GiB memory and 3.5 seconds",
            "INFO [gpu_worker.py:693] Available KV cache memory: 30.47 GiB",
            "INFO [kv_cache_utils.py:2465] GPU KV cache size: 1,234,560 tokens, Maximum "
            "concurrency for 16,384 tokens per request: 75.35x",
        ]
    )
    assert vllm_log_facts(log) == {
        "weights_gib": 8.12,
        "kv_cache_gib": 30.47,
        "kv_cache_tokens": 1234560.0,
    }
    assert vllm_log_facts("nothing") == dict.fromkeys(
        ("weights_gib", "kv_cache_gib", "kv_cache_tokens")
    )


def test_hosted_cache_hits_are_read_and_rep_token_totals_kept() -> None:
    async def stream(url: str, body: dict[str, Any], timeout_s: float) -> Any:
        usage = {"prompt_tokens": 4700, "completion_tokens": 3, "prompt_cache_hit_tokens": 4608}
        for chunk in ({"choices": [{"delta": {"content": "{}"}}]}, {"choices": [], "usage": usage}):
            yield f"data: {json.dumps(chunk)}\n".encode()

    clock = iter(float(i) for i in range(10))
    sample = asyncio.run(stream_chat(stream, "u", {}, 10.0, lambda: next(clock)))
    assert sample.cached_tokens == 4608
    other = Sample(ok=True, latency_s=1.0, ttft_s=0.5, prompt_tokens=100, completion_tokens=7)
    rep = serving_run.summarize_rep([sample, other], 2.0, [True, True], {})
    # One request did not report its cache: the total covers what was reported.
    assert rep["tokens_total"] == {"prompt": 4800, "completion": 10, "cached": 4608}
    rep = serving_run.summarize_rep([other], 1.0, [True], {})
    assert rep["tokens_total"]["cached"] is None


def test_committed_quant_scenario_changes_one_variable_per_comparison() -> None:
    scenario, _ = serving_run.load_scenario(DEV)
    servers = {s.name: s for s in scenario.servers}
    bf16 = servers["qwen3.5-4b"]
    pin = load_settings(ServingSettings, "serving", config_dir=REPO_ROOT / "configs").lora.planner
    assert pin is not None
    assert (bf16.model, bf16.revision) == (pin.base_model, pin.base_revision)
    for name in ("qwen3.5-4b-w4a16", "qwen3.5-4b-w8a8"):  # base precision only
        quant = servers[name]
        assert quant.model.startswith("RedHatAI/Qwen3.5-4B-quantized.")
        assert quant.revision
        assert (quant.args, quant.request_extra, quant.lora_modules, quant.variants) == (
            bf16.args,
            bf16.request_extra,
            bf16.lora_modules,
            bf16.variants,
        )
    assert bf16.lora_modules == {"all-r4": pin.path}
    adapter = next(v for v in bf16.variants if v.model == "all-r4")
    assert adapter.prompt == pin.prompt
    merged = servers["merged-all-r4"]  # adapter vs merged: the same weights in effect
    (merge,) = scenario.merges
    assert (merge.model, merge.revision, merge.adapter) == (
        pin.base_model,
        pin.base_revision,
        pin.path,
    )
    assert merged.args == bf16.args
    assert [v.prompt for v in merged.variants] == [pin.prompt]
    assert "--enable-prefix-caching" in bf16.args  # the deployment configuration (D40)
    assert all(v.record and v.load for s in scenario.servers for v in s.variants)
    assert scenario.split == "dev"
    assert scenario.load.repetitions >= 3


def test_committed_test_scenario_is_the_dev_server_recording_test_once() -> None:
    dev = {s.name: s for s in serving_run.load_scenario(DEV)[0].servers}
    test, _ = serving_run.load_scenario(TEST)
    (server,) = test.servers
    same = dev[server.name]
    assert (server.model, server.revision, server.args, server.lora_modules) == (
        same.model,
        same.revision,
        same.args,
        same.lora_modules,
    )
    (variant,) = server.variants
    assert (variant.model, variant.record, variant.load) == ("all-r4", True, False)
    assert test.split == "test"


def test_hosted_load_matches_the_pod_load_profile() -> None:
    dev, _ = serving_run.load_scenario(DEV)
    hosted, _ = serving_run.load_scenario(HOSTED)
    assert hosted.load == dev.load
    (server,) = hosted.servers
    assert server.engine == "external"
    assert server.use_llm_api_key
    assert not any(v.record for v in server.variants)  # its test accuracy is E-036's
    assert yaml.safe_load(HOSTED.read_text())["kind"] == "llm"


def test_the_self_hosted_planner_config_is_the_one_tested() -> None:
    """configs/serving.yaml planner_server (D43) is the server of the quantized-LoRA test run,
    with the pinned adapter, so the committed serving config is what the test number measured."""
    settings = load_settings(ServingSettings, "serving", config_dir=REPO_ROOT / "configs")
    deploy, pin = settings.planner_server, settings.lora.planner
    assert deploy is not None
    assert pin is not None
    (server,) = serving_run.load_scenario(TEST)[0].servers
    assert (deploy.model, deploy.revision, deploy.args, deploy.request_extra) == (
        server.model,
        server.revision,
        server.args,
        server.request_extra,
    )
    assert server.lora_modules == {pin.name: pin.path}
    assert [v.prompt for v in server.variants] == [pin.prompt]
