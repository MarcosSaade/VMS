"""LoRA fine-tune (task 8.3, D42): the compact planner prompt, training examples, the committed
ablation scenario, LoRA serving flags, adapter packing, the run loop with its time budget, and
replaying replies recorded with another prompt. No GPU, no trainer and no engine: the trainer,
clock and server are injected."""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from vms.common.settings import load_settings
from vms.eval.cli import load_replay
from vms.lora import run as lora_run
from vms.lora.data import (
    load_catalogs,
    render_examples,
    split_examples,
    with_prompt,
    write_jsonl,
)
from vms.query.plan import QueryPlan
from vms.query.prompts import PromptTemplate
from vms.query.settings import QuerySettings
from vms.serving.run import VLLM_LORA_RANKS, with_lora
from vms.serving.scenario import Server
from vms.serving.servers import ServerStartError

REPO_ROOT = Path(__file__).resolve().parents[3]
SCENARIO = REPO_ROOT / "bench/scenarios/lora-plan-a40.yaml"
COMPACT = Path("configs/prompts/planner.compact.v1.txt")
CATALOGS = REPO_ROOT / "configs/traindata/catalogs.v1.json"


@pytest.fixture(scope="module")
def qs() -> QuerySettings:
    return load_settings(QuerySettings, "query", config_dir=REPO_ROOT / "configs")


def test_planner_prompts_may_drop_schema_and_examples_but_not_the_catalog(tmp_path: Path) -> None:
    compact = PromptTemplate.load_planner(REPO_ROOT / COMPACT)
    assert {"plan_schema", "examples"}.isdisjoint(compact.fields)
    missing = tmp_path / "p.txt"
    missing.write_text("$schema_summary $classes $now")  # no cameras
    with pytest.raises(ValueError, match="must include"):
        PromptTemplate.load_planner(missing)
    unknown = tmp_path / "q.txt"
    unknown.write_text("$schema_summary $cameras $classes $now $secret")
    with pytest.raises(ValueError, match="stay within"):
        PromptTemplate.load_planner(unknown)


def test_compact_prompt_is_the_full_prompt_without_schema_and_examples(qs: QuerySettings) -> None:
    catalogs = load_catalogs(CATALOGS)
    pair = {"id": "p1", "state": "warehouse", "question": "how many cars today?"}
    plan = {"intent": "count", "target": "tracks", "filters": {"classes": ["car"]}}
    now = catalogs["warehouse"][1].isoformat()
    full = render_examples([{**pair, "now": now, "plan": plan}], catalogs, qs, REPO_ROOT)[0]
    compact = render_examples(
        [{**pair, "now": now, "plan": plan}], catalogs, with_prompt(qs, COMPACT), REPO_ROOT
    )[0]
    assert "JSON schema:" in full.system
    assert "Examples:" in full.system
    assert "JSON schema:" not in compact.system
    assert "Examples:" not in compact.system
    assert len(compact.system) < len(full.system) / 3
    for text in (full.system, compact.system):  # the state's own catalog and clock
        assert "main_gate (Asia/Tokyo)" in text
        assert "2026-05-14 12:15" in text  # 03:15 UTC in Tokyo
    assert QueryPlan.model_validate_json(compact.completion) == QueryPlan.model_validate(plan)


def test_a_pair_whose_clock_is_not_its_states_is_refused(qs: QuerySettings) -> None:
    pair = {
        "id": "p1",
        "state": "warehouse",
        "now": "2026-01-01T00:00:00+00:00",
        "question": "q",
        "plan": {"intent": "abstain", "abstain_reason": "unanswerable"},
    }
    with pytest.raises(ValueError, match="clock"):
        render_examples([pair], load_catalogs(CATALOGS), qs, REPO_ROOT)


def examples(n: int, qs: QuerySettings) -> list[Any]:
    catalogs = load_catalogs(CATALOGS)
    now = catalogs["campus"][1].isoformat()
    pairs = [
        {
            "id": f"c{k:05d}",
            "state": "campus",
            "now": now,
            "question": f"question {k}",
            "plan": {"intent": "abstain", "abstain_reason": "unanswerable"},
        }
        for k in range(n)
    ]
    return render_examples(pairs, catalogs, with_prompt(qs, COMPACT), REPO_ROOT)


def test_split_is_seeded_disjoint_and_independent_of_input_order(qs: QuerySettings) -> None:
    ex = examples(40, qs)
    train, val = split_examples(ex, 0.1, seed=3)
    again, val2 = split_examples(list(reversed(ex)), 0.1, seed=3)
    assert [e.id for e in train] == [e.id for e in again]
    assert [e.id for e in val] == [e.id for e in val2]
    assert len(val) == 4
    assert {e.id for e in train}.isdisjoint({e.id for e in val})
    assert [e.id for e in split_examples(ex, 0.1, seed=4)[0]] != [e.id for e in train]


def test_committed_scenario_ablates_one_variable_against_the_baseline() -> None:
    scenario, _ = lora_run.load_scenario(SCENARIO)
    runs = {r.name: r for r in scenario.runs}
    base = runs[scenario.baseline]
    assert base.pairs is None
    for run in scenario.runs:
        changed = {"pairs": run.pairs != base.pairs, "rank": run.rank != base.rank}
        assert sum(changed.values()) <= 1, run.name
    assert {r.pairs for r in scenario.runs} >= {500, 1000, None}
    assert len({r.rank for r in scenario.runs}) >= 3
    assert scenario.runs[0].pairs == 500  # the cheap run first: it measures throughput
    assert scenario.data.prompt == COMPACT
    prompts = scenario.serve.base_prompts
    assert set(prompts.values()) == {Path("configs/prompts/planner.v1.txt"), COMPACT}
    assert scenario.base.revision


def test_servers_only_serve_named_adapters_on_vllm() -> None:
    raw = {
        "name": "s",
        "engine": "vllm",
        "model": "Qwen/Q",
        "revision": "abc",
        "lora_modules": {"a": "models/lora/a.tar"},
        "variants": [{"name": "v", "model": "b"}],
    }
    with pytest.raises(ValidationError, match="unknown adapters"):
        Server.model_validate(raw)
    ollama = {"name": "o", "engine": "ollama", "model": "m", "variants": [{"name": "v"}]}
    with pytest.raises(ValidationError, match="vLLM only"):
        Server.model_validate({**ollama, "lora_modules": {"a": "x"}})


def fake_adapter(path: Path, rank: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(json.dumps({"r": rank}))
    (path / "adapter_model.safetensors").write_bytes(b"\0" * (rank * 8))


def test_adapters_are_packed_reproducibly_and_served_with_lora_flags(tmp_path: Path) -> None:
    fake_adapter(tmp_path / "a", 4)
    fake_adapter(tmp_path / "b", 20)
    sha_a = lora_run.pack_adapter(tmp_path / "a", tmp_path / "out/a.tar")
    assert lora_run.pack_adapter(tmp_path / "a", tmp_path / "out/a2.tar") == sha_a
    with tarfile.open(tmp_path / "out/a.tar") as tar:
        assert sorted(tar.getnames()) == ["adapter_config.json", "adapter_model.safetensors"]
        assert {m.mtime for m in tar.getmembers()} == {0}
    server = Server(
        name="base",
        engine="vllm",
        model="Qwen/Q",
        revision="abc",
        args=["--max-model-len", "16384"],
        lora_modules={"a": Path("out/a.tar"), "b": Path("b")},  # a .tar and a directory
        variants=[{"name": "a", "model": "a"}],
    )
    served, info = with_lora(server, Path("unpacked"), tmp_path)
    args = served.args
    assert args[:2] == ["--max-model-len", "16384"]
    assert args[args.index("--max-loras") + 1] == "2"
    assert args[args.index("--max-lora-rank") + 1] == "32"  # smallest vLLM rank >= 20
    modules = args[args.index("--lora-modules") + 1 :]
    assert modules == [
        f"a={(tmp_path / 'unpacked/a').resolve()}",
        f"b={(tmp_path / 'b').resolve()}",
    ]
    assert (tmp_path / "unpacked/a/adapter_config.json").is_file()
    assert info["a"] == {"path": "out/a.tar", "sha256": sha_a, "rank": 4}
    assert info["b"]["sha256"] is None
    assert 32 in VLLM_LORA_RANKS
    assert (
        with_lora(server.model_copy(update={"lora_modules": {}, "variants": []}), Path(), tmp_path)[
            1
        ]
        == {}
    )


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


class FakeTrainer:
    """Writes an adapter and metrics; takes `s_per_pass` clock seconds per example-epoch."""

    def __init__(self, clock: Clock, s_per_pass: float, fail: set[str] | None = None) -> None:
        self.clock = clock
        self.s_per_pass = s_per_pass
        self.fail = fail or set()
        self.specs: list[dict[str, Any]] = []

    def __call__(self, settings: Any, spec_path: Path, log_path: Path, timeout_s: float) -> Any:
        spec = json.loads(spec_path.read_text())
        self.specs.append(spec)
        lines = Path(spec["train_file"]).read_text().splitlines()
        self.clock.t += self.s_per_pass * len(lines) * spec["train"]["epochs"]
        if Path(spec["out_dir"]).parent.name in self.fail:
            raise RuntimeError("trainer exited with 1:\nCUDA out of memory")
        fake_adapter(Path(spec["out_dir"]), spec["rank"])
        return {"train_s": 1.0, "examples": {"train": len(lines)}}


class NoStart:
    """Records the server it was built for; its start fails (no engine in unit tests)."""

    def __init__(self) -> None:
        self.servers: list[Server] = []

    def __call__(self, server: Server, *args: Any) -> Any:
        self.servers.append(server)
        return self

    base_url = "http://127.0.0.1:1/v1"

    def start(self) -> dict[str, Any]:
        raise ServerStartError("no engine in unit tests")

    def metrics_text(self) -> str | None:
        return None

    def stop(self) -> None:
        return None


def lora_root(tmp_path: Path, qs: QuerySettings, n_pairs: int, runs: list[dict[str, Any]]) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    for d in ("configs", "eval"):
        (root / d).symlink_to(REPO_ROOT / d)
    catalogs = load_catalogs(CATALOGS)
    now = catalogs["transit"][1].isoformat()
    pairs = root / "pairs.jsonl"
    pairs.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"c{k:05d}",
                    "state": "transit",
                    "now": now,
                    "question": f"q{k}",
                    "plan": {"intent": "abstain", "abstain_reason": "ambiguous"},
                }
            )
            + "\n"
            for k in range(n_pairs)
        )
    )
    scenario = {
        "name": "lora-t",
        "kind": "lora",
        "seed": 1,
        "base": {"model": "Qwen/Q", "revision": "abc", "served_name": "q"},
        "data": {"pairs": "pairs.jsonl", "catalogs": str(CATALOGS), "prompt": str(COMPACT)},
        "train": {"epochs": 2},
        "baseline": runs[0]["name"],
        "runs": runs,
        "train_budget_minutes": 10,
        "serve": {
            "catalog": "eval/structured/catalog.v1.json",
            "base_prompts": {"base-full": "configs/prompts/planner.v1.txt"},
        },
    }
    (root / "s.yaml").write_text(json.dumps(scenario))
    return root


def test_runs_train_in_order_and_one_that_would_pass_the_budget_is_skipped(
    tmp_path: Path, qs: QuerySettings
) -> None:
    runs = [
        {"name": "small", "pairs": 20, "rank": 8},
        {"name": "all", "rank": 8},
        {"name": "big", "rank": 64},
    ]
    root = lora_root(tmp_path, qs, 100, runs)
    clock = Clock()
    # 3 s per example-epoch: small takes 120 s; all (95 train pairs, 2 epochs) is predicted at
    # 570 s, and 120 + 570 s is past the 600 s budget, so it and big are skipped.
    trainer = FakeTrainer(clock, 3.0)
    server = NoStart()
    path = lora_run.run_lora(root / "s.yaml", root, trainer=trainer, factory=server, clock=clock)
    result = json.loads(path.read_text())
    m = result["metrics"]
    assert result["kind"] == "bench.lora"
    assert (m["data"]["train"], m["data"]["val"]) == (95, 5)
    assert m["runs"]["small"]["pairs"] == 20
    assert "adapter_sha256" in m["runs"]["small"]
    assert "budget" in m["runs"]["all"]["skipped"]  # 120 s spent + 570 s predicted > 600 s
    assert "budget" in m["runs"]["big"]["skipped"]
    assert [s["rank"] for s in trainer.specs] == [8]
    assert trainer.specs[0]["alpha"] == 16.0  # alpha_per_rank 2
    # Size runs train on a prefix of one seeded order.
    small = (root / "data/lora/lora-t/small/train.jsonl").read_text().splitlines()
    assert len(small) == 20
    # One server: the base with the trained adapter, the control and one variant per adapter.
    (built,) = server.servers
    assert set(built.lora_modules) == {"small"}
    assert [(v.name, v.model) for v in built.variants] == [("base-full", None), ("small", "small")]
    assert m["servers"]["q"]["ok"] is False
    assert (root / "models/lora/lora-t-small.tar").is_file()


def test_a_failed_run_is_recorded_without_an_adapter_and_the_rest_train(
    tmp_path: Path, qs: QuerySettings
) -> None:
    runs = [{"name": "a", "pairs": 10, "rank": 8}, {"name": "b", "pairs": 10, "rank": 16}]
    root = lora_root(tmp_path, qs, 40, runs)
    clock = Clock()
    server = NoStart()
    path = lora_run.run_lora(
        root / "s.yaml",
        root,
        trainer=FakeTrainer(clock, 0.1, fail={"a"}),
        factory=server,
        clock=clock,
    )
    m = json.loads(path.read_text())["metrics"]
    assert "CUDA out of memory" in m["runs"]["a"]["error"]
    assert "adapter" not in m["runs"]["a"]
    assert "adapter_sha256" in m["runs"]["b"]
    assert set(server.servers[0].lora_modules) == {"b"}


def test_no_adapter_means_no_server(tmp_path: Path, qs: QuerySettings) -> None:
    root = lora_root(tmp_path, qs, 20, [{"name": "a", "pairs": 5, "rank": 8}])
    server = NoStart()
    clock = Clock()
    path = lora_run.run_lora(
        root / "s.yaml", root, trainer=FakeTrainer(clock, 0.1, {"a"}), factory=server, clock=clock
    )
    assert json.loads(path.read_text())["metrics"]["servers"] == {}
    assert server.servers == []


def recorded_file(tmp_path: Path, kind: str, split: str) -> Path:
    data = {
        "kind": kind,
        "run_id": "r",
        "git": {},
        "hardware": {},
        "config": {"split": split},
        "metrics": {
            "servers": {
                "q": {
                    "ok": True,
                    "engine": "vllm",
                    "model": "Qwen/Q",
                    "revision": "abc",
                    "variants": {
                        "all-r16": {
                            "record": {
                                "replies": {},
                                "call_latency_ms": {},
                                "model": "all-r16",
                                "prompt": str(COMPACT),
                            }
                        }
                    },
                }
            }
        },
    }
    path = tmp_path / f"{kind}.json"
    path.write_text(json.dumps(data))
    return path


def test_replay_of_a_lora_run_builds_messages_from_the_recorded_prompt(tmp_path: Path) -> None:
    replay = load_replay(recorded_file(tmp_path, "bench.lora", "test"), "q", None)
    assert replay.planner_prompt == COMPACT
    assert replay.source["split"] == "test"
    assert replay.source["request_model"] == "all-r16"
    with pytest.raises(SystemExit, match="recorded replies"):
        load_replay(recorded_file(tmp_path, "bench.other", "dev"), "q", None)


def test_example_records_are_chat_messages_plus_target(tmp_path: Path, qs: QuerySettings) -> None:
    ex = examples(2, qs)
    sha = write_jsonl(tmp_path / "x.jsonl", ex)
    rows = [json.loads(line) for line in (tmp_path / "x.jsonl").read_text().splitlines()]
    assert [m["role"] for m in rows[0]["messages"]] == ["system", "user"]
    assert rows[0]["completion"] == '{"intent": "abstain", "abstain_reason": "unanswerable"}'
    assert len(sha) == 64
