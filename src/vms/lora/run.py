"""Run a `kind: lora` scenario (task 8.3, D42) on a GPU pod: render the training examples,
train each run's adapter (`scripts/lora_train.py` in vLLM's environment), pack it, then serve
the base with every trained adapter on one vLLM server and record the dev split for each
adapter and for the base controls. One results JSON (kind `bench.lora`) whose recorded
replies the laptop replays through the unchanged eval (`make eval-replay`, D40).

Runs train in the listed order. After the first, the measured seconds per training example
predict each next run; a run that would push training past `train_budget_minutes` is skipped
and recorded as skipped, so the session's time limit is never what ends a run.
"""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from vms.bench.results import RunResult, git_info, hardware, package_versions, write_result
from vms.common.ids import new_ulid
from vms.common.logs import get_logger
from vms.common.settings import load_settings
from vms.lora.data import (
    load_catalogs,
    read_pairs,
    render_examples,
    split_examples,
    with_prompt,
    write_jsonl,
)
from vms.lora.scenario import LoraScenario, Run, load_spec_load
from vms.query.settings import QuerySettings
from vms.serving.merge import run_python_script
from vms.serving.run import Context, run_config, serve_all, sha256_file
from vms.serving.scenario import LlmScenario, Server, Variant
from vms.serving.servers import build_server
from vms.serving.settings import ServingSettings

log = get_logger(__name__)

KIND = "bench.lora"
PACKAGES = ("aiohttp", "pydantic")  # the trainer reports its own (torch, transformers, peft)

Trainer = Callable[[ServingSettings, Path, Path, float], dict[str, Any]]


def load_scenario(path: Path) -> tuple[LoraScenario, dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return LoraScenario.model_validate(raw), raw


def subprocess_trainer(
    settings: ServingSettings, spec_path: Path, log_path: Path, timeout_s: float
) -> dict[str, Any]:
    """Run the trainer script on one spec; its metrics file, or an error to record."""
    return run_python_script(
        settings.lora.python, settings.lora.script, spec_path, log_path, timeout_s
    )


def pack_adapter(adapter_dir: Path, tar_path: Path) -> str:
    """A reproducible .tar of the adapter directory (sorted, zero mtimes and owners)."""
    tar_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(p for p in adapter_dir.rglob("*") if p.is_file()):
            info = tarfile.TarInfo(str(path.relative_to(adapter_dir)))
            data = path.read_bytes()
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tar.addfile(info, io.BytesIO(data))
    tar_path.write_bytes(buf.getvalue())
    return sha256_file(tar_path)


def run_spec(
    scenario: LoraScenario, run: Run, train_file: Path, val_file: Path, out_dir: Path
) -> dict[str, Any]:
    return {
        "model": scenario.base.model,
        "revision": scenario.base.revision,
        "seed": scenario.seed,
        "rank": run.rank,
        "alpha": scenario.train.alpha_per_rank * run.rank,
        "train": scenario.train.model_dump(mode="json"),
        "train_file": str(train_file),
        "val_file": str(val_file),
        "out_dir": str(out_dir / "adapter"),
        "metrics_file": str(out_dir / "metrics.json"),
    }


def serving_scenario(scenario: LoraScenario, adapters: dict[str, Path]) -> LlmScenario:
    """One vLLM server: the base with every trained adapter; record-only variants."""
    serve = scenario.serve
    variants = [
        Variant(name=name, prompt=prompt, load=False, record=True)
        for name, prompt in serve.base_prompts.items()
    ]
    variants += [
        Variant(name=name, model=name, prompt=scenario.data.prompt, load=False, record=True)
        for name in adapters
    ]
    server = Server(
        name=scenario.base.served_name,
        engine="vllm",
        model=scenario.base.model,
        revision=scenario.base.revision,
        args=serve.args,
        request_extra=dict(serve.request_extra),
        start_timeout_s=serve.start_timeout_s,
        lora_modules=adapters,
        variants=variants,
    )
    return LlmScenario(
        name=scenario.name,
        seed=scenario.seed,
        catalog=serve.catalog,
        max_tokens=serve.max_tokens,
        results_dir=scenario.results_dir,
        logs_dir=scenario.logs_dir,
        load=load_spec_load(),
        record=serve.record,
        servers=[server],
    )


def run_lora(
    scenario_path: Path,
    root: Path = Path(),
    *,
    trainer: Trainer = subprocess_trainer,
    factory: Any = build_server,
    clock: Callable[[], float] = time.monotonic,
) -> Path:
    scenario, raw = load_scenario(scenario_path)
    settings = load_settings(ServingSettings, "serving", config_dir=root / "configs")
    qs = load_settings(QuerySettings, "query", config_dir=root / "configs")
    started = datetime.now(UTC)
    work = root / settings.lora.work_dir / scenario.name

    examples = render_examples(
        read_pairs(root / scenario.data.pairs),
        load_catalogs(root / scenario.data.catalogs),
        with_prompt(qs, scenario.data.prompt),
        root,
    )
    train, val = split_examples(examples, scenario.data.val_fraction, scenario.seed)
    val_file = work / "val.jsonl"
    data: dict[str, Any] = {
        "pairs": len(examples),
        "pairs_sha256": sha256_file(root / scenario.data.pairs),
        "catalogs_sha256": sha256_file(root / scenario.data.catalogs),
        "prompt": str(scenario.data.prompt),
        "prompt_sha256": sha256_file(root / scenario.data.prompt),
        "train": len(train),
        "val": len(val),
        "val_sha256": write_jsonl(val_file, val),
        "system_prompt_chars": {e.state: len(e.system) for e in examples},
    }

    budget_s = scenario.train_budget_minutes * 60.0
    spent_s = 0.0
    s_per_example: float | None = None  # wall seconds per training example-epoch, measured
    runs: dict[str, Any] = {}
    adapters: dict[str, Path] = {}
    for run in scenario.runs:
        n = min(run.pairs or len(train), len(train))
        passes = n * scenario.train.epochs
        entry: dict[str, Any] = {"pairs": n, "rank": run.rank, "requested_pairs": run.pairs}
        if s_per_example is not None and spent_s + s_per_example * passes > budget_s:
            entry["skipped"] = (
                f"predicted {s_per_example * passes / 60:.1f} min; "
                f"{(budget_s - spent_s) / 60:.1f} min of the training budget left"
            )
            runs[run.name] = entry
            log.warning("run skipped", extra={"run": run.name, "reason": entry["skipped"]})
            continue
        out_dir = work / run.name
        train_file = out_dir / "train.jsonl"
        entry["train_sha256"] = write_jsonl(train_file, train[:n])
        spec = run_spec(scenario, run, train_file, val_file, out_dir)
        spec_path = out_dir / "spec.json"
        spec_path.write_text(json.dumps(spec, indent=1) + "\n", encoding="utf-8")
        entry["spec"] = spec
        t0 = clock()
        try:
            # The budget bounds the run too: anything past the rest of it plus a margin hung.
            timeout = max(budget_s - spent_s, 0.0) + 1800.0
            entry["metrics"] = trainer(
                settings, spec_path, root / scenario.logs_dir / f"lora-{run.name}.log", timeout
            )
        except (RuntimeError, OSError, subprocess.TimeoutExpired, ValueError) as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
            log.error("run failed", extra={"run": run.name, "error": entry["error"]})
        wall = clock() - t0
        spent_s += wall
        entry["wall_s"] = round(wall, 2)
        if "metrics" in entry:
            s_per_example = wall / passes
            tar = root / scenario.adapters_dir / f"{scenario.name}-{run.name}.tar"
            entry["adapter"] = str(tar.relative_to(root))
            entry["adapter_sha256"] = pack_adapter(out_dir / "adapter", tar)
            adapters[run.name] = tar.relative_to(root)
        runs[run.name] = entry
        log.info("run done", extra={"run": run.name, "wall_s": entry["wall_s"]})

    earlier = {n: p for n, p in scenario.serve.adapters.items() if (root / p).is_file()}
    missing = sorted(set(scenario.serve.adapters) - set(earlier))
    if missing:
        log.error("earlier adapters missing", extra={"adapters": missing})
    if adapters:  # this session's runs first, then the earlier ones
        adapters = {**adapters, **earlier}
    servers: dict[str, Any] = {}
    config: dict[str, Any] = {"serving": settings.model_dump(mode="json")}
    if adapters:
        ctx = Context(serving_scenario(scenario, adapters), root)
        servers = serve_all(ctx, settings, root, factory)
        config = run_config(ctx, settings)
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
        config=config,
        scenario=raw,
        metrics={
            "data": data,
            "baseline": scenario.baseline,
            "baseline_scenario": scenario.baseline_scenario,
            "earlier_adapters_missing": missing,
            "train_spent_s": round(spent_s, 2),
            "runs": runs,
            "servers": servers,
        },
        notes="SYNTHETIC TRAINING DATA (8.2). Adapters trained on the pod; the dev split is "
        "recorded per adapter and for the base controls on one vLLM server and scored on the "
        "laptop by `python -m vms.eval run --planner replay` (D40, D42). Training time is "
        "wall time on the pod, model download and load included in wall_s, not in train_s.",
    )
    path = write_result(root / scenario.results_dir, result)
    log.info("lora scenario written", extra={"path": str(path)})
    return path
