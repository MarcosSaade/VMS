"""Merged models for a `kind: llm` scenario (task 8.4): the pinned base with one LoRA adapter
merged into its weights, built on the pod before any server starts and served like any model.

The merge itself runs in vLLM's virtualenv (`scripts/lora_merge.py`, torch and peft), so this
module only writes its spec, runs it as a child process and reads back its metrics file. The
pinned planner adapter is refused at any other hash, as when it is served (D42). A merge that
fails is recorded with its log tail; the servers that need it are then recorded as failed and
the scenario moves on.
"""

from __future__ import annotations

import json
import subprocess
import tarfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vms.common.logs import get_logger
from vms.serving.scenario import LlmScenario
from vms.serving.settings import ServingSettings

log = get_logger(__name__)

MERGE_ERRORS = (RuntimeError, OSError, ValueError, tarfile.TarError, subprocess.TimeoutExpired)

# (python, script, spec file, log file, timeout seconds) -> the script's metrics
ScriptRunner = Callable[[str, Path, Path, Path, float], dict[str, Any]]


def run_python_script(
    python: str, script: Path, spec_path: Path, log_path: Path, timeout_s: float
) -> dict[str, Any]:
    """Run `python script spec.json` (output appended to `log_path`); the metrics file the
    spec names, or a RuntimeError with the log tail."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    with log_path.open("ab") as out:
        proc = subprocess.run(
            [python, str(script), str(spec_path)],
            stdout=out,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )
    if proc.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]
        raise RuntimeError(f"{script.name} exited with {proc.returncode}:\n" + "\n".join(tail))
    metrics: dict[str, Any] = json.loads(Path(spec["metrics_file"]).read_text(encoding="utf-8"))
    return metrics


def build_merges(
    scenario: LlmScenario,
    settings: ServingSettings,
    root: Path,
    runner: ScriptRunner = run_python_script,
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Build every merge of the scenario: (name -> model directory for the ones that worked,
    name -> what was recorded for each)."""
    # Imported here: vms.serving.run imports this module.
    from vms.serving.run import check_pinned, sha256_file, unpack_adapter

    built: dict[str, Path] = {}
    entries: dict[str, Any] = {}
    lora = settings.lora
    for merge in scenario.merges:
        out = root / lora.merged_dir / merge.name
        entry: dict[str, Any] = {
            "model": merge.model,
            "revision": merge.revision,
            "adapter": str(merge.adapter),
        }
        try:
            src = root / merge.adapter
            check_pinned(merge.adapter, src, lora.planner)
            entry["adapter_sha256"] = sha256_file(src) if src.is_file() else None
            adapter = unpack_adapter(src, root / lora.unpack_dir, f"merge-{merge.name}")
            spec = {
                "model": merge.model,
                "revision": merge.revision,
                "adapter_dir": str(adapter.resolve()),
                "out_dir": str(out.resolve()),
                "metrics_file": str((out.parent / f"{merge.name}.metrics.json").resolve()),
            }
            spec_path = out.parent / f"{merge.name}.spec.json"
            spec_path.parent.mkdir(parents=True, exist_ok=True)
            spec_path.write_text(json.dumps(spec, indent=1) + "\n", encoding="utf-8")
            log_path = root / scenario.logs_dir / f"merge-{merge.name}.log"
            entry["metrics"] = runner(
                lora.python, root / lora.merge_script, spec_path, log_path, lora.merge_timeout_s
            )
            entry["ok"] = True
            built[merge.name] = out
        except MERGE_ERRORS as exc:
            entry["ok"], entry["error"] = False, f"{type(exc).__name__}: {exc}"
            log.error("merge failed", extra={"merge": merge.name, "error": entry["error"]})
        entries[merge.name] = entry
    return built, entries
