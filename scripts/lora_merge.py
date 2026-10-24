#!/usr/bin/env python3
"""Merge one LoRA adapter into its base model's weights (task 8.4): one merge per call.

    python scripts/lora_merge.py SPEC.json

Called by `vms.serving.merge` on a GPU pod, in vLLM's virtualenv (its CUDA torch and
transformers, plus peft; scripts/pod_remote.sh setup llm), so it imports nothing from the
project. The spec (JSON) names the base model and revision, the unpacked adapter directory, the
output model directory and the metrics file.

The base is loaded in FP32 and the adapter merged there (W + scale * B @ A in FP32), then the
result is cast to BF16, the dtype it is served in: one rounding of the merged weight instead of
rounding the low-rank update into BF16 weights. The output is an ordinary Hugging Face model
directory that vLLM serves by path: the merged weights and config from `save_pretrained`, plus
every other non-weight file of the base revision (tokenizer, chat template, processor
configs). The base's separate multi-token-prediction weights are not copied: they are used only
for speculative decoding, which is not served here. The metrics file gets the time, the
parameter counts, the merged weights' size and sha256 per file, and the package versions.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import shutil
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForImageTextToText

WEIGHTS = ("*.safetensors", "*.safetensors.index.json", "*.bin", "*.pt", "*.pth")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in ("torch", "transformers", "peft", "safetensors"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def main(spec_path: str) -> int:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(spec["out_dir"])
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    t0 = time.perf_counter()
    # Every non-weight file of the pinned base first; save_pretrained then writes the merged
    # weights, their index and the config over them.
    base_files = Path(
        snapshot_download(spec["model"], revision=spec["revision"], ignore_patterns=list(WEIGHTS))
    )
    copied = []
    for path in sorted(base_files.rglob("*")):
        # The snapshot directory also lists files cached by earlier downloads of the revision.
        if path.is_file() and not any(fnmatch.fnmatch(path.name, w) for w in WEIGHTS):
            target = out_dir / path.relative_to(base_files)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            copied.append(str(path.relative_to(base_files)))

    model = AutoModelForImageTextToText.from_pretrained(
        spec["model"], revision=spec["revision"], dtype=torch.float32
    ).to(device)
    base_params = sum(p.numel() for p in model.parameters())
    load_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    model = PeftModel.from_pretrained(model, spec["adapter_dir"])
    adapter_params = sum(p.numel() for n, p in model.named_parameters() if "lora_" in n)
    model = model.merge_and_unload()
    model = model.to(torch.bfloat16)
    merge_s = time.perf_counter() - t1

    t2 = time.perf_counter()
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="5GB")
    save_s = time.perf_counter() - t2

    weights = {
        str(p.relative_to(out_dir)): {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in sorted(out_dir.glob("*.safetensors"))
    }
    metrics: dict[str, Any] = {
        "load_s": round(load_s, 2),
        "merge_s": round(merge_s, 2),
        "save_s": round(save_s, 2),
        "merge_dtype": "float32",
        "saved_dtype": "bfloat16",
        "params": {"base": base_params, "adapter": adapter_params},
        "weights": weights,
        "weights_bytes": sum(w["bytes"] for w in weights.values()),
        "copied_from_base": copied,
        "versions": versions(),
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    }
    Path(spec["metrics_file"]).write_text(json.dumps(metrics, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in metrics.items() if k != "weights"}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
