#!/usr/bin/env python3
"""LoRA fine-tune of the planner on chat examples (task 8.3, D42): one run per call.

    python scripts/lora_train.py SPEC.json

Called by `vms.lora.run` on a GPU pod, in vLLM's virtualenv (its CUDA torch and transformers,
plus peft; scripts/pod_remote.sh setup llm), so it imports nothing from the project. The spec
(JSON) names the base model and revision, the train and validation files written by
`vms.lora.data`, the hyperparameters, the output adapter directory and the metrics file.

Each example is the planner's messages rendered with the model's own chat template (thinking
off, as served) plus the target plan and the end-of-turn token. The loss covers only the
target tokens and is averaged over every target token of an optimizer step (gradient
accumulation included), not per micro-batch. Batches group examples of similar length to cut
padding. The metrics file gets the loss curve, validation loss per epoch, throughput, peak GPU
memory and the package versions.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import random
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

# Before torch allocates: fewer fragmentation OOMs with the varying batch shapes.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForImageTextToText, AutoTokenizer

IGNORE = -100


def encode(tok: Any, example: dict[str, Any], max_len: int) -> tuple[list[int], list[int]] | None:
    """(input ids, labels) with the prompt masked, or None if longer than `max_len`."""
    prompt = tok.apply_chat_template(
        example["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tok(example["completion"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = prompt_ids + target_ids
    if len(ids) > max_len:
        return None
    return ids, [IGNORE] * len(prompt_ids) + target_ids


def batches(
    data: list[tuple[list[int], list[int]]], size: int, rng: random.Random
) -> list[list[int]]:
    """Index batches of similar length: shuffle, sort within windows of 50 batches, shuffle."""
    order = list(range(len(data)))
    rng.shuffle(order)
    window = size * 50
    out: list[list[int]] = []
    for start in range(0, len(order), window):
        chunk = sorted(order[start : start + window], key=lambda i: len(data[i][0]))
        out += [chunk[k : k + size] for k in range(0, len(chunk), size)]
    rng.shuffle(out)
    return out


def collate(
    data: list[tuple[list[int], list[int]]], idx: list[int], pad: int, device: str
) -> dict[str, torch.Tensor]:
    width = max(len(data[i][0]) for i in idx)
    ids = torch.full((len(idx), width), pad, dtype=torch.long)
    labels = torch.full((len(idx), width), IGNORE, dtype=torch.long)
    mask = torch.zeros((len(idx), width), dtype=torch.long)
    for row, i in enumerate(idx):
        x, y = data[i]
        ids[row, : len(x)] = torch.tensor(x)
        labels[row, : len(y)] = torch.tensor(y)
        mask[row, : len(x)] = 1
    return {
        "input_ids": ids.to(device),
        "labels": labels.to(device),
        "attention_mask": mask.to(device),
    }


def target_tokens(labels: torch.Tensor) -> int:
    """Tokens the shifted loss counts (a label at position 0 is never predicted)."""
    return int((labels[:, 1:] != IGNORE).sum())


def lr_at(step: int, total: int, warmup: int, peak: float) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for name in ("torch", "transformers", "peft", "flash-linear-attention", "causal-conv1d"):
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


def fit_micro_batch(
    model: Any,
    data: list[tuple[list[int], list[int]]],
    micro: int,
    accum: int,
    pad: int,
    device: str,
) -> tuple[int, int, list[int]]:
    """Largest micro-batch, at most `micro`, whose forward and backward on the longest
    examples fit in GPU memory; the accumulation grows to keep the effective batch. Returns
    (micro-batch, accumulation, the sizes that ran out of memory)."""
    longest = sorted(range(len(data)), key=lambda i: -len(data[i][0]))
    failed: list[int] = []
    while True:
        batch = collate(data, longest[:micro], pad, device)
        try:
            out = model(**batch, num_items_in_batch=target_tokens(batch["labels"]))
            out.loss.backward()
            model.zero_grad(set_to_none=True)
            return micro, accum, failed
        except torch.OutOfMemoryError:
            model.zero_grad(set_to_none=True)
            del batch
            torch.cuda.empty_cache()
            if micro == 1:
                raise
            failed.append(micro)
            micro, accum = micro // 2, accum * 2


@torch.no_grad()
def eval_loss(
    model: Any, data: list[tuple[list[int], list[int]]], size: int, pad: int, device: str
) -> float:
    model.eval()
    total, count = 0.0, 0
    for start in range(0, len(data), size):
        batch = collate(data, list(range(start, min(start + size, len(data)))), pad, device)
        n = target_tokens(batch["labels"])
        out = model(**batch, num_items_in_batch=n)
        total += float(out.loss.detach()) * n
        count += n
    model.train()
    return total / max(1, count)


def main(spec_path: str) -> int:
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    hp = spec["train"]
    seed = int(spec["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(spec["model"], revision=spec.get("revision"))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    encoded = {}
    dropped = {}
    for split in ("train", "val"):
        rows = [encode(tok, e, hp["max_len"]) for e in read_jsonl(spec[f"{split}_file"])]
        encoded[split] = [r for r in rows if r is not None]
        dropped[split] = sum(r is None for r in rows)
    train, val = encoded["train"], encoded["val"]

    model = AutoModelForImageTextToText.from_pretrained(
        spec["model"], revision=spec.get("revision"), dtype=dtype, attn_implementation="sdpa"
    ).to(device)
    model.config.use_cache = False
    if hp["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    lora = LoraConfig(
        r=spec["rank"],
        lora_alpha=spec["alpha"],
        lora_dropout=hp["dropout"],
        target_modules=hp["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=hp["learning_rate"], weight_decay=hp["weight_decay"])
    load_s = time.perf_counter() - t0

    micro_fails: list[int] = []
    if device == "cuda":
        hp["micro_batch"], hp["grad_accum"], micro_fails = fit_micro_batch(
            model, train, hp["micro_batch"], hp["grad_accum"], pad, device
        )
    rng = random.Random(seed)
    per_step = hp["micro_batch"] * hp["grad_accum"]
    steps_per_epoch = math.ceil(len(train) / per_step)
    total_steps = steps_per_epoch * hp["epochs"]
    warmup = max(1, round(total_steps * hp["warmup_ratio"]))
    curve: list[dict[str, float]] = []
    val_losses = [{"epoch": 0, "loss": eval_loss(model, val, hp["micro_batch"], pad, device)}]
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    step = 0
    seen_tokens = 0
    train_s = 0.0
    model.train()
    for epoch in range(1, hp["epochs"] + 1):
        groups = batches(train, hp["micro_batch"], rng)
        for s in range(0, len(groups), hp["grad_accum"]):
            t_step = time.perf_counter()
            micro = [collate(train, idx, pad, device) for idx in groups[s : s + hp["grad_accum"]]]
            n = sum(target_tokens(b["labels"]) for b in micro)
            loss_sum = 0.0
            for b in micro:
                out = model(**b, num_items_in_batch=n)
                out.loss.backward()
                loss_sum += float(out.loss.detach())
                seen_tokens += int(b["attention_mask"].sum())
            torch.nn.utils.clip_grad_norm_(params, hp["max_grad_norm"])
            lr = lr_at(step, total_steps, warmup, hp["learning_rate"])
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if device == "cuda":
                torch.cuda.synchronize()
            train_s += time.perf_counter() - t_step
            step += 1
            if step % spec.get("log_every", 10) == 0 or step == total_steps:
                point = {
                    "step": step,
                    "epoch": epoch,
                    "loss": round(loss_sum, 5),
                    "lr": lr,
                    "tokens_per_s": round(seen_tokens / train_s, 1),
                }
                curve.append(point)
                print(json.dumps(point), flush=True)
        val_losses.append(
            {"epoch": epoch, "loss": eval_loss(model, val, hp["micro_batch"], pad, device)}
        )
        print(json.dumps({"val": val_losses[-1]}), flush=True)

    out_dir = Path(spec["out_dir"])
    model.save_pretrained(out_dir)
    metrics = {
        "examples": {"train": len(train), "val": len(val)},
        "dropped_too_long": dropped,
        "tokens": {
            "train_per_epoch": sum(len(x) for x, _ in train),
            "train_targets_per_epoch": sum(target_tokens(torch.tensor([y])) for _, y in train),
            "seen": seen_tokens,
            "max_len_seen": max(len(x) for x, _ in train),
        },
        "params": {"trainable": trainable, "total": total_params},
        "micro_batch": hp["micro_batch"],
        "grad_accum": hp["grad_accum"],
        "micro_batch_out_of_memory": micro_fails,
        "steps": step,
        "steps_per_epoch": steps_per_epoch,
        "warmup_steps": warmup,
        "load_s": round(load_s, 2),
        "train_s": round(train_s, 2),
        "tokens_per_s": round(seen_tokens / train_s, 1) if train_s else None,
        "peak_gpu_memory_mib": (
            round(torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else None
        ),
        "loss_curve": curve,
        "val_loss": val_losses,
        "versions": versions(),
        "fast_linear_attention": importlib.util.find_spec("fla") is not None,
        "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    }
    Path(spec["metrics_file"]).write_text(json.dumps(metrics, indent=1) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
