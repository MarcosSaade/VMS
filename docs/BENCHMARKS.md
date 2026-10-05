# Benchmarks

> Location in repo: `docs/BENCHMARKS.md`. Curated current numbers. Every number links to its experiment (`docs/EXPERIMENTS.md`) and results JSON, states the hardware, and labels synthetic data and self-measured results as such. Raw history stays in `EXPERIMENTS.md`.

## Methodology

_Placeholder (task 2.3)._ Covers scenario YAML in `bench/scenarios/`, warmup exclusion, at least 3 repetitions or a steady state, reported statistics (p50, p95, p99, throughput, error rate, utilization, variance), one variable changed per run against a named baseline, and the checks that the harness is not the bottleneck.

## Hardware

_Placeholder._ One entry per environment used (CPU, RAM, GPU model, driver, CUDA, instance type, Region).

## Headline results

_No numbers yet._ Targets: rated capacity (cameras at 5 FPS per GPU), freshness p95, query p95 per mode, structured-query execution accuracy, retrieval recall@10, cost per camera-hour, drift detection delay.

## Ingest and pipeline capacity

_Placeholder (tasks 2.5, 5.5, 6.6)._

## Query latency

_Placeholder (task 2.5)._

## Storage

_Placeholder (tasks 4.3, 4.4)._

## Inference

_Placeholder (tasks 5.2, 5.3)._ Model-only and end-to-end numbers are reported separately, with mAP and recall@k next to speed.

## Evaluation

_Placeholder (tasks 3.5, 7.3, 8.5)._

## LLM serving

_Placeholder (tasks 8.1, 8.4)._

## Drift detection

_Placeholder (tasks 9.3, 9.5)._

## Cost

_Placeholder (task 10.3)._

## How to reproduce

_Placeholder._ One `make` or `python -m vms...` command per table.
