# Benchmarks

> Location in repo: `docs/BENCHMARKS.md`. Curated current numbers. Every number links to its experiment (`docs/EXPERIMENTS.md`) and results JSON, states the hardware, and labels synthetic data and self-measured results as such. Raw history stays in `EXPERIMENTS.md`.

## Methodology

_Placeholder (task 2.3)._ Covers scenario YAML in `bench/scenarios/`, warmup exclusion, at least 3 repetitions or a steady state, reported statistics (p50, p95, p99, throughput, error rate, utilization, variance), one variable changed per run against a named baseline, and the checks that the harness is not the bottleneck.

## Hardware

One entry per environment used (CPU, RAM, GPU model, driver, CUDA, instance type, Region).

| ID | Machine | Notes |
|---|---|---|
| laptop | Intel i7-1355U (10 cores, 12 threads), 15 GiB RAM, no GPU, Linux 6.8 | Development laptop with other work running; numbers are indicative, not a quiet benchmark host. |

## Headline results

_No numbers yet._ Targets: rated capacity (cameras at 5 FPS per GPU), freshness p95, query p95 per mode, structured-query execution accuracy, retrieval recall@10, cost per camera-hour, drift detection delay.

## Ingest and pipeline capacity

_Rated capacity: placeholder (tasks 2.5, 5.5)._

### Load generator ceiling (harness, not the system)

Self-measured on `laptop`, 1280x720 JPEG q85 frames from `make data-mini`, 3 repetitions each, warmup excluded. These say how much load one loadgen process can offer before it, rather than the system under test, becomes the limit.

| Scenario | What it measures | Frames/s (mean, std) | CPU per frame | Evidence |
|---|---|---|---|---|
| `loadgen-selftest` | Generator core, no I/O | 36,412 (1,754) | 27 us | E-009, `bench/results/2026-10-09-loadgen-selftest-01M4GK6TJTHH0D2T3JW9KHWK28.json` |
| `loadgen-stub-max` | Generator + multipart HTTP over loopback, stub with 6 workers | 1,252 (44) | 0.80 ms | E-010, `bench/results/2026-10-09-loadgen-stub-max-01M4GK893V2X7WP677GQEXG8GW.json` |
| `loadgen-stub-20x5` | 20 cameras x 5 FPS paced, 1-worker stub | 100.0 offered = accepted, 0 dropped; send p50/p95/p99 8.7/17.6/24.1 ms; schedule lag p99 2.3 ms | 1.6 ms | E-011, `bench/results/2026-10-09-loadgen-stub-20x5-01M4GKA7GNXHJX3ABMV91AKYJ1.json` |

One process should offer at most ~70% of 1,250 frames/s (~175 cameras at 5 FPS) with `generator_ok` true. Above that, shard cameras across processes.

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

One `make` or `python -m vms...` command per table.

- Load generator ceiling: `make data-mini`, then `make loadgen-selftest`; `make loadgen-stub STUB_WORKERS=6` in one shell and `make loadgen SCENARIO=bench/scenarios/loadgen-stub-max.yaml` in another; restart the stub with `STUB_WORKERS=1` and run `make loadgen SCENARIO=bench/scenarios/loadgen-stub-20x5.yaml`. Check that port 8080 is free before starting each stub.
