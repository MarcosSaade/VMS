# Benchmarks

> Location in repo: `docs/BENCHMARKS.md`. Curated current numbers. Every number links to its experiment (`docs/EXPERIMENTS.md`) and results JSON, states the hardware, and labels synthetic data and self-measured results as such. Raw history stays in `EXPERIMENTS.md`.

## Methodology

Applies to every table below unless it says otherwise.

- **Scenarios first.** Each number comes from a scenario in `bench/scenarios/`, written before the run and run with `make bench` (RUNBOOK, "Benchmarks"). Results files in `bench/results/` record the git SHA and dirty flag, the resolved config, the scenario as written, hardware, package versions, seeds and start and end times.
- **Steady state.** Every step runs `warmup_s + hold_s`; only the last `hold_s` counts. Loadgen drops warmup samples itself; query samples are kept only if they started inside the window; server-side numbers are PromQL over exactly that window (`rate(...[hold_s])` at the window's end, read after one more scrape). Before each ingest step the worker backlog is drained to zero, so a step does not inherit the previous one's queue.
- **Repetitions and variance.** At least 3 repetitions; each repetition is a full pass over the steps, so slow drift of the host becomes variance rather than a trend across steps. Tables give mean (sample std) across repetitions; min and max are in the results file.
- **Statistics.** Client-side latencies (loadgen send latency, query latency) are exact percentiles of raw samples. Server-side quantiles (freshness, stage latencies) come from Prometheus histogram buckets with `histogram_quantile`, so they are interpolated within a bucket; the freshness buckets have an edge at 2 s, so the SLO verdict is exact.
- **Rated capacity.** A step meets the ingest SLO when freshness p95 < 2 s, zero frames are lost (dropped, rejected or failed at ingest, or lost by the worker) and the backlog slope is at most 0.5 frames/s (ARCHITECTURE 2, D27). Rated capacity is the largest camera count at 5 FPS that, with every smaller step, met it in every repetition; if the largest step passed, it is a lower bound.
- **Query load** is closed loop: `concurrency` clients each ask their next question when the last answer arrives. Achieved queries/s is reported next to latency and is not a capacity.
- **The harness is not the bottleneck.** Loadgen runs in its own process and flags `generator_ok: false` when its schedule lag p99 exceeds 10 ms (its ceilings are below); the query clients' CPU is recorded and a step is flagged `harness_ok: false` above 0.8 cores. Flagged steps are not reported as numbers.
- **One variable at a time** against a named baseline scenario; only results from the same hardware are compared, and the hardware ID is stated with every table.
- **Labels.** Data is public video replayed as virtual cameras (synthetic load); every number is self-measured.

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

_Precision ladder and batch sweep: placeholder (tasks 5.2, 5.3)._ Model-only and end-to-end numbers are reported separately, with mAP and recall@k next to speed.

### Worker model-only latency, CPU baseline

Self-measured on `laptop` with `make bench-model`: the worker's per-frame work in its own thread (JPEG decode, input statistics, YOLO11 pre-processing + forward + NMS; tracking is not included), batch size 1, imgsz 640, detector threshold 0.1 (the tracking default, D21), PyTorch 2.14.1 CPU build, 150 frames from the three data-mini clips at 1280x720 q85, 10 warmup frames excluded, 3 repetitions. No database, object store or HTTP, so end-to-end throughput will be lower (task 2.5). No accuracy number yet: mAP comes in task 3.4, so the yolo11n speedup has an unmeasured quality cost.

| Model | Frames/s (mean, std) | Infer p50 / p95 ms | Total p95 ms | CPU cores | Raw detections/frame | Evidence |
|---|---|---|---|---|---|---|
| yolo11s | 9.8 (0.22) | 89 / 105 | 119 | ~7.5 | 27.8 | E-016, `bench/results/2026-10-12-worker-model-cpu-01M4QVF9ZAKC9HXMTFERF164VV.json` |
| yolo11n | 18.8 (1.9) | 40 / 47 | 62 | ~7.5 | 23.2 | E-016, `bench/results/2026-10-12-worker-model-cpu-n-01M4QVGTE15P83D77JXS71K9VF.json` |

At the earlier threshold of 0.25 the numbers were the same within noise (E-013, E-014).

At 5 FPS per camera, one CPU worker keeps up with about 2 cameras on yolo11s and about 4 on yolo11n, before the database and object store are added. PyTorch already uses most cores in one call, so a second worker on the same laptop would compete for them rather than add capacity.

## Evaluation

_No accuracy numbers yet (tasks 3.2, 3.5)._

### Structured-query eval set v1 (task 3.1, D29, E-029)

Questions over a **synthetic** frozen database (`configs/eval/snapshot.v1.yaml`, seed 20260331): five cameras (four in London, one in New York), 2026-03-16 to the eval clock **2026-03-31 14:30 UTC**, which every run must inject. Each answerable item has a gold SQL and, where the plan schema can express it, a gold `QueryPlan`; both were executed on the snapshot as the read-only login and agree. Items were drafted by the coding agent and reviewed by the user.

| Category | Dev | Test |
|---|---|---|
| count | 7 | 8 |
| aggregate | 4 | 4 |
| list | 4 | 4 |
| time_window | 6 | 6 |
| join | 6 | 5 |
| trend | 5 | 5 |
| top_k | 4 | 5 |
| dwell | 4 | 4 |
| frames | 3 | 3 |
| ambiguous | 3 | 4 |
| unanswerable | 3 | 4 |
| unsupported (SQL answer, outside the plan) | 7 | 8 |
| adversarial (red team: DDL, DML, injection, exfiltration, exhaustion, prompt) | 6 | 7 |
| **total** | **62** | **67** |

- **Frozen test split:** `eval/structured/test.v1.jsonl`, sha256 `5560a5fbb3803ae0c1458ae98eb363a44455ae2e1cab6936c4867e8c58a5059b` (also in `eval/structured/dataset.v1.json`). Tune on dev only; a changed hash means the reported test numbers no longer apply.
- **Snapshot:** `eval/structured/snapshot.v1/` (DVC), file hashes in `eval/structured/snapshot.v1.json`.
- **Precision to expect:** with 67 test questions, a 95% interval is about plus or minus 11 points at 70% accuracy; differences smaller than that between two planners are not evidence on their own.

## LLM serving

_Placeholder (tasks 8.1, 8.4)._

## Drift detection

_Placeholder (tasks 9.3, 9.5)._

## Cost

_Placeholder (task 10.3)._

## How to reproduce

One `make` or `python -m vms...` command per table. `make bench SCENARIO=bench/scenarios/<name>.yaml` reruns any scenario named in a table (`make demo` first for end-to-end ones; `make loadgen-stub` for the stub scenarios; `make data-mini` and `make models` for the model-only ones), and `make bench-report RESULTS=...` renders a results file as the table rows above.

- Structured eval set: `make up`, `make eval-db` (pulls or regenerates the snapshot and checks its hashes), `make eval-build` (re-runs every gold; the split files must come out unchanged), `make eval-check`.
- Load generator ceiling: `make data-mini`, then `make loadgen-selftest`; `make loadgen-stub STUB_WORKERS=6` in one shell and `make loadgen SCENARIO=bench/scenarios/loadgen-stub-max.yaml` in another; restart the stub with `STUB_WORKERS=1` and run `make loadgen SCENARIO=bench/scenarios/loadgen-stub-20x5.yaml`. Check that port 8080 is free before starting each stub.
