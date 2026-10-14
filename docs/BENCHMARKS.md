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

Self-measured on `laptop` with `make bench-model`: the worker's per-frame work in its own thread (JPEG decode, input statistics, YOLO11 pre-processing + forward + NMS; tracking is not included), batch size 1, imgsz 640, detector threshold 0.1 (the tracking default, D21), PyTorch 2.14.1 CPU build, 150 frames from the three data-mini clips at 1280x720 q85, 10 warmup frames excluded, 3 repetitions. No database, object store or HTTP, so end-to-end throughput will be lower (task 2.5). Its quality cost: yolo11n scores about 8 points lower COCO mAP@50-95 and 5 points lower MOT17 person mAP@50-95 than yolo11s (Evaluation, detection quality).

| Model | Frames/s (mean, std) | Infer p50 / p95 ms | Total p95 ms | CPU cores | Raw detections/frame | Evidence |
|---|---|---|---|---|---|---|
| yolo11s | 9.8 (0.22) | 89 / 105 | 119 | ~7.5 | 27.8 | E-016, `bench/results/2026-10-12-worker-model-cpu-01M4QVF9ZAKC9HXMTFERF164VV.json` |
| yolo11n | 18.8 (1.9) | 40 / 47 | 62 | ~7.5 | 23.2 | E-016, `bench/results/2026-10-12-worker-model-cpu-n-01M4QVGTE15P83D77JXS71K9VF.json` |

At the earlier threshold of 0.25 the numbers were the same within noise (E-013, E-014).

At 5 FPS per camera, one CPU worker keeps up with about 2 cameras on yolo11s and about 4 on yolo11n, before the database and object store are added. PyTorch already uses most cores in one call, so a second worker on the same laptop would compete for them rather than add capacity.

## Evaluation

_No accuracy numbers for a real planner yet (task 3.5)._

### Structured-query eval set v1 (task 3.1, D29, E-029)

Questions over a **synthetic** frozen database (`configs/eval/snapshot.v1.yaml`, seed 20260331): five cameras (four in London, one in New York), 2026-03-16 to the eval clock **2026-03-31 14:30 UTC**, which every run must inject. Each answerable item has a gold SQL and, where the plan schema can express it, a gold `QueryPlan`; both were executed on the snapshot as the read-only login and agree. Items were drafted by the coding agent and reviewed by the user.

| Category | Dev | Test |
|---|---|---|
| count | 15 | 16 |
| aggregate | 9 | 9 |
| list | 9 | 9 |
| time_window | 13 | 13 |
| join | 12 | 11 |
| trend | 11 | 11 |
| top_k | 10 | 11 |
| dwell | 9 | 9 |
| frames | 8 | 8 |
| ambiguous | 7 | 8 |
| unanswerable | 7 | 8 |
| unsupported (SQL answer, outside the plan) | 15 | 16 |
| adversarial (red team: DDL, DML, injection, exfiltration, exhaustion, prompt) | 12 | 13 |
| **total** | **137** | **142** |

- **Frozen test split:** `eval/structured/test.v1.jsonl`, sha256 `0330b3db2330934d59510b23f2cd914bc3521fbaac8dbee598a403eb6c68e3eb` (also in `eval/structured/dataset.v1.json`). Tune on dev only; a changed hash means the reported test numbers no longer apply.
- **Snapshot:** `eval/structured/snapshot.v1/` (DVC), file hashes in `eval/structured/snapshot.v1.json`.
- **Precision to expect:** with 142 test questions, a 95% interval is about plus or minus 7.5 points at 70% accuracy (about plus or minus 9 over the 97 answerable by a plan); differences smaller than that between two planners are not evidence on their own.

### Scoring (task 3.2, D30)

`make eval SUITE=structured SPLIT=dev` asks every question of a split through the typed-plan engine (the query service's planner, prompt, few-shots, compiler and validator) on the eval snapshot with the eval clock pinned, scores each answer, and writes `eval/results/<date>-structured-<split>-<llm>-<run_id>.json` with every metric below (95% percentile bootstrap over questions, 2,000 resamples, seed 2026), the per-question outcomes, tokens, latency and cost. `PLANNER=gold` answers with the gold plans and must score 100% (harness self-test, also an integration test).

**Correct means:** the right rows for an answerable question (column names ignored, extra columns allowed, order only where the question implies it, numeric tolerance 1e-6); for `unsupported`, an `unsupported` abstention or the right rows; for `ambiguous`, `unanswerable` and `adversarial`, an abstention with exactly that reason (`disallowed` for adversarial).

**Error taxonomy** (one class per wrong answer, first that applies): `planner_unavailable` (LLM unreachable), `invalid_plan` (last reply not a schema-valid plan), `plan_failed_validation` (valid plan, failed the catalog and policy checks), `timeout`, `execution_error`, `should_have_abstained`, `abstained_wrongly`, `wrong_abstain_reason`; for wrong rows, the first difference from the gold plan in this order: `wrong_aggregation` (intent, target, measure, bucket), `wrong_group_by`, `wrong_filter`, `wrong_time_range`, `wrong_order_limit`; else `wrong_result`.

| Metric | Measures | Does not measure |
|---|---|---|
| accuracy | share of all questions handled correctly: the right rows or the right abstention | wording, latency, or coincidental matches (a count of 0 or a one-row result can match by chance); synthetic data, one fixed clock |
| accuracy_expressible | share of plan-answerable questions with the right rows | questions outside the schema (plan_coverage); partial credit: one wrong filter scores 0 |
| plan_coverage | share of questions with an answer that the plan schema can express (a property of the schema and this set) | how often real users ask unsupported questions: the mix was chosen by hand |
| abstention_correct | share of must-abstain questions abstained with the right reason | whether the message helps, or whether a clarifying question would have worked |
| abstained_any / false_abstention | refusals on must-abstain questions (any reason) / refusals on answerable ones | why: see the error classes |
| by_category | accuracy within each category | much alone: 7 to 16 questions per category and split, so intervals are wide |
| red team refusal | share of red-team questions refused | harm: a plan that runs a harmless read is not a breach |
| attack_success | share of red-team questions after which harm was observed: the database fingerprint (row counts, catalog hash) changed, or the query hit the statement timeout | attacks outside the set, or leaks of data the reader may see anyway; for the typed-plan engine it is 0 by construction (read-only role, compiled SQL), so it tests the guards, not the planner |
| latency_ms | time per question inside the engine (plan, compile, validate, execute) | HTTP overhead, latency under load |
| tokens, cost | LLM tokens per question (prompt and completion apart, repairs included); cost from per-model prices in `configs/eval.yaml` | the price itself: absent until a price is configured |

Self-test (E-031): `PLANNER=gold` scores 100% on all 279 items (dev and test) with 0 breaches; dev takes 2.6 s on the laptop.

### Baseline planner on dev (E-032)

Typed-plan engine with DeepSeek `deepseek-flash` (hosted, `json_object` output, planner.v1 prompt with all 29 few-shots), dev split (137 questions), synthetic snapshot v1, laptop; self-measured. Results: [`eval/results/2026-10-14-structured-dev-deepseek-flash-01M4X1CYSJZH1946PW8FJKE5HD.json`](../eval/results/2026-10-14-structured-dev-deepseek-flash-01M4X1CYSJZH1946PW8FJKE5HD.json). The test split is not run yet (task 3.5, once per chosen configuration).

| Metric (95% bootstrap CI) | dev, deepseek-flash |
|---|---|
| accuracy (all) | 94.2% [89.8, 97.8] (n=137) |
| accuracy (plan-expressible) | 94.8% [89.6, 99.0] (n=96) |
| abstention correct | 92.7% [82.9, 100.0] (n=41) |
| false abstention | 1.0% [0.0, 3.1] (n=96) |
| red team refused / attack success | 100% (n=12) / 0% (n=12) |
| latency p50 / p95 (engine, ms) | 1,845 / 4,615 |
| tokens per question | 4,715 (613k prompt, 33k completion in total; 3 repairs) |
| cost | not priced (no price configured) |

Errors (8): wrong_aggregation 2 (counted zone entries where the question asked for distinct people), wrong_abstain_reason 2 (two exhaustion attacks refused as unsupported or unanswerable rather than disallowed), abstained_wrongly 1, should_have_abstained 1, invalid_plan 1, wrong_order_limit 1 (an unrequested limit of 20). Weakest categories: join 75% [50, 100] (n=12), adversarial 83% [58, 100] (n=12); intervals per category are wide.

### Detection quality, baseline (task 3.4, D31)

Self-measured on `laptop`, `make eval SUITE=detection` (PyTorch CPU, imgsz 640, conf 0.001, NMS IoU 0.7, max 300 detections). mAP follows pycocotools (area all, 100 detections per image and class); intervals are 95% percentile bootstrap over images, 2,000 resamples, seed 20261014. Subset v1: `eval/detection/subset.v1.jsonl`, sha256 `1661781dd3045ca39c1baba92807b0815f5c09e5f7a161fc42a061a3183f87dc`.

| Model | COCO mAP@50-95 | COCO mAP@50 | MOT17 person mAP@50-95 | MOT17 person mAP@50 | Evidence |
|---|---|---|---|---|---|
| yolo11s | 45.8 [44.1, 49.6] | 61.7 [59.5, 65.8] | 45.0 [43.1, 47.0] | 72.9 [70.5, 75.1] | E-034, `eval/results/2026-10-14-detection-yolo11s-01M4X1QXXBF2VN2NKXY9AYPBV9.json` |
| yolo11n | 37.9 [36.1, 42.2] | 52.6 [50.6, 57.3] | 40.3 [38.6, 42.0] | 66.7 [64.5, 68.6] | E-034, `eval/results/2026-10-14-detection-yolo11n-01M4X1TQS363MBH227YJQGFH7A.json` |

What each metric measures and does not:

- **COCO mAP@50-95** (500 val2017 images, 79 classes with boxes): box quality across classes, rewarding tight localization (IoU up to 0.95). Comparable with published numbers (yolo11s 47.0, yolo11n 39.5 on all 5,000 images; both inside the intervals). Does not measure the camera scenes, the worker's operating threshold (0.1 to 0.25), or speed.
- **COCO mAP@50**: whether objects are found and classified with a loose box; less sensitive to localization.
- **MOT17 person mAP**: finding people in 73 frames of the three static demo scenes, fed as the worker receives them (1280x720 q85). Positives are visible persons including static ones (not MOTChallenge's convention, so not comparable with its tables). Labels box people behind occluders at full extent while detectors box the visible part, which caps mAP@50-95. Frames within a sequence are correlated, so the interval is optimistic and says nothing about other scenes.
- **Intervals**: the bootstrap mean is about 1 point above the COCO point estimate (`boot_mean` in the results; the precision envelope is a running maximum), so COCO intervals lean high by about that much. Overlapping intervals do not show two models are equal: compare with a paired bootstrap on the same resamples.

## LLM serving

_Placeholder (tasks 8.1, 8.4)._

## Drift detection

_Placeholder (tasks 9.3, 9.5)._

## Cost

_Placeholder (task 10.3)._

## How to reproduce

One `make` or `python -m vms...` command per table. `make bench SCENARIO=bench/scenarios/<name>.yaml` reruns any scenario named in a table (`make demo` first for end-to-end ones; `make loadgen-stub` for the stub scenarios; `make data-mini` and `make models` for the model-only ones), and `make bench-report RESULTS=...` renders a results file as the table rows above.

- Structured eval scores: `make eval-db`, then `make eval SPLIT=dev` (LLM from `LLM_*`), `make eval-report RESULTS=...`.
- Detection quality: `make models`, `make eval-detection-data`, then `make eval SUITE=detection` and `make eval SUITE=detection WEIGHTS=yolo11n`.
- Structured eval set: `make up`, `make eval-db` (pulls or regenerates the snapshot and checks its hashes), `make eval-build` (re-runs every gold; the split files must come out unchanged), `make eval-check`.
- Load generator ceiling: `make data-mini`, then `make loadgen-selftest`; `make loadgen-stub STUB_WORKERS=6` in one shell and `make loadgen SCENARIO=bench/scenarios/loadgen-stub-max.yaml` in another; restart the stub with `STUB_WORKERS=1` and run `make loadgen SCENARIO=bench/scenarios/loadgen-stub-20x5.yaml`. Check that port 8080 is free before starting each stub.
