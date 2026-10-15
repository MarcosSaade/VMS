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
| laptop | Intel i7-1355U (2 performance + 8 efficiency cores, 12 threads), 15 GiB RAM, no GPU, Linux 6.8 | Development laptop. Runs before 2026-10-14 had other work running and are indicative. The 2.5/2.6 runs (E-039 to E-042) stopped the unrelated containers; a browser and desktop stayed open, so it is quieter, not a dedicated host. |

## Headline results

Self-measured on `laptop`; 95% bootstrap intervals. Still to come: rated capacity on a GPU, the optimized worker (5.5), cost per camera-hour, drift detection delay.

| Result | Number | Evidence |
|---|---|---|
| Structured-query execution accuracy, frozen test split (142 questions over a **synthetic** snapshot), typed-plan engine with DeepSeek `deepseek-flash` | 95.8% [92.3, 98.6]; plan-expressible 95.9% [91.8, 99.0]; red team 13/13 refused, 0 breaches; $0.24 for the split (upper bound) | E-036, [`2026-10-14-structured-test-deepseek-flash-01M4X596J9CHNDG20XEMRVQT2F.json`](../eval/results/2026-10-14-structured-test-deepseek-flash-01M4X596J9CHNDG20XEMRVQT2F.json) |
| Rated capacity, CPU (one yolo11n worker, PyTorch, batch 1) | **2 cameras at 5 FPS** (3/3 repetitions; freshness p95 0.24 s, 0 lost); the worker saturates at 14.2 frames/s | E-039, [`ramp-cpu`](../bench/results/2026-10-14-ramp-cpu-01M4XTS4MEBS8ZM7X2GERGGHS3.json) |
| Structured-query latency under ingest, CPU, canned planner (LLM excluded) | p95 39 ms at 1 client, 98 ms at 8 (159 q/s, one core saturated) | E-040, [`query-load-cpu`](../bench/results/2026-10-14-query-load-cpu-01M4XW866TFS51PVHPXJW25EA0.json) |
| Detection quality, yolo11s (PyTorch CPU) | COCO-500 mAP@50-95 45.8 [44.1, 49.6]; MOT17 person mAP@50-95 45.0 [43.1, 47.0] | E-034 (Evaluation, detection quality) |
| Precision ladder, yolo11s CPU, model-only (`detect_batch`, batch 1) | ONNX Runtime FP32 **1.65x** PyTorch (21.3 vs 12.9 images/s), same mAP; INT8 3.0x (38.2 images/s) but COCO mAP@50-95 -1.07 [-1.84, +0.40] fails D32, so not kept | E-044, E-045 (Inference, precision ladder) |

## Ingest and pipeline capacity

### Baseline ramp, CPU (task 2.5, E-039)

`make bench SCENARIO=bench/scenarios/ramp-cpu.yaml` on `laptop`, commit 7803f35: containerized ingest, one worker (yolo11n, PyTorch CPU build, batch size 1, imgsz 640), Postgres and SeaweedFS from `make up`; loadgen replays the data-mini clips at 5 FPS per camera (1280x720 q85); no query load. Warmup 20 s, hold 90 s, 3 repetitions; mean (std). GPU column: waits for the GPU quota (2.4).

| Cameras | Offered = accepted fps | Worker done fps | Freshness p50 / p95 s | Backlog slope fps (max frames) | Lost | Worker cores | Ingest SLO |
|---|---|---|---|---|---|---|---|
| 1 | 5.0 | 5.0 | 0.17 / 0.24 | 0.00 (1) | 0 | 1.40 | 3/3 |
| **2** | 10.0 | 10.0 | 0.16 / 0.24 | 0.00 (2) | 0 | 2.59 | **3/3** |
| 3 | 15.0 | 14.2 (0.3) | 4.67 / 9.41 | +0.85 (99) | 0 | 3.53 | 0/3 |
| 4 | 20.0 | 13.9 (0.2) | 19.5 / 37.8 | +5.82 (633) | 0 | 3.54 | 0/3 |

**Rated capacity on CPU: 2 cameras at 5 FPS.** The worker's ceiling is 14.2 frames/s (2.8 cameras); whole-camera steps hide that headroom, so a gain of about 1.1x would rate 3 cameras. Above the ceiling nothing is lost (frames queue in Postgres); freshness grows with the queue instead, which is what `worker.queue_wait` shows (p95 9.4 s at 3 cameras, equal to freshness). Per-frame stage p95 at every load: infer 73 ms, decode 9.5, fetch 8.5, write 9, stats 8, track 4.5, which sum to about the 70 ms per frame the ceiling implies, because the stages run one after another (section "Baseline bottlenecks"). Harness: loadgen 0.04 cores and `generator_ok` in every step; ingest 0.15 cores at 20 frames/s.

Evidence: [`bench/results/2026-10-14-ramp-cpu-01M4XTS4MEBS8ZM7X2GERGGHS3.json`](../bench/results/2026-10-14-ramp-cpu-01M4XTS4MEBS8ZM7X2GERGGHS3.json). A first attempt was aborted by a full disk (E-038).

### Load generator ceiling (harness, not the system)

Self-measured on `laptop`, 1280x720 JPEG q85 frames from `make data-mini`, 3 repetitions each, warmup excluded. These say how much load one loadgen process can offer before it, rather than the system under test, becomes the limit.

| Scenario | What it measures | Frames/s (mean, std) | CPU per frame | Evidence |
|---|---|---|---|---|
| `loadgen-selftest` | Generator core, no I/O | 36,412 (1,754) | 27 us | E-009, `bench/results/2026-10-09-loadgen-selftest-01M4GK6TJTHH0D2T3JW9KHWK28.json` |
| `loadgen-stub-max` | Generator + multipart HTTP over loopback, stub with 6 workers | 1,252 (44) | 0.80 ms | E-010, `bench/results/2026-10-09-loadgen-stub-max-01M4GK893V2X7WP677GQEXG8GW.json` |
| `loadgen-stub-20x5` | 20 cameras x 5 FPS paced, 1-worker stub | 100.0 offered = accepted, 0 dropped; send p50/p95/p99 8.7/17.6/24.1 ms; schedule lag p99 2.3 ms | 1.6 ms | E-011, `bench/results/2026-10-09-loadgen-stub-20x5-01M4GKA7GNXHJX3ABMV91AKYJ1.json` |

One process should offer at most ~70% of 1,250 frames/s (~175 cameras at 5 FPS) with `generator_ok` true. Above that, shard cameras across processes.

## Query latency

### Under ingest load, CPU, canned planner (task 2.5, E-040)

`make bench SCENARIO=bench/scenarios/query-load-cpu.yaml` on `laptop`, commit 7803f35: closed-loop clients asking the 10 demo questions (30% are intended abstentions) against the containerized query service with the canned planner, while ingest runs at 70% of the rated capacity (2 cameras x 3.5 FPS = 7 frames/s). Warmup 15 s, hold 60 s, 3 repetitions; mean (std across repetitions below 2 ms and 2 q/s everywhere).

| Clients | Query p50 / p95 / p99 ms | Queries/s | Query service cores | Freshness p95 s (ingest SLO) |
|---|---|---|---|---|
| 1 | 16 / 39 / 54 | 58 | 0.58 | 0.24 (3/3) |
| 2 | 20 / 46 / 67 | 95 | 0.85 | 0.24 (3/3) |
| 4 | 28 / 63 / 98 | 130 | 0.98 | 0.24 (3/3) |
| 8 | 46 / 98 / 131 | 159 | 1.00 | 0.24 (3/3) |

What this measures: everything a question costs except the LLM call (plan lookup, validation, compile, SQL validation, execution as the read-only login, answer, `query_log`), with ingest running. It does not include the planner: with the hosted LLM, the call alone has p50 1.9 s and p95 5.1 s (E-036), so the 1.5 s p95 SLO fails on the LLM, not on this path. From 4 clients the service pins one core (one Python process), so latency then grows with the queue; query load did not disturb ingest.

Evidence: [`bench/results/2026-10-14-query-load-cpu-01M4XW866TFS51PVHPXJW25EA0.json`](../bench/results/2026-10-14-query-load-cpu-01M4XW866TFS51PVHPXJW25EA0.json).

### Baseline bottlenecks, CPU (task 2.6, E-041, E-042)

Profiles: `make bench SCENARIO=bench/scenarios/profile-worker-cpu.yaml` and `profile-query-cpu.yaml` (py-spy at 100 Hz on the service's code run on the host, the container stopped; RUNBOOK, "Profiling"). Shares are of sampled Python-thread time, startup excluded; native time (PyTorch, OpenCV, libpq) counts against the Python frame that called it. Open a profile at https://www.speedscope.app after `gunzip`.

**Worker at saturation** (4 cameras offered, 14 frames/s done; [results](../bench/results/2026-10-14-profile-worker-cpu-01M4XXTACAXZPASNPFMDASJC9D.json), [profile](../bench/profiles/2026-10-14-profile-worker-cpu-01M4XXTACAXZPASNPFMDASJC9D.speedscope.json.gz)):

| Stage | Share | Thread |
|---|---|---|
| PyTorch forward pass | 65.9% | inference |
| JPEG decode | 6.6% | inference |
| Input statistics (brightness, contrast, Laplacian) | 6.1% | inference |
| Database (claim, write) | 5.0% | event loop |
| Event loop | 4.1% | event loop |
| Tracking and zones | 2.8% | inference |
| Object store fetch | 2.1% | event loop, s3 |
| Ultralytics letterbox / NMS and box scaling | 1.8% / 1.5% | inference |
| Other, telemetry | 3.5% | |

The inference thread was busy 57 s of the 80 s under load (71%).

**Query service, canned planner, 8 clients** (130 q/s with the profiler attached; [results](../bench/results/2026-10-14-profile-query-cpu-01M4XXP40EA6Q9CQ3GV2C73DCB.json), [profile](../bench/profiles/2026-10-14-profile-query-cpu-01M4XXP40EA6Q9CQ3GV2C73DCB.speedscope.json.gz)): SQL validation (sqlglot parse and regenerate) 53.2%, HTTP and pydantic 14.0%, OpenTelemetry 11.6%, execution 10.4%, planning 4.8%, event loop 3.4%, compile 0.7%.

**Ranked bottlenecks** (for the CPU deployment; the GPU column comes with the quota):

1. **The detector's forward pass** sets the worker's ceiling: 66% of the worker's time at saturation, infer p95 73 ms per frame, at batch size 1. PyTorch runs 10 intra-op threads on a 2P+8E hybrid CPU yet the worker uses only 3.5 cores at saturation; whether fewer threads (on the performance cores) are faster is a hypothesis for 5.2, not measured here. This is what Phase 5 attacks: ONNX Runtime (already at parity, 5.1), the precision ladder (5.2) and batching (5.3). Fix first: largest share, and the levers are planned.
2. **The worker runs each frame's stages one after another**: claim, fetch, infer, write. The inference thread idles 29% of the time while the event loop does I/O, a ceiling of about 1/0.71 = 1.4x left on the table. Overlapping the next frame's fetch and the previous frame's write with inference is part of 5.5 (batching across cameras); no plan change needed.
3. **Pre-model CPU work in the inference thread**: decode 6.6% + input statistics 6.1%, about 9 ms per frame. Statistics compute a full-resolution float64 Laplacian (E-013); a downscaled float32 version would remove most of it. Small, cheap, and in the same thread as the model; it matters more once the model is faster.
4. **Query service, only without an LLM**: one process, one core, ~160 q/s, half of it SQL validation. Not a bottleneck while the planner is a hosted LLM (~1.9 s per question); with a fast local planner, cache validated SQL per plan or run more server processes.

Not bottlenecks: ingest (0.15 cores at 20 frames/s), loadgen (0.04 cores), tracking (2.8%), Ultralytics pre- and post-processing (3.3% together, which is why GPU pre/post-processing, 5.4, stays deferred).

## Storage

_Placeholder (tasks 4.3, 4.4)._

## Inference

_Batch sweep (5.3), the GPU rungs of the ladder (TensorRT FP16 and INT8) and the optimized worker end to end (5.5) are still to come._ Model-only and end-to-end numbers are reported separately, with mAP next to speed.

### Worker model-only latency, CPU baseline

Self-measured on `laptop` with `make bench-model`: the worker's per-frame work in its own thread (JPEG decode, input statistics, YOLO11 pre-processing + forward + NMS; tracking is not included), batch size 1, imgsz 640, detector threshold 0.1 (the tracking default, D21), PyTorch 2.14.1 CPU build, 150 frames from the three data-mini clips at 1280x720 q85, 10 warmup frames excluded, 3 repetitions. No database, object store or HTTP, so end-to-end throughput will be lower (task 2.5). Its quality cost: yolo11n scores about 8 points lower COCO mAP@50-95 and 5 points lower MOT17 person mAP@50-95 than yolo11s (Evaluation, detection quality).

| Model | Frames/s (mean, std) | Infer p50 / p95 ms | Total p95 ms | CPU cores | Raw detections/frame | Evidence |
|---|---|---|---|---|---|---|
| yolo11s | 9.8 (0.22) | 89 / 105 | 119 | ~7.5 | 27.8 | E-016, `bench/results/2026-10-12-worker-model-cpu-01M4QVF9ZAKC9HXMTFERF164VV.json` |
| yolo11n | 18.8 (1.9) | 40 / 47 | 62 | ~7.5 | 23.2 | E-016, `bench/results/2026-10-12-worker-model-cpu-n-01M4QVGTE15P83D77JXS71K9VF.json` |

At the earlier threshold of 0.25 the numbers were the same within noise (E-013, E-014).

At 5 FPS per camera, one CPU worker keeps up with about 2 cameras on yolo11s and about 4 on yolo11n, before the database and object store are added. PyTorch already uses most cores in one call, so a second worker on the same laptop would compete for them rather than add capacity.

### ONNX export parity (task 5.1, E-037)

`make onnx-parity` on `laptop`: ONNX Runtime 1.31 CPU with our numpy pre/post-processing vs Ultralytics PyTorch CPU, both float32, over the detection subset (500 COCO + 73 MOT17) and 150 data-mini frames, at the worker's threshold (0.1) and the eval's (0.001). Correctness only; speed is 5.2.

| Check | yolo11n | yolo11s | Tolerance (`vms.worker.parity.Tolerances`) |
|---|---|---|---|
| Letterboxed input vs Ultralytics | identical bytes, 723/723 images | identical, 723/723 | exact |
| Raw head output, max abs diff: boxes (input px) / scores | 0.0096 / 6.1e-5 | 0.021 / 1.1e-4 | 0.1 / 1e-3 |
| Detections after NMS, max abs diff: boxes (image px) / scores | 0.0083 / 1.6e-5 | 0.018 / 5.6e-6 | 0.1 / 1e-4 |
| Detections on one side only (conf 0.001; conf 0.1) | 3 of 133k, all explained (2 at the threshold, 1 IoU at 0.7); 0 | 0 of 116k; 0 | unexplained: 0 |
| Batch of 8 vs one at a time (boxes) | 0.0002 px | 0.0002 px | 0.1 |
| COCO / MOT17 mAP@50-95 (PyTorch value) | 37.9 [36.1, 42.2] / 40.3 [38.6, 42.0] (37.9 / 40.3) | 45.8 [44.1, 49.6] / 45.0 [43.1, 47.0] (45.8 / 45.0) | |

Evidence: [`yolo11n parity`](../bench/results/2026-10-14-onnx-parity-yolo11n-01M4XQQGJWF596G49X0GQ06AZY.json), [`yolo11s parity`](../bench/results/2026-10-14-onnx-parity-yolo11s-01M4XSHPCKT13PEV7NGAW18RX5.json), [`yolo11n ONNX mAP`](../eval/results/2026-10-14-detection-yolo11n-onnx-01M4XSHQVKYXR93NSRME7F6SRC.json), [`yolo11s ONNX mAP`](../eval/results/2026-10-14-detection-yolo11s-onnx-01M4XSKH477FFM21CWP6P58Z66.json).

**Why these tolerances.** Two float32 runtimes run the same graph with different kernels (oneDNN in PyTorch, MLAS in ONNX Runtime) and summation orders, so exact equality is not expected; differences come from rounding that accumulates over about 300 layers. A tenth of a pixel moves no pixel of a crop and the IoU of a 20 px box by about 1%; 1e-4 in score is 1/1500 of the gap between the worker's 0.1 and 0.25 thresholds. The tolerances sit 5x to 50x above the observed maxima, so a real bug (a half-pixel letterbox shift, RGB/BGR swapped, a wrong NMS rule) fails them by orders of magnitude while rounding never does. Where outputs can legitimately differ, near a threshold, the check classifies the case instead of loosening the tolerance.

### Precision ladder, CPU (task 5.2; E-043 to E-045)

Self-measured on `laptop` (clean commit 9b71ac7). **Model-only**: `detect_batch` (letterbox, forward pass, NMS) for yolo11s at imgsz 640 and threshold 0.1, on 300 decoded data-mini frames (1280x720, as loadgen sends them; JPEG decode excluded). `make bench SCENARIO=bench/scenarios/ladder-cpu.yaml`: each backend in its own process, 5 repetitions with the step order rotated, 15 s cooldown, 2 warmup batches per batch size, compose stack stopped. Thread defaults as the worker (one intra-op thread per physical core). Quality: the detection subset (D31) per backend, and the INT8 difference as a paired bootstrap against ORT FP32 (`make eval-detection-compare`).

| Backend | Batch 1 images/s (std) | Batch 1 latency p50 / p95 / p99 ms | Batch 4 images/s (std) | Batch 8 images/s (std) | CPU cores | Peak RSS MB | COCO mAP@50-95 | MOT17 mAP@50-95 |
|---|---|---|---|---|---|---|---|---|
| PyTorch FP32 (Ultralytics) | 12.9 (0.5) | 78 / 87 / 88 | 15.0 (0.7) | 14.9 (0.2) | 8.0 | 1,820 | 45.8 [44.1, 49.6] | 45.0 [43.1, 47.0] |
| ONNX Runtime FP32 | 21.3 (1.5) | 44 / 53 / 53 | 22.3 (1.7) | 21.0 (1.5) | 10.0 | 1,531 | 45.8 (paired delta 0.00) | 45.0 (paired delta 0.00) |
| ONNX Runtime INT8 (static QDQ, head decode FP32) | 38.2 (0.8) | 26 / 31 / 31 | 45.9 (0.2) | 46.1 (0.7) | 10.0 | 1,214 | 44.8, paired delta **-1.07 [-1.84, +0.40]** | 45.2, paired delta +0.13 [-0.27, +0.55] |

Peak RSS includes about 830 MB of decoded frames held by the harness. Per-batch latencies at batch 4 and 8 are in the results file. Evidence: [`ladder-cpu`](../bench/results/2026-10-15-ladder-cpu-01M50QBZK10K8HA5XBGGFJ5R3G.json), [`PyTorch mAP`](../eval/results/2026-10-15-detection-yolo11s-01M50Q4B0MS40CJ1DFBR0F3YVQ.json), [`ORT FP32 mAP`](../eval/results/2026-10-15-detection-yolo11s-onnx-01M50Q73K4R740EX5CW84K1BJB.json), [`INT8 mAP`](../eval/results/2026-10-15-detection-yolo11s-int8-01M50Q94H5G5N3H2Q1CRT5XVTM.json), [`INT8 vs ORT FP32`](../eval/results/2026-10-15-detection-compare-yolo11s-int8-vs-onnx-01M50R9S67XGGJFQ3SYKXE14JT.json), [`ORT FP32 vs PyTorch`](../eval/results/2026-10-15-detection-compare-yolo11s-onnx-vs-pytorch-01M50RBNWP219YDN57J11REJDW.json).

- **ONNX Runtime FP32 is 1.65x PyTorch at batch 1 for no quality change** (identical mAP; 5.1 showed identical detections within rounding).
- **INT8 is another 1.8x (2.1x at batch 4) but fails D32 on COCO by 0.07 points**: a 1.07-point drop against a 1.0 limit, set before the run. The upper end of the drop's interval (1.84) is within the 2.0 limit, and on MOT17 persons INT8 is no worse (mAP@50 +0.69 [+0.16, +1.49]). Quantizing the whole graph instead gives mAP 0 (E-043).
- **Batching does not help ONNX Runtime on this CPU** (batch 4 is +5%, inside the repetition std): one image already uses all 10 cores. PyTorch gains 17% at batch 4.
- Calibration: 64 COCO train2017 images and 64 frames of MOT17-05/10/11/13 (`configs/calibration.v1.lock.jsonl`, sha256 35018eeb...), disjoint from the eval subset (unit test). MinMax ranges, per-channel weights.
- Not measured: the GPU rungs (TensorRT FP16 and INT8, waiting for the GPU quota), yolo11n (the model of the E-039 ramp), and anything end to end (5.5).

## Evaluation

### Structured-query eval set v1 (task 3.1, D29, E-029)

Questions over a **synthetic** frozen database (`configs/eval/snapshot.v1.yaml`, seed 20260331): five cameras (four in London, one in New York), 2026-03-16 to the eval clock **2026-03-31 14:30 UTC**, which every run must inject. Each answerable item has a gold SQL and, where the plan schema can express it, a gold `QueryPlan`; both were executed on the snapshot as the read-only login and agree. Items were drafted by the coding agent and reviewed by the user.

| Category | Dev | Test |
|---|---|---|
| count | 16 | 16 |
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
| unsupported (SQL answer, outside the plan) | 14 | 16 |
| adversarial (red team: DDL, DML, injection, exfiltration, exhaustion, prompt) | 12 | 13 |
| **total** | **137** | **142** |

- **Frozen test split:** reviewed and approved by the user on 2026-10-14; every item's side is pinned in `eval/structured/splits.v1.json`, so later edits or additions never move a test question. `eval/structured/test.v1.jsonl`, sha256 `0330b3db2330934d59510b23f2cd914bc3521fbaac8dbee598a403eb6c68e3eb` (also in `eval/structured/dataset.v1.json`). Tune on dev only; a changed hash means the reported test numbers no longer apply.
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

Self-test (E-031): `PLANNER=gold` scores 100% on all 279 items (dev and test) with 0 breaches; dev takes 2.6 s on the laptop. CI runs it on dev through `make eval` on every push and fails on any wrong item (task 3.5).

### Baseline planner (task 3.5; E-032, E-035, E-036)

Typed-plan engine with DeepSeek `deepseek-flash` (hosted, `json_object` output, planner.v1 prompt with all 29 few-shots, temperature 0, nothing tuned on dev), synthetic snapshot v1, laptop; self-measured. This is the provisional planner of D7: the second LLM (a local open model on dev) runs on the first GPU session. The test split was run once, from clean commit 4228e11, and is the reported number; dev is for tuning.

- Test: [`2026-10-14-structured-test-deepseek-flash-01M4X596J9CHNDG20XEMRVQT2F.json`](../eval/results/2026-10-14-structured-test-deepseek-flash-01M4X596J9CHNDG20XEMRVQT2F.json)
- Dev (current, after sq-103 was accepted as a class listing): [`2026-10-14-structured-dev-deepseek-flash-01M4X45EXKGJP2055D6A04VPVK.json`](../eval/results/2026-10-14-structured-dev-deepseek-flash-01M4X45EXKGJP2055D6A04VPVK.json); first dev run: [`eval/results/2026-10-14-structured-dev-deepseek-flash-01M4X1CYSJZH1946PW8FJKE5HD.json`](../eval/results/2026-10-14-structured-dev-deepseek-flash-01M4X1CYSJZH1946PW8FJKE5HD.json)

| Metric (95% bootstrap CI) | test (reported) | dev |
|---|---|---|
| accuracy (all) | 95.8% [92.3, 98.6] (n=142) | 94.2% [89.8, 97.8] (n=137) |
| accuracy (plan-expressible) | 95.9% [91.8, 99.0] (n=97) | 94.8% [89.7, 99.0] (n=97) |
| abstention correct | 95.6% [88.9, 100.0] (n=45) | 92.5% [85.0, 100.0] (n=40) |
| false abstention | 0.0% (n=97) | 0.0% (n=97) |
| red team refused / attack success | 100% / 0% (n=13) | 100% / 0% (n=12) |
| latency p50 / p95 (engine, ms) | 1,855 / 5,126 | 1,929 / 5,917 |
| tokens per question | 4,855 (653k prompt, 36k completion; 7 repairs) | 4,840 (627k prompt, 37k completion; 6 repairs) |
| cost | $0.24 for 142 questions | $0.23 for 137 questions |

Cost is an upper bound (peak, cache-miss prices; E-035). Latency is per question inside the engine, with 4 questions in flight (red-team questions alone), and is dominated by the hosted LLM call; it is not latency under load.

Test errors (6), the same failure modes as on dev:

| Error class | Test | Dev | What went wrong |
|---|---|---|---|
| wrong_aggregation | 2 | 3 | counted zone-entry events where the question asked how many (different) people: the plan should count distinct tracks |
| wrong_abstain_reason | 2 | 2 | refused correctly with the wrong reason: a resource-exhaustion attack called `unsupported` (expected `disallowed`); a face-mask question called `unsupported` (expected `unanswerable`, no such attribute) |
| wrong_order_limit | 2 | 1 | "find the frames ..." got an unrequested `limit 20` (36 and 80 rows expected). Arguable as a UI choice, but the question asks for all of them |
| should_have_abstained | 0 | 1 | answered a question about a car's color with the latest car frame |
| invalid_plan | 0 | 1 | |

Weakest categories on test: frames 75.0% [37.5, 100.0] (n=8), unanswerable 87.5% [62.5, 100.0] (n=8), join 90.9% [72.7, 100.0] (n=11); with 8 to 16 questions per category, none of these differences is established. Dev and test agree within their intervals, so the dev numbers did not overfit the prompt (nothing was tuned). Run to run at temperature 0, 3 of 137 dev answers changed between two runs with the same accuracy (E-035): differences of a few points between configurations are noise unless the intervals say otherwise.

**What this does and does not show.** The questions were written by the coding agent and reviewed by the user, over a synthetic database whose schema the prompt describes in full; real users' questions would be messier, so this is an upper-end estimate for the planner on in-domain questions. The 0 breaches hold by construction for the typed-plan engine (read-only login, compiled SQL): they test the guards, not the model.

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

- Structured eval scores: `make eval-db`, then `make eval SPLIT=dev` and, once per configuration, `make eval SPLIT=test` (LLM from `LLM_*`; DeepSeek needs `VMS_LLM__RESPONSE_FORMAT=json_object`), `make eval-report RESULTS=...`.
- Detection quality: `make models`, `make eval-detection-data`, then `make eval SUITE=detection` and `make eval SUITE=detection WEIGHTS=yolo11n`.
- CPU baseline: `make up`, `make data-mini`, `make models`, then start the app containers with the canned planner (`LLM_MODEL= docker compose --env-file .env -f deploy/compose/docker-compose.yml --profile app up -d --wait ingest worker query` after `make app`), stop other CPU-heavy work, and run `make bench SCENARIO=bench/scenarios/ramp-cpu.yaml` (~45 min), `query-load-cpu.yaml` (~20 min), `profile-worker-cpu.yaml` and `profile-query-cpu.yaml` (~2 min each).
- ONNX parity: `make models`, `make eval-detection-data`, `make data-mini`, `make onnx`, then `make onnx-parity` and `make onnx-parity WEIGHTS=yolo11s` (~25 min each); ONNX mAP: `VMS_DETECTOR__BACKEND=onnxruntime make eval SUITE=detection` (and `WEIGHTS=yolo11n`).
- Precision ladder, CPU: `make models`, `make data-mini`, `make eval-detection-data`, `make onnx`, `make calibration-data`, `make quantize WEIGHTS=yolo11s`; then `make eval SUITE=detection NAME=detection-yolo11s`, `... BACKEND=onnxruntime NAME=detection-yolo11s-onnx`, `... BACKEND=onnxruntime PRECISION=int8 NAME=detection-yolo11s-int8` (~1.5 min each), `make eval-detection-compare BASE=<onnx run> OTHER=<int8 run>` (~2 min), and with the compose stack stopped `make bench SCENARIO=bench/scenarios/ladder-cpu.yaml` (~15 min). The whole-graph INT8 of E-043: `VMS_INT8__EXCLUDE_NODES='[]' make quantize WEIGHTS=yolo11s`.
- Structured eval set: `make up`, `make eval-db` (pulls or regenerates the snapshot and checks its hashes), `make eval-build` (re-runs every gold; the split files must come out unchanged), `make eval-check`.
- Load generator ceiling: `make data-mini`, then `make loadgen-selftest`; `make loadgen-stub STUB_WORKERS=6` in one shell and `make loadgen SCENARIO=bench/scenarios/loadgen-stub-max.yaml` in another; restart the stub with `STUB_WORKERS=1` and run `make loadgen SCENARIO=bench/scenarios/loadgen-stub-20x5.yaml`. Check that port 8080 is free before starting each stub.
