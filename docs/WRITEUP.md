# Technical write-up

The story of the system: what it does, how it was designed, what was measured, what did not work and what it does not show. Numbers are summarized here and curated in [BENCHMARKS.md](BENCHMARKS.md); each one links to its experiment (E-ids in [EXPERIMENTS.md](EXPERIMENTS.md)) and decision (D-ids in [DECISIONS.md](DECISIONS.md)), and through them to a results JSON with the git SHA, config, hardware and seeds that produced it.

## 1. Summary

VMS is a video management system: it ingests camera streams, detects and tracks objects, records when tracks enter and leave zones, answers natural-language questions about all of that ("how many people entered the loading dock yesterday, by hour?"), and watches each camera for drift. It was built as an ML systems project, so the product is less the features than the evidence behind each design choice.

The main results:

- **Question answering is accurate and safe on a frozen test set**: 95.8% [92.3, 98.6] execution accuracy on 142 test questions over a synthetic database, with every red-team question refused and no breach (E-036). The LLM never writes SQL; it fills a typed plan that the system compiles and checks (D13, D14).
- **Inference optimization survives end to end, but not all of it**: ONNX Runtime with cross-camera batching raised one CPU worker from 2 to 5 cameras at 5 FPS within the freshness SLO; the worker's throughput gain was 1.8x against 1.84x to 2.13x for the model alone (E-046 to E-049). On an AWS c7i.2xlarge the full pipeline costs $0.12 per camera-hour (E-059).
- **Precision ladders with quality next to speed**: TensorRT FP16 is 2.1x PyTorch at batch 1 on an A40 with mAP unchanged; INT8 failed an acceptance rule fixed before the first INT8 number, on CPU and on GPU (D32, E-044 to E-046, E-064).
- **Drift detection scored like a detector**: every injected drift caught, median delay 4.8 s, no false alarm in 4.2 clean camera-hours out of sample, and CUSUM shown to add nothing at these drift sizes (D34, E-053 to E-055).
- **A fine-tuned small LLM did not beat a hosted one**: LoRA on Qwen3.5-4B gained about 4 points over its base and 4-bit quantization made it 1.5x faster for free, but the hosted model stayed 12.7 [+7.0, +19.0] points ahead on test, and costs less below about 1.5M questions a month (D7, D42, D43, E-072 to E-078).
- **Two negative results kept in the record**: detector fine-tunes against blur lost more on clean video than they recovered on blurred video, with a control run that showed why (D36, D39).

## 2. Problem and workload

The system is sized for a site with tens of fixed cameras, each sending 5 frames per second at 720p ([ARCHITECTURE 2](ARCHITECTURE.md#2-workload-model)). Load comes from a generator that replays public video clips (MOT17, MOT20 and a Wikimedia Commons street scene) as N virtual cameras at a fixed rate, with optional drift injected at seeded onsets (D15, D17). Its own ceiling was measured first, so it is never mistaken for the system's: about 1,250 frames/s per process over HTTP, far above anything tested (E-009 to E-011).

Three service levels drive every capacity number:

| SLO | Target |
|---|---|
| Freshness: capture to detections queryable | p95 under 2 s |
| Frames lost (dropped, rejected, failed) | 0 |
| Structured question, LLM included | p95 under 1.5 s |

**Rated capacity** is the largest number of cameras at 5 FPS whose step, and every smaller one, meets the first two in every repetition with a backlog that does not grow (D27).

## 3. Architecture

![Camera wall](images/ui-cameras.png)

<sub>Frames: MOT17 (Milan et al., arXiv:1603.00831, CC BY-NC-SA 3.0) and Wikimedia Commons (Qviri, CC BY-SA 4.0).</sub>

The write path: **loadgen** posts JPEGs to **ingest**, which stores the image in an S3-compatible object store (SeaweedFS locally, D12) and inserts a frame row in Postgres, idempotently on `(camera_id, run_id, seq)` (D18). The `frames` table is the queue: **workers** lease whole cameras and claim their frames with `SELECT ... FOR UPDATE SKIP LOCKED`, so each camera's frames are processed in order by one worker (tracking needs that), and completion is fenced by the lease so a worker that lost its camera cannot write (D20). A worker decodes, runs YOLO11, tracks with ByteTrack, decides zone enter and exit events from the box's bottom centre, with a two-frame debounce against box jitter, and writes detections, tracks and events (D21).

The read path: the **query service** asks an LLM for a `QueryPlan`, a pydantic model with an intent (count, aggregate, list, top-k, trend, dwell, frames, abstain), a target, filters, a time range and grouping (D13). The plan is validated against the camera and zone catalog and against policy limits, repaired once if invalid, compiled to parameterized SQL with time ranges resolved in each camera's timezone, regenerated and checked by a sqlglot validator (statement type, table and column allowlist, function denylist, join and limit caps; D14), and executed by a read-only login with a statement timeout (D19, D22). Every question is logged with its plan, SQL, timings and tokens. The **UI** is vanilla TypeScript served by the query service: a live camera wall with boxes and zones, and a question box that renders a number, a chart, a table or frame thumbnails, with the plan and SQL one click away (D23).

The **drift monitor** reads windows of frames and detections from Postgres and computes, per camera, input statistics (brightness, contrast, blur, noise) and output statistics (detections per frame, confidence distribution, class mix) against a stored reference range (D33, D34).

Everything is instrumented with OpenTelemetry: one trace follows a frame from loadgen through ingest, the queue (the trace context is stored with the frame row) and the worker's stages, and one follows a question through planning, validation, compilation and execution (D25). Five Grafana dashboards are generated from Python and provisioned at startup (D26).

| Asking a question | Pipeline dashboard |
|---|---|
| ![Question with chart and plan](images/ui-query.png) | ![Pipeline dashboard](images/grafana-pipeline.png) |

What was deliberately left out, each with a recorded reason: Kafka or Redis as the queue (Postgres with `SKIP LOCKED` is enough at this scale until a benchmark says otherwise, D3), Triton (the in-process worker is measured first, D5), MLflow and W&B (results JSON files in git are the record, D16), Kubernetes (D6). These are **accepted defaults, not benchmark results**, and are marked as such in [DECISIONS.md](DECISIONS.md) (D1 to D4, D6, D8 to D11). The benchmarks that would settle them are on the [roadmap](ROADMAP.md).

## 4. Methodology

Every number in this document comes from a scenario file written before the run and executed through `make` ([BENCHMARKS, Methodology](BENCHMARKS.md#methodology)):

- **Steady state only**: each step runs warmup plus hold, and only the hold window counts; server-side numbers are PromQL over exactly that window. The worker's backlog is drained before each step.
- **At least three repetitions**, each a full pass over the steps, reported as mean and standard deviation.
- **The harness is checked**: the load generator flags its own lag, and the query clients' CPU is recorded; flagged steps are not reported.
- **One variable at a time** against a named baseline, on the same hardware.
- **Quality next to speed**: every precision or serving change reports mAP or execution accuracy, compared with a paired bootstrap on the same images or questions.
- **Results files are the record**: each holds the git SHA and dirty flag, the resolved config, hardware, package versions, seeds and start and end times. There is no experiment tracker (D16).
- **Rented hardware is scripted**: GPU sessions on Runpod and CPU sessions on AWS are created, run, collected and deleted by one command, tagged or named per session, armed to delete themselves after a time limit, and checked for leftovers after teardown (D28, D35, D37). All the rented compute in the project cost about $6 ([BENCHMARKS, Cost](BENCHMARKS.md#cost)).

## 5. Baseline and bottlenecks

The first ramp on the laptop CPU (yolo11n, PyTorch, batch 1) rated **2 cameras**: at 3 cameras the worker topped out at 14.2 frames/s, freshness p95 jumped to 9.4 s and the backlog grew, with nothing lost because frames wait in Postgres (E-039). Profiles with py-spy ranked the bottlenecks (E-041, E-042):

1. The model's forward pass: 66% of the worker's time at saturation.
2. Serial stages: claim, fetch, infer and write ran one after another; the inference thread was idle 29% of the time.
3. CPU work before the model: JPEG decode and a full-resolution float64 Laplacian for the input statistics, about 9 ms per frame.

The query path without an LLM held p95 under 100 ms at 8 concurrent clients (159 questions/s, one core), with half its time in SQL validation (E-040). With a hosted LLM the call alone is about 1.9 s, so the 1.5 s question SLO fails on the LLM, not on the system.

## 6. Inference optimization

**Parity before speed.** The ONNX export with our own numpy letterbox and NMS matches PyTorch within 0.02 px and 1e-4 in score on 723 images, with identical mAP. The tolerances sit 5 to 50x above the observed differences, so a real bug (a half-pixel shift, swapped channels) fails them by orders of magnitude while rounding never does (E-037).

**Precision ladders** (model-only, yolo11n unless noted; [CPU](BENCHMARKS.md#precision-ladder-cpu-task-52-e-043-to-e-046), [GPU](BENCHMARKS.md#precision-ladder-gpu-task-52-e-064-d38)):

| Rung | Laptop CPU, batch 1 | A40, batch 1 | mAP@50-95 change (paired, 95% CI) |
|---|---|---|---|
| PyTorch FP32 | 26.5 images/s | 8.5 ms | baseline |
| ONNX Runtime FP32 | 48.8 (1.84x) | 5.7 ms | 0.00 |
| TensorRT FP16 | | **3.9 ms (2.1x)** | -0.06 [-0.23, +0.04] COCO |
| INT8 (ORT on CPU, TensorRT on GPU) | 65.8 | 5.2 ms | **-1.30 [-1.99, -0.52]** COCO (CPU), **-2.15 [-2.91, -1.18]** (GPU) |

INT8 was judged by a rule written before any INT8 number existed: at most 1 point of mAP@50-95 on either source, with the interval's upper end under 2 (D32). It failed everywhere. On the GPU it was also slower than FP16, because only the convolutions were INT8 and the rest of the graph stayed FP32. Calibrating every node instead gave an mAP of 0 (E-043), and a scalar constant wrapped in Q/DQ crashed TensorRT's parser until the quantizer was restricted to convolutions (E-062).

**End to end.** The worker on ONNX Runtime FP32 with cross-camera batches of 4 rated **5 cameras** instead of 2 (E-047, E-048). Two lessons from that ramp:

- **The model-only speedup shrinks end to end** (1.84x to 2.13x model-only, 1.8x for the worker's ceiling), because fetch, decode, statistics, tracking and writes remain, and the worker shares cores with ingest and Postgres.
- **Default thread settings can waste a machine**: ONNX Runtime's intra-op threads busy-wait between calls, so the worker showed about 9 cores in use at any load. Letting them sleep kept the same capacity at 1.3 cores for one camera instead of 9.2, at the price of 0.12 s of freshness at 5 cameras (E-049; D5).

On an AWS c7i.2xlarge (4 cores) the same worker rated 3 cameras, about **$0.12 per camera-hour** of on-demand compute for the whole pipeline (E-059). The batch sweep picked batch 4 on the laptop and 8 on the 4-core server: with fewer cores and sleeping threads, a bigger batch amortizes the wake-up (E-057, E-059).

On the A40, batching barely helped the exported models: the backends that letterbox and run NMS in numpy on the CPU leveled off at 250 to 300 images/s, while PyTorch, which does both on the GPU, kept scaling to 386 images/s at batch 8 (E-064, E-065). The next GPU lever is pre- and post-processing, not the backend.

## 7. Evaluation

### Structured questions

There was no labeled set of questions about camera data, so one was built (D29): a **synthetic** database of five cameras over two weeks, generated from a seeded spec, and 279 questions across counts, aggregates, lists, time windows, joins, trends, top-k, dwell times and frame retrieval, plus questions the system must refuse: ambiguous, unanswerable, unsupported by the plan schema, and a red team (DDL, DML, injection, exfiltration, resource exhaustion, prompt injection). Every answerable question has two independent golds, a hand-written SQL query and a `QueryPlan`, executed on the database and required to agree. Questions were split by paraphrase group, so no test question has a paraphrase in dev, and the test file is frozen by hash.

The scorer (D30) compares result sets, not SQL text: column names ignored, extra columns allowed, order checked only when the question implies one. Each wrong answer gets exactly one error class, the first difference from the gold plan (wrong aggregation, grouping, filter, time range, order or limit, or a missed or wrong abstention), so the error table points at what to fix. Red-team success is measured as harm, a change in the database's fingerprint or a statement timeout, not as the model "agreeing". Intervals are percentile bootstraps over questions. A gold-plan self-test must score 100% and runs in CI.

With a hosted planner (DeepSeek), the one test run scored **95.8% [92.3, 98.6]**, abstained correctly on 95.6% of the questions it should refuse and never refused an answerable one; all 13 red-team questions were refused with no harm (E-036). The six errors were two "how many people" questions counted as events instead of distinct tracks, two refusals with the wrong reason and two unrequested limits on frame listings. Dev and test agree within their intervals; at temperature 0, 3 of 137 dev answers changed between identical runs, so differences of a few points are noise (E-035).

### Detection

A frozen subset of 500 COCO validation images and 73 MOT17 frames from the demo scenes, mAP computed exactly as pycocotools does, with bootstrap intervals over images and paired comparisons between models (D31). yolo11s scores 45.8 [44.1, 49.6] COCO and 45.0 [43.1, 47.0] MOT17 person mAP@50-95; the published full-COCO figures fall inside the intervals (E-034).

## 8. Drift detection

Drift was injected by the load generator at seeded onsets: darkening, blur, noise, a camera shift, occlusion, and a scene change; abrupt on one camera and as a two-minute ramp on another, with two cameras left clean (E-053). First, its cost: blur at sigma 6 removes 17 points of the demo scenes' mAP@50-95, the largest drop (E-051).

Detection was scored like any detector, on **delay against false-alarm rate**, out of sample (calibrate on two repetitions, score the third) over 84 camera streams ([BENCHMARKS](BENCHMARKS.md#detection-delay-and-false-alarms-task-93-e-053-to-e-055-d34)):

| Detector | False alarms (per clean camera-hour, 95% upper bound) | Missed | Median / p90 / max delay |
|---|---|---|---|
| Fixed threshold on calibrated z | 0 (< 0.72) | 0 / 36 | 4.8 / 18.1 / 33.1 s |
| CUSUM | 2 (< 1.51) | 0 / 36 | 4.8 / 18.1 / 33.1 s |

![Delay vs false alarms](images/drift-delay-vs-false-alarms.png)

CUSUM accumulates small persistent deviations, which is its advantage; these drifts move their signal by tens of clean standard deviations, so a one-window threshold already fires at the next update, and CUSUM only added false alarms. The deployed rule is a fixed threshold at h=10, with margin over the sensitive end (D34). Abrupt drifts fire at the first 5 s update after the onset; live, the alert reaches Grafana 14 s after an abrupt and 27 s after a gradual darkening (E-055). Signals are not one-to-one with causes (blur moves under darkening, noise and shifts too), so an alert names the channel, and the [runbook](RUNBOOK.md) says what to check in which order.

![Drift dashboard](images/grafana-drift.png)

## 9. LLM serving and the planner decision

The planner was the slowest and most expensive stage, so Phase 8 asked whether a small self-hosted model could replace the hosted one. All of it ran on one rented A40 with vLLM; accuracy was recorded on the pod and replayed through the unchanged eval on the laptop, keyed by the exact messages sent, with zero replay misses (D40).

**Serving.** On the planner's real ~4k-token prompt, prefix caching cut time to first token from 728 to 162 ms and tripled throughput at 16 concurrent requests (Qwen3.5-9B). Structured outputs cost nothing measurable in speed but did not improve plan validity: the failures were semantic (unknown names, policy limits), which a JSON schema cannot express. Ollama was 2 to 5x slower than vLLM on the same model (E-068).

**Base models** scored 75% on dev against the hosted model's 94% (E-069). Qwen3.5-4B matched the 9B at twice the speed, so it became the fine-tuning base.

**Training data** (D41): 2,660 synthetic (question, plan) pairs over three new synthetic sites, generated by the hosted model for $2.99. A pair was kept only if two independent generations produced the same executed result and that result was not empty, and if it was not a near-duplicate of any eval question or few-shot example (0 leaks after filtering; E-070).

**LoRA** (D42): trained with a compact prompt (the schema and examples removed: 3k instead of 15k characters). Data size mattered (all pairs beat 500 or 1,000 by 6 to 10 dev points); rank did not (4, 16 and 64 within 0.8 points, less than the 3.6 points between two identical runs). On test the rank-4 adapter scored **83.1%** against 78.9% for the base on its full prompt; paired, **+4.2 [-0.7, +9.9]**, likely but not shown (E-072 to E-074, E-078).

**Quantization and merging** (D43): on published GPTQ checkpoints of the same base, W4A16 made the adapter **1.5x faster** at one request at a time (decode 79 vs 51 tokens/s, memory-bound) and 17% faster at 32, with weights at 4.5 instead of 8.0 GiB, for **-0.7 [-3.5, +1.4]** test points. Merging the adapter into the weights gave 146 identical replies out of 147 and removed vLLM's 13 to 14% LoRA overhead (E-075, E-076).

**The decision** (D7), on test accuracy, p95 under the same load and cost per 1,000 questions (E-077, E-078):

| Option | Test accuracy | p95, 1 request at a time | Cost per 1,000 questions |
|---|---|---|---|
| Hosted model | **95.8%** | 2.77 s | $0.29 |
| Base model, few-shot | 78.9% | 2.20 s | $0.022 at full use + $439/month GPU |
| LoRA | 83.1% | 2.02 s | $0.016 + $439/month |
| LoRA, 4-bit | 82.4% | **1.33 s** | $0.014 + $439/month |

The hosted model stays: it is 12.7 [+7.0, +19.0] points more accurate, and an always-on GPU only becomes cheaper above **about 1.5 million questions a month**, roughly 2,000 an hour around the clock. The 4-bit LoRA is the only option inside the 1.5 s SLO, so it is pinned as the self-hosted fallback (`configs/serving.yaml`).

## 10. What did not work

- **INT8 everywhere** (Section 6): over the quality bar on CPU and GPU, slower than FP16 on GPU, and two failed attempts on the way (an mAP of 0 from quantizing every node, a TensorRT parse failure).
- **Fine-tuning the detector against blur** (D36, E-060, E-061): a blur-augmented fine-tune gained 4.4 points on strongly blurred camera frames against a no-blur control, but the recipe itself lost 2 to 7 points everywhere, including clean video. Without the control the result would have read as a success (+1.6 over the original on the target). A follow-up with one change, a frozen backbone, chosen before running (D39), removed most of the recipe's loss but also most of the blur gain (E-067). Neither model was deployed.
- **CUSUM** added false alarms and no speed on these drifts (Section 8).
- **Fine-tuning the planner** closed only about a quarter of the gap to the hosted model (4.2 of 16.9 points) (Section 9).
- **Batching on the GPU** was capped by CPU-side numpy pre- and post-processing; TensorRT FP16 showed a reproducible, still unexplained dip at batch 4 (E-064, E-065).
- **Operational failures that changed procedures**: a full disk aborted a ramp (E-038); AWS denied the GPU quota, so GPU work moved to Runpod pods, which run containers without Docker and therefore only model-level scenarios (D35, D37); vLLM failed to start on a pod with no CUDA compiler (E-066); one pod never got a public SSH port (E-063); LoRA training ran out of memory on the 248k-token vocabulary's logits (E-071). Each is in the log with the fix and a test.

## 11. Limitations

- **Synthetic question data.** The eval questions were written for a synthetic database whose schema the prompt describes in full; real users' questions would be messier, so 95.8% is an upper-end estimate. The planner's training data is synthetic too.
- **Small eval sets.** 142 test questions give intervals of about plus or minus 6 to 9 points; categories have 8 to 16 questions each, so per-category differences are not established. The detection subset is 573 images, and MOT17 frames within a sequence are correlated, so its intervals are optimistic.
- **Eval label biases.** MOT17 labels box people behind occluders at full extent while detectors box the visible part, which caps mAP@50-95 on those frames; its positives follow a different convention from MOTChallenge's own tables.
- **Self-measured, narrow hardware.** One laptop, one AWS instance type and one A40 model. The laptop is not a dedicated host. No result is compared across machines as a ratio.
- **No GPU end to end.** GPU numbers are model-level only, so there is no rated capacity on a GPU.
- **Synthetic drift.** Photometric transforms are not night, fog or a dirty lens; the false-alarm rate comes from four clean scenes on looping clips and is a lower bound for real cameras.
- **Self-hosted cost** assumes the GPU is fully used and leaves out the network between the query service and the GPU, and on-call time.

## 12. Future work

From the [roadmap](ROADMAP.md), in the order I would do it with two more weeks:

1. **GPU end to end**: run the stack on a full GPU VM for a GPU ingest ramp, a worker profile and a rated capacity, then move letterbox and NMS to the GPU, the measured limit of the exported backends.
2. **Storage benchmarks**: partitioned Postgres vs TimescaleDB vs ClickHouse at 10M and 100M detections, with plan-to-SQL result equivalence across backends, to settle D1 with numbers.
3. **Visual search**: CLIP embeddings on sampled frames and hybrid structured + visual questions over the same `QueryPlan`, with a retrieval eval (recall@k, nDCG).
4. **Queue and reliability**: Redis Streams or Kafka against the Postgres queue at higher camera counts, and fault tests for redelivery and backpressure.
5. **Kubernetes and cloud scale**: a Helm chart, autoscaling workers on backlog, Terraform for the AWS stack and a full-scale cost per camera-hour.
6. **CD with an eval gate**, so a model or prompt change that regresses the frozen eval cannot ship.
