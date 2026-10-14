# Plan (compact)

> Location in repo: `docs/PLAN.md`. Tick tasks as they complete. Details, prompts and done criteria for each task are in `docs/PLAN_FULL.md`. Task IDs are stable: deferred tasks keep their numbers.

**Scope lock (rule).** This plan is capped. Before and after the checkpoint below, no new task is added without removing an equal amount of planned work, and the swap is recorded under "Plan revisions". Ideas that come up mid-task go to `EXPERIMENTS.md` or the "Deferred (not planned)" list, not into the plan.

**Checkpoint C1 (end of this plan).** Every task below is ticked: the system runs end to end on CPU with a UI and dashboards; it has a measured baseline with a bottleneck analysis; structured-query and detection quality are evaluated with bootstrap confidence intervals; a precision ladder shows accuracy against latency; drift detection is scored on delay against false-alarm rate; and the README and write-up present it all. At C1, stop and decide whether to continue at all. Deferred work comes back only through the scope-lock rule.

**Estimate.** 29 tasks remain, about 75 to 80 hours of the user's time: roughly 6 weeks at 12 to 15 hours per week. Assumption: the coding agent implements; the user's hours go to specifying, reviewing, running, verifying and answering each task's "Understand it" question (tasks 0.5, 0.6 and 1.1 each took about 1 to 1.5 hours of agent time). The biggest schedule risk is the GPU quota: spot is denied by the account's SCP, and the on-demand G quota is 0 (4 vCPUs requested 2026-10-13; D28): Phase 5 and 9.5 need a GPU, and both have a CPU fallback in `PLAN_FULL.md`.

**Order of work.** Phase 1, then 2.1 to 2.3, then Phase 3, then 2.4 to 2.6 (first GPU session), then Phase 5, then Phase 9, then Phase 11. Phase numbers are kept for stable IDs, so this order differs from the numbering.

**Understand it.** Each task carries one question the user must be able to answer with specifics before the task is ticked. This is how the user verifies they understand what the agent built.

## Phase 0: Foundation
- [x] 0.1 Repo scaffold: uv, src layout, Makefile, ruff, mypy, pytest markers, pre-commit, DVC with local remote (`make data-pull`)
  - Understand it: what does `make check` run, and what would CI catch that pre-commit does not?
- [x] 0.2 Docs skeleton with the fixed structure; decisions D1 to D11 as proposed
  - Understand it: why is `EXPERIMENTS.md` append-only while `BENCHMARKS.md` is curated?
- [x] 0.3 Shared libraries: config, JSON logging, OpenTelemetry setup, IDs, injectable clock (the `Tracker` was removed in 0.7)
  - Understand it: in what order are config layers applied, and why is the clock injected?
- [x] 0.4 CI: lint, typecheck, unit tests
  - Understand it: which job runs against the compose stack, and what happens to the stack when a test fails?
- [x] 0.5 Compose stack: Postgres + pgvector, object store (SeaweedFS, D12), OTel collector, Prometheus, Tempo, Grafana (Redis and MLflow removed in 0.7)
  - Understand it: what does `make up` wait for, and how does a trace get from a service to Grafana?
- [x] 0.6 Query assets built from scratch (no legacy code): draft `QueryPlan`, sqlglot SQL validator, few-shot (question, gold plan, gold SQL) triples, prompt templates
  - Understand it: name three attacks the SQL validator blocks, and why it executes regenerated SQL instead of the input.
- [x] 0.7 Trim the stack: remove Redis, MLflow and the MLflow/W&B `Tracker` (no measured need; D16)
  - Understand it: what does each remaining service do in this project, and which measurement would bring Redis back?

## Phase 1: End-to-end baseline (CPU)
- [x] 1.1 Dataset download scripts with checksums and license notes
  - Understand it: how is each clip pinned, and why is VIRAT opt-in?
- [x] 1.2 Loadgen v0: N virtual cameras at fixed FPS over REST
  - Understand it: how do you know the load generator is not the bottleneck?
- [x] 1.3 Ingest v0: REST, JPEG to object store, frame rows to Postgres, idempotent on `(camera_id, run_id, seq)` (D17)
  - Understand it: what happens when the same frame is sent twice, or the object store write succeeds but the DB insert fails?
- [x] 1.4 Schema v1 and migrations (SHRINK: no `embeddings` table)
  - Understand it: why are detections and events append-only while tracks are updated, and what does the read-only role prevent?
- [x] 1.5 Worker v0: polls Postgres with `SKIP LOCKED`, in-process PyTorch YOLO11 at batch size 1 (SHRINK: no CLIP)
  - Understand it: what does `SKIP LOCKED` guarantee with two workers, and what breaks tracking if one camera's frames are processed out of order?
- [x] 1.6 Tracking (ByteTrack) and zone enter/exit events (SHRINK: loiter optional)
  - Understand it: how is a zone event decided, and what happens at the boundary when a box jitters?
- [x] 1.7 Query service v0: typed `QueryPlan` planner, validation and repair, Postgres compiler, SQL validator, read-only execution, `query_log` (SHRINK: JSON responses, no semantic search, no cache)
  - Understand it: trace one question from text to SQL; where can it fail, and what does the user see for each failure?
- [x] 1.8 Minimal UI: live camera tiles, query box with answer and chart or table, plan JSON and compiled SQL panel (SHRINK: no feedback buttons, no semantic mode)
  - Understand it: what does the UI call, and how does it show an abstention or a validator rejection?
- [x] 1.9 `make demo` end to end on CPU, plus integration test
  - Understand it: what exactly does the demo prove, and what does it fake (LLM, models)?

## Phase 2: Observability and baseline
- [x] 2.1 Tracing and metrics: frame and query traces, freshness, throughput, backlog, stage latencies (SHRINK: GPU metrics only during the GPU session)
  - Understand it: how is freshness measured, and which span would you open first if p95 freshness doubled?
- [x] 2.2 Grafana dashboards as code: Pipeline, Inference, Query, plus Drift (filled in 9.4); designed to be screenshotted
  - Understand it: for each panel, what decision would you make from it?
- [x] 2.3 Benchmark framework: scenario YAML, runner, results JSON schema, warmup and repetitions
  - Understand it: why exclude warmup, how many repetitions, and how is variance reported?
- [ ] 2.4 GPU session on AWS: one tagged on-demand instance by script (spot is denied by the account's SCP, D28), Budgets alarm, verified teardown (SHRINK: no Terraform, k3s, ECR or S3 remote). Built and budget created 2026-10-13; waiting for the on-demand G quota for the real session
  - Understand it: how do you know nothing is still billing after a session, and what does an hour cost?
- [ ] 2.5 Ingest ramp (rated capacity) on CPU and one GPU run, plus a small structured-query load (SHRINK)
  - Understand it: what is the definition of rated capacity, and where did the ramp break?
- [ ] 2.6 Baseline report and bottleneck analysis (profiling)
  - Understand it: rank the top three bottlenecks with evidence, and say which one you would fix first and why.

## Phase 3: Evaluation harness
- [ ] 3.1 Structured-query dataset: about 120 questions incl. `unsupported` and adversarial, gold results, plans and SQL, DVC-tracked DB snapshot, frozen dev/test split (SHRINK). Built 2026-10-14 (129 items on a synthetic snapshot, D29, E-029); waiting for the user's review of the test split and a dev sample (`make eval-review`)
  - Understand it: how were gold answers produced and checked, and how do you know the test split has not leaked into prompts?
- [ ] 3.2 Execution-accuracy scorer with bootstrap confidence intervals, error taxonomy, a "measures / does not measure" note per metric, red-team subset
  - Understand it: what is a 95% bootstrap CI here, why is it wide, and what does execution accuracy miss?
- [ ] 3.4 Detection quality subset (mAP@50 and mAP@50-95 with bootstrap CIs)
  - Understand it: what does mAP@50-95 reward that mAP@50 does not, and how representative is the subset of the clips?
- [ ] 3.5 Baseline eval numbers and `make eval` (structured and detection suites; two LLMs compared on dev)
  - Understand it: is the difference between the two LLMs real given the CIs, and what are the top error classes?

## Phase 5: Model internals
- [ ] 5.1 Export YOLO11 to ONNX with parity tests (SHRINK: no Triton, no CLIP)
  - Understand it: what tolerance did parity use and why, and where do ONNX and PyTorch outputs differ?
- [ ] 5.2 Precision ladder: PyTorch, ORT, TensorRT FP16, TensorRT INT8, each with mAP (CI) and latency (CPU ladder as fallback)
  - Understand it: why does INT8 need calibration, what did it cost in mAP, and is that drop inside the CI?
- [ ] 5.3 Batch-size sweep in-process: throughput against latency (SHRINK: no Triton dynamic batching)
  - Understand it: why do throughput and latency both rise with batch size, and which point meets the freshness SLO?
- [ ] 5.5 Worker on the best backend in-process; re-run ramp; decide D5 (SHRINK: no Triton)
  - Understand it: how much of the model-only speedup survived end to end, and why not all of it?

## Phase 9: Drift detection
- [ ] 9.1 Per-camera drift statistics (input: brightness, blur, noise; output: detections per frame, confidence and class mix)
  - Understand it: what is the reference window, and which statistic reacts to darkening vs a class-mix change?
- [ ] 9.2 Drift injection scenarios with known, seeded onsets, and mAP under each drift
  - Understand it: why must onsets be known and seeded, and which drift hurt mAP most?
- [ ] 9.3 Sequential change detection: CUSUM vs fixed threshold, detection delay vs false-alarm rate across seeds
  - Understand it: what does CUSUM accumulate, which knob trades delay for false alarms, and what did the curve show?
- [ ] 9.4 Drift dashboard and runbook procedure
  - Understand it: an alert fires on one camera at 3 a.m.; what do you check, in order?
- [ ] 9.5 Detector fine-tune on drifted data: before/after mAP on drifted and clean sets (SHRINK: no sweep, no W&B)
  - Understand it: did fine-tuning help the drifted set without hurting the clean set, and why is augmentation not real domain shift?

## Phase 11: Packaging
- [ ] 11.1 README with headline numbers, UI and dashboard screenshots, diagram
  - Understand it: which three numbers lead the README, and what hardware and CIs go with each?
- [ ] 11.2 Technical write-up
  - Understand it: what did not work, and what would you do next with two more weeks?
- [ ] 11.3 Demo: 2-minute screen recording (SHRINK: no live demo)
  - Understand it: what is the one thing a viewer should remember from the recording?
- [ ] 11.4 Fresh-clone reproduction test
  - Understand it: which steps needed fixing on a clean machine, and is anything still manual?

## Interview defense

The finished project should let the user answer each of these with specifics from their own measurements:

1. What is the system's rated capacity on CPU and on the GPU you used, how did you define it, and what limited it?
2. Where does a frame spend its time from camera to stored detections, and how do you know (traces, profiles)?
3. How accurate is the question answering, with what confidence interval, on what test set, and how did you keep that set clean?
4. What are the most common failure modes of the planner, and what would you change first to reduce them?
5. Why a typed query plan instead of letting the LLM write SQL, and what does that choice cost (unsupported questions)?
6. How is untrusted user input kept from damaging the database? Walk through the layers.
7. What did TensorRT FP16 and INT8 buy in latency and cost in mAP, and which would you ship?
8. How did the speedup measured on the model alone compare with the end-to-end gain, and why do they differ?
9. How fast does drift detection fire, at what false-alarm rate, and why CUSUM over a fixed threshold (or not)?
10. Which tools did you remove or not use (Redis, Kafka, MLflow, Triton, Kubernetes), and what measurement would make you add each one?

## Plan revisions

- **2026-10-04:** AWS is the default provider (D9 narrowed to long GPU runs), with tagged resources and verified teardown. Kafka replaces Redpanda as the D3 candidate, and D3 also covers a detections/events stream. W&B (training) and MLflow (eval, registry) sit behind a `Tracker` interface; DVC tracks large artifacts. The default structured path is now a typed `QueryPlan` compiled to SQL per backend, with free-form SQL as an alternative and the choice recorded in D11. Added 9.5 (detector fine-tune on drifted data, Phase 9 now two weeks) and stretch 10.5 (retraining loop, D10); total about 18 weeks.
- **2026-10-06:** The local object store is SeaweedFS instead of MinIO, which no longer publishes images (D12). Task 0.6 no longer ports the old smart-vision prototype: by the user's decision, the query assets (draft `QueryPlan`, SQL validator, few-shot bank, prompt templates) are built from scratch (D13, D14).
- **2026-10-06 (re-scope):** The 18-week plan is cut to a finishable core of about 6 part-time weeks, chosen for hiring signal and learning per hour, because the semester is shared with a research paper and quant work. Kept: the end-to-end baseline with a UI and good dashboards (the demo and write-up are visual for recruiters who skim), observability, the benchmark runner and bottleneck analysis, the evaluation harness with bootstrap confidence intervals and an error taxonomy, a minimal model-internals track (ONNX parity, precision ladder, batch sweep), drift detection framed as sequential change detection, and packaging. Deferred: storage and queue benchmarks, gRPC, Kubernetes, autoscaling, chaos and capacity tests, CLIP with semantic and hybrid search (retrieval labels are the most expensive labeling in the plan), the free-form SQL engine, LLM serving and LoRA, full Terraform, CD and the retraining stretch. Decisions that depended on deferred tasks now have defaults marked "accepted (default, not benchmarked)" (D1 to D4, D6 to D11). Redis, MLflow and W&B are removed because no kept task measures a need for them (new task 0.7, D16). Tally over the original 65 tasks: 24 kept (7 already done), 14 shrunk, 27 deferred; plus the new task 0.7. A scope lock and checkpoint C1 now head this file.

## Deferred (not planned)

Kept unchanged so nothing is lost. A deferred task comes back only through the scope-lock rule. Details stay in `PLAN_FULL.md`, marked deferred.

- [ ] 3.3 Retrieval eval set and metrics (recall@k, nDCG)
- [ ] 4.1 Synthetic detection generator fitted to real replay data (outputs DVC-tracked)
- [ ] 4.2 Storage workload: ingest benchmark and query suite
- [ ] 4.3 Benchmark Postgres partitioned, TimescaleDB, ClickHouse at 10M and 100M rows (1B optional)
- [ ] 4.4 Vector store benchmark: pgvector vs Qdrant, including filtered search
- [ ] 4.5 Decide D1 and D2 (incl. compiler complexity and plan-to-SQL equivalence), migrate, re-evaluate
- [ ] 5.4 Profile and optimize pre- and post-processing (DALI, or custom C++/CUDA if justified)
- [ ] 6.1 Queue keyed by camera: Redis Streams vs Kafka, plus event-stream question; decide D3
- [ ] 6.2 gRPC ingest; decide D4
- [ ] 6.3 Reliability: idempotency, backpressure, dead letters, graceful shutdown, fault tests
- [ ] 6.4 Helm chart and kind smoke test in CI
- [ ] 6.5 Autoscaling with KEDA (workers) and HPA (query service)
- [ ] 6.6 Capacity and chaos tests on the GPU cluster; decide D6
- [ ] 6.7 Embedding sampling policy; decide D8
- [ ] 7.1 Hybrid search: shared `QueryPlan` with visual query, filtered ANN or prefilter + rerank
- [ ] 7.2 Query engine improvements and D11: few-shot retrieval, schema and plan pruning, repair, caching; free-form SQL engine; compare once on test
- [ ] 7.3 Re-evaluate all modes; error analysis
- [ ] 8.1 vLLM serving with prefix caching and structured outputs (overhead measured); load test vs llama.cpp/Ollama
- [ ] 8.2 Synthetic (question, QueryPlan) training data with filtering and leakage checks (DVC)
- [ ] 8.3 LoRA fine-tune on plan JSON with ablations, W&B tracking (optional SQL adapter for D11)
- [ ] 8.4 Quantization; multi-LoRA (plan + SQL adapters) or adapter vs merged overhead
- [ ] 8.5 Decide D7: hosted vs few-shot vs LoRA on accuracy, p95, cost
- [ ] 10.1 Full AWS stack in Terraform (EC2 spot GPU + k3s, S3, ECR, IAM, Budgets; EKS only if justified)
- [ ] 10.2 CD pipeline with eval gate; champion in MLflow registry
- [ ] 10.3 Full-scale cloud benchmark; cost per camera-hour
- [ ] 10.4 Production readiness checklist in RUNBOOK
- [ ] 10.5 (stretch, unscheduled) Orchestrated retraining loop; decide D10 (Prefect or Airflow)
