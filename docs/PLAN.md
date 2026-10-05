# Plan (compact)

> Location in repo: `docs/PLAN.md`. Tick tasks as they complete. Details, prompts and done criteria for each task are in `docs/PLAN_FULL.md`. Week estimates assume roughly 15 hours per week (about 18 weeks in total) and are guides, not deadlines.

## Phase 0: Foundation (week 1)
- [x] 0.1 Repo scaffold: uv, src layout, Makefile, ruff, mypy, pytest markers, pre-commit, DVC with local remote (`make data-pull`)
- [x] 0.2 Docs skeleton with the fixed structure; decisions D1 to D11 as proposed
- [x] 0.3 Shared libraries: config, JSON logging, OpenTelemetry setup, IDs, injectable clock, `Tracker` (MLflow, W&B, no-op)
- [x] 0.4 CI: lint, typecheck, unit tests
- [ ] 0.5 Compose stack: Postgres + pgvector, MinIO, Redis, OTel collector, Prometheus, Grafana, MLflow
- [ ] 0.6 Port reusable assets from the old project; few-shots as (question, gold plan, gold SQL) triples; SQL validator

## Phase 1: Naive end-to-end baseline (weeks 2 to 3)
- [ ] 1.1 Dataset download scripts with checksums and license notes
- [ ] 1.2 Loadgen v0: N virtual cameras at fixed FPS over REST
- [ ] 1.3 Ingest v0: REST, JPEG to MinIO, frame rows to Postgres
- [ ] 1.4 Schema v1 and migrations
- [ ] 1.5 Worker v0: polls DB, in-process PyTorch YOLO11 and CLIP, batch size 1
- [ ] 1.6 Tracking (ByteTrack) and zone events
- [ ] 1.7 Query service v0: typed `QueryPlan` planner, Postgres compiler, SQL validator, read-only role; semantic search
- [ ] 1.8 Minimal frontend (debug panel: plan JSON and compiled SQL)
- [ ] 1.9 `make demo` end to end on CPU, plus integration test

## Phase 2: Observability and baseline load test (week 4)
- [ ] 2.1 Tracing and metrics across all services, including freshness and query stage timings
- [ ] 2.2 Grafana dashboards as code
- [ ] 2.3 Benchmark framework: scenario YAML, runner, results JSON schema
- [ ] 2.4 GPU bench environment on AWS: Terraform (S3 + DVC remote, ECR, IAM, Budgets, EC2 spot + k3s), tagged, teardown verified (D9)
- [ ] 2.5 Ingest ramp test (rated capacity) and mixed query load test, CPU and GPU
- [ ] 2.6 Baseline report and bottleneck analysis (profiling)

## Phase 3: Evaluation harness (week 5)
- [ ] 3.1 Structured-query dataset: 300+ questions incl. `unsupported`, gold results, plans and SQL, DVC-tracked DB snapshot, dev/test split
- [ ] 3.2 Execution-accuracy scorer (all and expressible, coverage), plan and SQL error taxonomy, MLflow logging, red-team subset
- [ ] 3.3 Retrieval eval set and metrics (recall@k, nDCG)
- [ ] 3.4 Detection quality subset (mAP)
- [ ] 3.5 Baseline eval numbers and `make eval` (suites: structured, retrieval, detection)

## Phase 4: Storage (weeks 6 to 7)
- [ ] 4.1 Synthetic detection generator fitted to real replay data (outputs DVC-tracked)
- [ ] 4.2 Storage workload: ingest benchmark and query suite
- [ ] 4.3 Benchmark Postgres partitioned, TimescaleDB, ClickHouse at 10M and 100M rows (1B optional)
- [ ] 4.4 Vector store benchmark: pgvector vs Qdrant, including filtered search
- [ ] 4.5 Decide D1 and D2 (incl. compiler complexity and plan-to-SQL equivalence), migrate, re-evaluate

## Phase 5: Inference optimization (weeks 8 to 9)
- [ ] 5.1 Export YOLO and CLIP to ONNX; Triton model repository; parity tests
- [ ] 5.2 Precision ladder: PyTorch, ORT, TensorRT FP16, TensorRT INT8
- [ ] 5.3 Dynamic batching and instance count sweep
- [ ] 5.4 Profile and optimize pre- and post-processing (DALI, or custom C++/CUDA if justified)
- [ ] 5.5 Worker on Triton; re-run ramp; decide D5

## Phase 6: Scaling the pipeline (weeks 10 to 11)
- [ ] 6.1 Queue keyed by camera: Redis Streams vs Kafka, plus event-stream question; decide D3
- [ ] 6.2 gRPC ingest; decide D4
- [ ] 6.3 Reliability: idempotency, backpressure, dead letters, graceful shutdown, fault tests
- [ ] 6.4 Helm chart and kind smoke test in CI
- [ ] 6.5 Autoscaling with KEDA (workers) and HPA (query service)
- [ ] 6.6 Capacity and chaos tests on the GPU cluster; decide D6
- [ ] 6.7 Embedding sampling policy; decide D8

## Phase 7: Search quality (week 12)
- [ ] 7.1 Hybrid search: shared `QueryPlan` with visual query, filtered ANN or prefilter + rerank
- [ ] 7.2 Query engine improvements and D11: few-shot retrieval, schema and plan pruning, repair, caching; free-form SQL engine; compare once on test
- [ ] 7.3 Re-evaluate all modes; error analysis

## Phase 8: LLM serving and LoRA (weeks 13 to 14)
- [ ] 8.1 vLLM serving with prefix caching and structured outputs (overhead measured); load test vs llama.cpp/Ollama
- [ ] 8.2 Synthetic (question, QueryPlan) training data with filtering and leakage checks (DVC)
- [ ] 8.3 LoRA fine-tune on plan JSON with ablations, W&B tracking (optional SQL adapter for D11)
- [ ] 8.4 Quantization; multi-LoRA (plan + SQL adapters) or adapter vs merged overhead
- [ ] 8.5 Decide D7: hosted vs few-shot vs LoRA on accuracy, p95, cost

## Phase 9: Drift detection (weeks 15 to 16)
- [ ] 9.1 Drift statistics per camera (input, output, embedding)
- [ ] 9.2 Drift injection scenarios with known onsets
- [ ] 9.3 Alert rules; detection delay and false-alarm rate
- [ ] 9.4 Drift dashboard and runbook procedure
- [ ] 9.5 Detector fine-tune on drifted data: before/after mAP (drifted and clean), W&B sweep, weights in DVC

## Phase 10: Cloud and continuous delivery (week 17)
- [ ] 10.1 Full AWS stack in Terraform (EC2 spot GPU + k3s, S3, ECR, IAM, Budgets; EKS only if justified)
- [ ] 10.2 CD pipeline with eval gate; champion in MLflow registry
- [ ] 10.3 Full-scale cloud benchmark; cost per camera-hour
- [ ] 10.4 Production readiness checklist in RUNBOOK
- [ ] 10.5 (stretch, unscheduled) Orchestrated retraining loop; decide D10 (Prefect or Airflow)

## Phase 11: Packaging (week 18)
- [ ] 11.1 README with headline numbers and diagram
- [ ] 11.2 Technical write-up
- [ ] 11.3 Demo video; optional cost-capped live demo
- [ ] 11.4 Fresh-clone reproduction test

## Cut lines (priority if time runs short)
1. **Core, portfolio-ready on its own:** Phases 0 to 5, tasks 6.1 to 6.4, Phase 11.
2. **Strong additions:** 6.5 to 6.7, Phase 8, Phase 9 (including 9.5).
3. **Polish:** Phase 7, tasks 10.1 to 10.4.
4. **Stretch (not scheduled in the week estimates):** 10.5 orchestrated retraining loop.

## Plan revisions

- **2026-10-04:** AWS is the default provider (D9 narrowed to long GPU runs), with tagged resources and verified teardown. Kafka replaces Redpanda as the D3 candidate, and D3 also covers a detections/events stream. W&B (training) and MLflow (eval, registry) sit behind a `Tracker` interface; DVC tracks large artifacts. The default structured path is now a typed `QueryPlan` compiled to SQL per backend, with free-form SQL as an alternative and the choice recorded in D11. Added 9.5 (detector fine-tune on drifted data, Phase 9 now two weeks) and stretch 10.5 (retraining loop, D10); total about 18 weeks.
