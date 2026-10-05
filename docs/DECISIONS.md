# Decisions

> Location in repo: `docs/DECISIONS.md`. One record per choice a reviewer could reasonably ask "why?" about. Raw runs live in `docs/EXPERIMENTS.md`; the design they shape is in `docs/ARCHITECTURE.md`. Never delete a record: supersede it with a new one and update the old record's status.

## Template

```
## Dn: Title
Status: proposed | accepted | superseded by Dn    Date: YYYY-MM-DD
Context: the problem and constraints.
Options: candidates considered.
Decision: what was chosen.
Evidence: E-ids and key numbers.
Trade-offs and consequences: what we gave up, what to revisit and when.
```

---

## D1: Detection store
Status: proposed    Date: 2026-10-05
Context: Detections and events are append-only, high-volume writes: about 4,000 rows/s and 345M rows/day at 100 cameras x 5 FPS, kept for 30 days. Reads are analytical time-range aggregates filtered by camera, class, zone and time (ARCHITECTURE 2, 4.6). Mutable track state may stay in Postgres because column stores handle updates badly. The LLM emits a `QueryPlan`, not SQL, so model accuracy does not depend on the SQL dialect. Each viable backend instead needs its own plan-to-SQL compiler, and the same plan must return the same result set on every backend.
Options: Postgres partitioned by time with BRIN indexes; TimescaleDB (hypertables, compression); ClickHouse (MergeTree).
Decision: pending (Phase 4, task 4.5).
Evidence: pending. To measure: ingest rows/s, query p95 by query class, bytes/row at 10M and 100M rows (1B optional), compiler complexity, and plan-to-SQL result equivalence on the structured dev split's gold plans.
Trade-offs and consequences: pending. Known tension: a column store likely wins on aggregates and compression but adds a second database to operate next to the Postgres metadata DB.

## D2: Vector store
Status: proposed    Date: 2026-10-05
Context: CLIP frame embeddings (512-d), written according to the D8 sampling policy, are searched by text with optional camera and time filters. Hybrid queries need filtered ANN search, or an SQL prefilter followed by vector rerank, across a range of filter selectivities (ARCHITECTURE 4.6, 4.7).
Options: pgvector HNSW (same Postgres as the metadata DB); Qdrant.
Decision: pending (Phase 4, task 4.5).
Evidence: pending. To measure: recall@10 vs latency and QPS at 1M and 10M vectors, and filtered search across selectivities.
Trade-offs and consequences: pending. pgvector means one fewer system to run and transactional joins with metadata. Qdrant may handle filtered search and scale better.

## D3: Queue and event stream
Status: proposed    Date: 2026-10-05
Context: The frame queue decouples ingest from GPU work and absorbs bursts. It must be keyed by `camera_id` so one consumer processes a camera's frames in order, because tracking depends on that order. It delivers at least once, with acks only after a durable write, a dead-letter stream, and consumer lag exported to drive KEDA autoscaling (ARCHITECTURE 4.3). A separate question is whether detections and events should also flow through a Kafka topic that feeds the detection store (ClickHouse Kafka engine or a consumer) and the drift monitor, for decoupling and replay. The two questions are decided independently.
Options: frame queue: Redis Streams; Apache Kafka (KRaft, single broker for local runs and benchmarks). Event stream: Kafka topic; no stream, with workers writing directly to the stores.
Decision: pending (Phase 6, task 6.1). Kafka is adopted only with benchmark evidence. "Redis Streams suffices at this scale" is a valid outcome for both questions.
Evidence: pending. To measure: throughput, per-key ordering, replay, rebalancing behaviour and operational cost. For the event stream: decoupling and replay value against the added operations work.
Trade-offs and consequences: pending.

## D4: Ingest transport
Status: proposed    Date: 2026-10-05
Context: Cameras send 1280x720 JPEGs (~100 to 200 KB) at 5 FPS each. Phase 1 uses REST multipart. The public REST API is unaffected by this choice; only the loadgen-to-ingest path changes (ARCHITECTURE 4.2, 6).
Options: REST multipart; gRPC client streaming (`Ingest.StreamFrames`).
Decision: pending (Phase 6, task 6.2).
Evidence: pending. To measure: frames/s per core, p99 latency, CPU per frame.
Trade-offs and consequences: pending.

## D5: Inference backend and precision
Status: proposed    Date: 2026-10-05
Context: YOLO11 detection and the CLIP image and text encoders run in Triton from Phase 5. Phase 1 runs them in-process with PyTorch at batch size 1. Any speedup must report its quality cost (ARCHITECTURE 4.5).
Options: PyTorch eager; ONNX Runtime CUDA; TensorRT FP16; TensorRT INT8. Also dynamic batching and instance count settings, and GPU pre- and post-processing (DALI, TensorRT NMS) if profiling shows they are CPU-bound.
Decision: pending (Phase 5, task 5.5).
Evidence: pending. To measure: img/s, latency, GPU memory, mAP and recall@k delta against PyTorch.
Trade-offs and consequences: pending. TensorRT engines must be built on the target GPU.

## D6: Service boundaries
Status: proposed    Date: 2026-10-05
Context: One codebase (modular monolith) with several deployables. A service boundary needs a measured reason, such as a different scaling profile (ARCHITECTURE 1, 10). Each deployable is a Dockerfile target, so splitting or merging services is a deployment change.
Options: deploy each of loadgen, ingest-gateway, inference-worker, query-service, llm-service and drift-monitor separately, or merge some of them.
Decision: pending (Phase 6, task 6.6). Merging is a valid outcome.
Evidence: pending. To measure: CPU, GPU and memory profiles and scaling behaviour per component on the GPU cluster.
Trade-offs and consequences: pending.

## D7: LLM strategy
Status: proposed    Date: 2026-10-05
Context: The structured-query planner turns natural language into a `QueryPlan` under constrained decoding. It is self-hosted on vLLM with prefix caching and structured outputs, with a hosted LLM as the circuit-breaker fallback (ARCHITECTURE 4.7, 4.8).
Options: hosted model; small open base model with few-shot prompting; LoRA adapter on plan JSON targets; quantized LoRA (AWQ or GPTQ base, multi-LoRA).
Decision: pending (Phase 8, task 8.5).
Evidence: pending. To measure: structured-query execution accuracy, p95 latency, cost per 1,000 queries.
Trade-offs and consequences: pending.

## D8: Embedding sampling
Status: proposed    Date: 2026-10-05
Context: Embedding every analyzed frame (5 FPS per camera) costs GPU time and vector storage. The default is 1 FPS per camera (ARCHITECTURE 2).
Options: every frame; fixed rate; scene-change triggered.
Decision: pending (Phase 6, task 6.7).
Evidence: pending. To measure: retrieval recall@k and nDCG against GPU time and storage.
Trade-offs and consequences: pending.

## D9: Provider for long GPU benchmark runs
Status: proposed    Date: 2026-10-05
Context: AWS is the default provider for all infrastructure (S3, ECR, IAM, Budgets, EC2 spot GPU instances running k3s), in the project's single configured Region. That part is not up for decision. D9 only decides whether long GPU benchmark runs move to a cheaper GPU provider. GPU time is rented per session through scripts, with teardown verified by tag (ARCHITECTURE 9, 12).
Options: AWS EC2 spot g5 or g6 in the project Region; cheaper GPU clouds for long runs only.
Decision: pending (Phase 2, task 2.4).
Evidence: pending. To compare: cost per hour and spot availability for g5 and g6 in the project Region against alternatives.
Trade-offs and consequences: pending. A second provider adds setup and cross-provider comparison caveats, since numbers are only comparable on the same hardware.

## D10: Orchestrator for the retraining loop (stretch)
Status: proposed    Date: 2026-10-05
Context: Stretch task 10.5 chains a drift alert or manual trigger, frame sampling, a detector fine-tune (task 9.5 script), the eval gate and MLflow registry registration. It must run on the compose CPU stack with tiny models and on k3s (ARCHITECTURE 4.9).
Options: Prefect; Airflow.
Decision: pending (Phase 10, task 10.5; unscheduled).
Evidence: pending. To compare: operational weight on compose and k3s, DAG expressiveness, implementation effort.
Trade-offs and consequences: pending.

## D11: Structured query interface
Status: proposed    Date: 2026-10-05
Context: Structured questions go through a `StructuredQueryEngine` interface, so engines can be swapped and compared on the same eval set. The typed plan bounds the output space: validation is exhaustive, values are always bound as parameters, accuracy does not depend on the SQL dialect, and training targets are short. The cost is expressiveness: out-of-schema questions must abstain with `unsupported` (ARCHITECTURE 4.7, 11).
Options: typed `QueryPlan` compiled to parameterized SQL (default); free-form SQL with `sqlglot` validation and one self-correction attempt; agent loop with tool calling (optional prototype); hybrid (plan by default, SQL fallback for `unsupported`).
Decision: pending (Phase 7, task 7.2). The typed plan is the default until this record is accepted.
Evidence: pending. To compare: execution accuracy (all questions and expressible ones), abstention correctness, p95 latency, tokens and cost, plan coverage, red-team attack success rate.
Trade-offs and consequences: pending.
