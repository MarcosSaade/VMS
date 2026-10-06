# Decisions

> Location in repo: `docs/DECISIONS.md`. One record per choice a reviewer could reasonably ask "why?" about. Raw runs live in `docs/EXPERIMENTS.md`; the design they shape is in `docs/ARCHITECTURE.md`. Never delete a record: supersede it with a new one and update the old record's status.

## Template

```
## Dn: Title
Status: proposed | accepted | accepted (default, not benchmarked) | superseded by Dn    Date: YYYY-MM-DD
Context: the problem and constraints.
Options: candidates considered.
Decision: what was chosen.
Evidence: E-ids and key numbers.
Trade-offs and consequences: what we gave up, what to revisit and when.
```

---

## D1: Detection store

Status: accepted (default, not benchmarked)    Date: 2026-10-06
Context: Detections and events are append-only, high-volume writes: about 4,000 rows/s and 345M rows/day at 100 cameras x 5 FPS, kept for 30 days. Reads are analytical time-range aggregates filtered by camera, class, zone and time (ARCHITECTURE 2, 4.6). Mutable track state may stay in Postgres because column stores handle updates badly. The LLM emits a `QueryPlan`, not SQL, so model accuracy does not depend on the SQL dialect. Each viable backend instead needs its own plan-to-SQL compiler, and the same plan must return the same result set on every backend.
Options: Postgres partitioned by time with BRIN indexes; TimescaleDB (hypertables, compression); ClickHouse (MergeTree).
Decision: Postgres (the same instance as the metadata DB), detections and events in plain tables with simple time and camera indexes; time partitioning only if 2.6 shows it is needed. Rationale: one database to operate, and Phase 4 was deferred in the 2026-10-06 re-scope; at demo and benchmark scale (minutes of clips, not 345M rows a day) Postgres is not the bottleneck to study.
Evidence: none; Phase 4 deferred.
Trade-offs and consequences: We give up the column-store comparison and its compression and aggregate-speed story. Evidence that would change it: the 2.6 bottleneck analysis or the 3.5 query p95 showing detection-table queries as a top bottleneck, or a retention target of days at 100 cameras. Then revive 4.2 to 4.5.

## D2: Vector store

Status: accepted (default, not benchmarked)    Date: 2026-10-06
Context: CLIP frame embeddings (512-d), written according to the D8 sampling policy, are searched by text with optional camera and time filters. Hybrid queries need filtered ANN search, or an SQL prefilter followed by vector rerank, across a range of filter selectivities (ARCHITECTURE 4.6, 4.7).
Options: pgvector HNSW (same Postgres as the metadata DB); Qdrant.
Decision: pgvector in the same Postgres, if CLIP and semantic search return. Rationale: CLIP, semantic search and 4.4 were deferred in the 2026-10-06 re-scope, so no vector store runs in this plan; pgvector keeps one system and transactional joins with metadata.
Evidence: none; 4.4 deferred.
Trade-offs and consequences: No filtered-ANN comparison. Evidence that would change it: if CLIP returns, recall@10 against latency at the real vector count, especially under selective camera and time filters, falling short of the 300 ms p95 target.

## D3: Queue and event stream

Status: accepted (default, not benchmarked)    Date: 2026-10-06
Context: The frame queue decouples ingest from GPU work and absorbs bursts. It must be keyed by `camera_id` so one consumer processes a camera's frames in order, because tracking depends on that order. It delivers at least once, with acks only after a durable write, a dead-letter stream, and consumer lag exported to drive KEDA autoscaling (ARCHITECTURE 4.3). A separate question is whether detections and events should also flow through a Kafka topic that feeds the detection store (ClickHouse Kafka engine or a consumer) and the drift monitor, for decoupling and replay. The two questions are decided independently.
Options: frame queue: Redis Streams; Apache Kafka (KRaft, single broker for local runs and benchmarks). Event stream: Kafka topic; no stream, with workers writing directly to the stores.
Decision: No separate queue: the worker claims `pending` frames from Postgres with `SELECT ... FOR UPDATE SKIP LOCKED`, claiming per camera to keep order for tracking. No event stream: workers write detections and events directly. Rationale: 6.1 was deferred in the 2026-10-06 re-scope, and Redis was removed (D16) because nothing measured needed it.
Evidence: none; 6.1 deferred.
Trade-offs and consequences: Polling adds latency and DB load that a push queue avoids, and lag-based autoscaling is out of scope. Evidence that would change it: the 2.5 ramp or 2.6 profile showing claim queries or DB contention limiting rated capacity, or freshness dominated by polling delay. Then compare Redis Streams first (smaller step), Kafka only with evidence.

## D4: Ingest transport

Status: accepted (default, not benchmarked)    Date: 2026-10-06
Context: Cameras send 1280x720 JPEGs (~100 to 200 KB) at 5 FPS each. Phase 1 uses REST multipart. The public REST API is unaffected by this choice; only the loadgen-to-ingest path changes (ARCHITECTURE 4.2, 6).
Options: REST multipart; gRPC client streaming (`Ingest.StreamFrames`).
Decision: REST multipart from loadgen to ingest. Rationale: simplest to debug, already planned for Phase 1; 6.2 was deferred in the 2026-10-06 re-scope.
Evidence: none; 6.2 deferred.
Trade-offs and consequences: Per-request overhead is higher than a gRPC stream. Evidence that would change it: the 2.6 profile showing ingest CPU per frame or request overhead limiting rated capacity before the worker does.

## D5: Inference backend and precision

Status: proposed    Date: 2026-10-05
Context: YOLO11 detection and the CLIP image and text encoders run in Triton from Phase 5. Phase 1 runs them in-process with PyTorch at batch size 1. Any speedup must report its quality cost (ARCHITECTURE 4.5).
Options: PyTorch eager; ONNX Runtime CUDA; TensorRT FP16; TensorRT INT8. Also dynamic batching and instance count settings, and GPU pre- and post-processing (DALI, TensorRT NMS) if profiling shows they are CPU-bound.
Decision: pending (Phase 5, task 5.5). Since the 2026-10-06 re-scope the options are in-process backends only (PyTorch, ONNX Runtime CUDA, TensorRT FP16, TensorRT INT8, or a CPU ladder if no GPU is available) with a batch-size sweep; Triton, its dynamic batching and GPU pre/post-processing (5.4) are deferred.
Evidence: pending. To measure: img/s, latency, GPU memory, mAP and recall@k delta against PyTorch.
Trade-offs and consequences: pending. TensorRT engines must be built on the target GPU.

## D6: Service boundaries

Status: accepted (default, not benchmarked)    Date: 2026-10-06
Context: One codebase (modular monolith) with several deployables. A service boundary needs a measured reason, such as a different scaling profile (ARCHITECTURE 1, 10). Each deployable is a Dockerfile target, so splitting or merging services is a deployment change.
Options: deploy each of loadgen, ingest-gateway, inference-worker, query-service, llm-service and drift-monitor separately, or merge some of them.
Decision: One Python package and one image; loadgen, ingest, worker and query run as separate compose services, plus the UI. Rationale: separate processes keep traces and metrics per component readable; 6.6 (capacity and chaos tests) was deferred in the 2026-10-06 re-scope, so no Kubernetes boundary is decided.
Evidence: none; 6.6 deferred.
Trade-offs and consequences: No measured case for or against merging services. Evidence that would change it: CPU, GPU and memory profiles from 2.5 and 2.6 showing two services that always scale together (merge) or one that needs independent scaling (keep split).

## D7: LLM strategy

Status: accepted (default, not benchmarked)    Date: 2026-10-06
Context: The structured-query planner turns natural language into a `QueryPlan` under constrained decoding. It is self-hosted on vLLM with prefix caching and structured outputs, with a hosted LLM as the circuit-breaker fallback (ARCHITECTURE 4.7, 4.8).
Options: hosted model; small open base model with few-shot prompting; LoRA adapter on plan JSON targets; quantized LoRA (AWQ or GPTQ base, multi-LoRA).
Decision: Few-shot prompting with a static few-shot bank and JSON-schema structured outputs through an OpenAI-compatible client; no fine-tuning. The model is the better of the two compared on the dev split in 3.5 (one local model through Ollama, one hosted), taking cost into account. Rationale: Phase 8 (vLLM, LoRA) was deferred in the 2026-10-06 re-scope.
Evidence: the 3.5 dev comparison will name the model; no LoRA or serving comparison.
Trade-offs and consequences: No answer to whether a small fine-tuned model can match a hosted one. Evidence that would change it: 3.5 test accuracy too low to support the demo (say, below 60% on expressible questions), or hosted cost or latency unacceptable; then revive Phase 8 under the scope lock.

## D8: Embedding sampling

Status: accepted (default, not benchmarked)    Date: 2026-10-06
Context: Embedding every analyzed frame (5 FPS per camera) costs GPU time and vector storage. The default is 1 FPS per camera (ARCHITECTURE 2).
Options: every frame; fixed rate; scene-change triggered.
Decision: If CLIP returns: embed at a fixed 1 FPS per camera. Rationale: CLIP and 6.7 were deferred in the 2026-10-06 re-scope; 1 FPS was already the planned default.
Evidence: none; 6.7 deferred.
Trade-offs and consequences: Fixed-rate sampling misses brief events and wastes work on static scenes. Evidence that would change it: retrieval recall@k against GPU time once a retrieval eval exists.

## D9: Provider for long GPU benchmark runs

Status: accepted (default, not benchmarked)    Date: 2026-10-06
Context: AWS is the default provider for all infrastructure (S3, ECR, IAM, Budgets, EC2 spot GPU instances running k3s), in the project's single configured Region. That part is not up for decision. D9 only decides whether long GPU benchmark runs move to a cheaper GPU provider. GPU time is rented per session through scripts, with teardown verified by tag (ARCHITECTURE 9, 12).
Options: AWS EC2 spot g5 or g6 in the project Region; cheaper GPU clouds for long runs only.
Decision: AWS only, in the project's selected Region: short scripted EC2 spot GPU sessions (2.4) with tag-verified teardown and a Budgets alarm; no second GPU provider. Rationale: the 2026-10-06 re-scope removed the long GPU runs (Phase 6 cluster, 10.3) that this decision was about.
Evidence: none needed; session cost is recorded per run.
Trade-offs and consequences: A cheaper provider could lower the cost of GPU hours. Evidence that would change it: GPU spot quota still unavailable when Phase 5 starts, or planned GPU hours (summed from session results) passing about 20 hours.

## D10: Orchestrator for the retraining loop (stretch)

Status: accepted (default, not benchmarked)    Date: 2026-10-06
Context: Stretch task 10.5 chains a drift alert or manual trigger, frame sampling, a detector fine-tune (task 9.5 script), the eval gate and MLflow registry registration. It must run on the compose CPU stack with tiny models and on k3s (ARCHITECTURE 4.9).
Options: Prefect; Airflow.
Decision: No orchestrator. Rationale: the only user, the retraining loop (10.5), was deferred in the 2026-10-06 re-scope; the 9.5 fine-tune runs as a single script through `make`.
Evidence: none; 10.5 deferred.
Trade-offs and consequences: No scheduled or triggered retraining. Evidence that would change it: reviving 10.5 because drift alerts (9.3) need an automated response.

## D11: Structured query interface

Status: accepted (default, not benchmarked)    Date: 2026-10-06
Context: Structured questions go through a `StructuredQueryEngine` interface, so engines can be swapped and compared on the same eval set. The typed plan bounds the output space: validation is exhaustive, values are always bound as parameters, accuracy does not depend on the SQL dialect, and training targets are short. The cost is expressiveness: out-of-schema questions must abstain with `unsupported` (ARCHITECTURE 4.7, 11).
Options: typed `QueryPlan` compiled to parameterized SQL (default); free-form SQL with `sqlglot` validation and one self-correction attempt; agent loop with tool calling (optional prototype); hybrid (plan by default, SQL fallback for `unsupported`).
Decision: The typed `QueryPlan` compiled to parameterized SQL (D13, D14) is the only engine. Rationale: the free-form SQL engine and the comparison (7.2) were deferred in the 2026-10-06 re-scope; the typed plan bounds the output space and makes validation exhaustive.
Evidence: none for the comparison. Plan coverage of the eval set (3.5) measures what the choice costs: the share of questions that must abstain as `unsupported`.
Trade-offs and consequences: Questions outside the schema abstain. Evidence that would change it: plan coverage in 3.5 below about 80% of plausible questions, or `unsupported` being the largest error class; then build the free-form engine (7.2) as a fallback for `unsupported` only. The few-shot bank already stores gold SQL for it.

## D12: Local object store

Status: accepted    Date: 2026-10-06
Context: The compose stack needs an S3-compatible object store for frames (ingest) and MLflow artifacts, standing in for S3 so the same boto3 code runs locally and in the cloud. It must run as one small container on a laptop and in CI, create buckets without manual steps, and have a pinned, pullable image. The plan named MinIO, but MinIO no longer publishes community images (E-001).
Options: build MinIO from its last source release; SeaweedFS (`weed mini`); RustFS; Garage.
Decision: SeaweedFS 4.48 in `weed mini` mode (single process: master, volume, filer, S3 gateway). Buckets are created at startup with `-bucket`, so no bootstrap job is needed; the S3 admin identity comes from `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY`.
Evidence: E-001 (MinIO images unavailable). Integration tests pass against it: authenticated put/get/delete, anonymous access refused (403), MLflow proxied artifact upload lands in the bucket (`tests/integration/test_compose_stack.py`). Idle memory about 140 MiB (E-002).
Trade-offs and consequences: SeaweedFS is Apache-2.0 and has been developed since 2015, but its S3 layer is less widely used than MinIO's, so an S3 feature we need may behave differently (watch multipart uploads, presigned URLs and lifecycle rules). Building MinIO from source would keep a familiar server, but it adds a slow Go build to every fresh clone and CI run, for a project that no longer ships updates. RustFS reached 1.0 only days ago. Garage needs a cluster layout step before it will serve requests. Application code talks only to the S3 API through an injected client, so changing the store later means editing compose, not code. In the cloud this is S3 regardless. Revisit if an S3 incompatibility shows up in ingest benchmarks (Phase 2).

## D13: QueryPlan draft shape

Status: proposed    Date: 2026-10-06
Context: D11 makes a typed plan the default structured-query interface. Task 0.6 needs a concrete draft so the few-shot bank can be validated against it; task 1.7 finalizes it with the compiler. The ARCHITECTURE sketch left open what a count counts, how aggregates name their field, and how adversarial requests abstain.
Options: the sketch as written (intent, filters, aggregation, group_by, order, limit, visual_query, abstain_reason); the sketch plus an explicit target and constrained measure/bucket fields.
Decision: v0.1 adds `target` (tracks, detections, events or frames), a `measure` of fixed op and field, a `bucket` for trends, and a fourth abstain reason, `disallowed`, for requests to change data or reveal system data (the eval's adversarial category, task 3.1). All times are camera-local wall-clock times; the compiler resolves them in each camera's timezone with the injected clock. Field combinations are validated in the model, and all errors are reported together to feed the single repair attempt.
Evidence: 29 hand-written few-shot triples covering every intent, time-range kind and abstain reason validate against it (`eval/fewshot/structured.v1.jsonl`); 63 unit tests on the model. No accuracy evidence yet: that comes from the Phase 3 eval.
Trade-offs and consequences: an explicit target removes the most common ambiguity ("how many people" means distinct tracks, not boxes), at the cost of one more field for the model to get right. Fixed enums keep the space small and checkable but push comparisons, funnels and window-style questions to `unsupported`; the bank includes three such questions with gold SQL so plan coverage is measured honestly. Revisit after the first eval run (task 3.5) if `unsupported` covers too much of the dev set.

## D14: SQL validator policy

Status: accepted    Date: 2026-10-06
Context: Every SQL statement, compiled from a plan or written by a model (free-form engine, task 7.2), passes one validator before running on the read-only role (ARCHITECTURE 11). Text checks are easy to evade (comments, case, quoting, nested statements), so checks run on a parsed tree.
Options: regex or keyword filters; sqlglot AST checks that pass the input SQL through; sqlglot AST checks that emit SQL regenerated from the checked tree.
Decision: sqlglot AST checks, executing only the regenerated SQL. Rules: one SELECT or set operation; no DML/DDL nodes anywhere; no SELECT INTO, locking clauses or recursive CTEs; FROM sources are named tables or subqueries (no table functions, VALUES or LATERAL); tables resolved per scope against an allowlist in `configs/query.yaml`, never schema-qualified; columns resolved with `qualify` against the same allowlist; a function denylist matched by name and by prefix; at most 4 joins; outer LIMIT must be an integer literal, injected (100) or capped (1000). The allowlist leaves out `embeddings` and `query_log`.
Evidence: E-003; 133 validator unit tests, including multiple statements, comments, data-modifying CTEs, system catalogs, scoped CTE shadowing, denied functions and limit abuse.
Trade-offs and consequences: regenerating means the executed text can differ from the model's (aliases, qualified columns), which the debug panel must show. The literal-LIMIT rule means compilers inline the validated limit instead of binding it. A denylist can miss a dangerous function that a future Postgres or extension adds; the prefix rules, the read-only role and the statement timeout bound that risk. A function allowlist is the stricter alternative if the red-team set (task 3.1) finds a gap. The allowlist must be kept in step with migrations (task 1.4); an integration test runs every few-shot statement through `EXPLAIN` against it.

## D15: Video sources for virtual cameras

Status: accepted    Date: 2026-10-06
Context: Loadgen replays public clips as cameras (task 1.1). Clips must be legally fetchable by script with pinned checksums, come from static cameras, and vary enough for drift scenarios (day and night, indoor and outdoor, crowd density) and retrieval queries (people and vehicles). The default `make data` must not require signing anything, and `make data-mini` must stay small for CI.
Options: MOT17/MOT20 sequences; the VIRAT ground videos; freely licensed Wikimedia Commons videos; stock-video sites (no published checksums, terms that restrict bulk scripting).
Decision: MOT17 (4 static sequences) and MOT20 (2, one trimmed to 30 s) read out of the archives with HTTP range requests and pinned by a sha256 over the selected frames; one Commons CC BY-SA 4.0 street clip with vehicles, pinned by the sha1 Commons publishes; three VIRAT parking-lot clips, pinned by Kitware's sha512, that are fetched only with explicit license acceptance (`make data ACCEPT=virat`). All clips are normalized to at most 720 px high and 30 fps, as JPEG quality 90.
Evidence: E-004 (availability and licenses), E-005 (fetch cost: 824 MB, about 3 minutes; mini 42 s; content hash reproduced on rebuild), E-006 and E-007 (clips rejected or trimmed).
Trade-offs and consequences: MOT's NC license and VIRAT's agreement rule out commercial use and redistribution of frames; that is acceptable for a portfolio project, but screenshots must stay few and attributed. Without VIRAT, vehicles appear in only two clips (a parked car in MOT17-04 and the Commons street), which limits vehicle queries and the few-shot scenes (parking lot, loading dock) until the user opts in. Content hashes depend on the OpenCV/libjpeg build pinned in uv.lock; a mismatch warns rather than fails by default. Revisit when the Phase 3 eval needs more vehicle scenes, or if a source goes away.

## D16: Trim the stack (Redis, MLflow, W&B)

Status: accepted    Date: 2026-10-06
Context: The 2026-10-06 re-scope requires every kept technology to solve a problem the project measures. Redis was in the stack as a queue candidate (D3) and query cache; MLflow for eval runs, a model registry and the CD champion; W&B for training runs and sweeps, both behind a `Tracker` interface. The re-scope deferred the queue comparison (6.1), caching, the CD eval gate (10.2), LoRA training (Phase 8) and the fine-tune sweep (9.5 is now one fixed run).
Options: keep them idle for later; keep only the `Tracker` interface with a no-op backend; remove them.
Decision: remove Redis and MLflow from the compose stack, the `Tracker` package with its MLflow and W&B backends, the `tracking` dependency group and the Redis client (task 0.7). Results JSON files in the repo, already the source of truth, are the only record of runs; comparing two runs is a diff of two results files.
Evidence: no kept task depends on them. MLflow alone used 534 to 565 MiB of the stack's ~1.0 GiB at idle (E-002), and `make up` gets lighter for every laptop and CI run.
Trade-offs and consequences: no experiment UI, model registry or run history beyond the results files, and no cache for repeated questions. Evidence that would bring each back: Redis, if the 2.6 profile or the D3 trade-offs above show polling limits capacity, or repeated queries dominate query load; MLflow, if eval runs become frequent enough that comparing results files by hand is error-prone, or a promotion gate is built (10.2); W&B, if a training sweep (Phase 8 or a larger 9.5) is revived. The interface removed in 0.7 is in git history (task 0.3) if needed.
