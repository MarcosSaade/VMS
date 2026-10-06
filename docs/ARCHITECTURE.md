# Architecture

> Location in repo: `docs/ARCHITECTURE.md`. This file describes the target design. Items marked **(decided by benchmark)** are hypotheses until the matching decision in `docs/DECISIONS.md` is accepted. Update this file whenever the design changes.

## 1. Purpose and principles

A scalable Video Management System (VMS). It ingests many simulated camera streams, runs object detection, tracking and image embedding on GPUs, stores the results efficiently, and answers questions about the footage in three ways:

- **Structured:** natural language to a typed query plan, compiled deterministically to SQL over detections, tracks, zones and events ("how many people entered zone B between 9 and 10?").
- **Semantic:** CLIP text-to-image search over frames ("person carrying a large box").
- **Hybrid:** the same query plan with a visual query: structured filters plus semantic ranking ("red car near the loading dock last night").

The project exists to demonstrate ML systems design, so the architecture follows four principles:

1. **Baseline first, optimize with evidence.** A naive end-to-end version ships first. Every optimization is a before/after measurement on the same hardware.
2. **Split only where scaling profiles differ.** One codebase (modular monolith), several deployables. A service boundary needs a measured reason.
3. **Quality is measured alongside speed.** Any optimization reports its accuracy effect (mAP, recall@k, execution accuracy).
4. **Everything is reproducible** from scripts, pinned dependencies and versioned config.

**Non-goals:** real camera hardware, edge deployment (covered by a separate project), continuous video recording and playback (we store sampled frames), multi-tenant auth. Also out of scope because no workload here needs them: feature stores, Spark-scale batch processing, Kubeflow and multi-GPU training.

## 2. Workload model

Benchmarks are only meaningful against a defined workload. All values are config, not code.

| Parameter | Default | Notes |
|---|---|---|
| Cameras | 1 to 200+ (swept) | Virtual cameras replaying public video |
| Analysis rate | 5 FPS per camera | Detection + tracking on every analyzed frame |
| Frame | 1280x720 JPEG, quality 85 | ~100 to 200 KB |
| Embedding policy | 1 FPS per camera | **(decided by benchmark, D8)**: every frame vs fixed rate vs scene change |
| Detections per frame | ~8 average | Scene dependent |
| Retention | detections 30 days, frames 7 days | Configurable |

Example write volume at 100 cameras: 100 x 5 FPS = 500 frames/s, ~4,000 detection rows/s, ~345M rows/day.

Read mix for load tests: 60% structured, 30% semantic, 10% hybrid, at swept concurrency, **while ingest runs at rated load**.

**Rated capacity** = the largest number of cameras at 5 FPS for which the freshness SLO (section 8) holds with zero frame loss. This is the headline throughput number, reported per GPU type.

## 3. System overview

```
 loadgen: N virtual cameras, fixed FPS, drift injection
        |  gRPC client stream (JPEG + metadata)          [REST baseline in Phase 1]
        v
 ingest-gateway ------------------------------> object store (SeaweedFS / S3): frames
        |  frame refs (not images)
        v
 queue (Redis Streams | Kafka), keyed by camera_id   <- lag metric drives autoscaling
        |
        v
 inference-worker: batching, tracking, zone events --gRPC--> Triton
        |                     |                              (YOLO11, CLIP image, CLIP text)
        |  [optional, D3: detections/events topic (Kafka) -> store sink + drift-monitor]
        v                     v
 detection store          vector store
 (ClickHouse | Postgres)  (pgvector | Qdrant)
        ^                     ^
        |  compiled,          |
        |  parameterized SQL  |
 query-service (FastAPI, REST + SSE): planner -> QueryPlan -> compiler -> executor
        |        --OpenAI-compatible API, structured outputs--> llm-service (vLLM + LoRA adapters)
        ^                                                        fallback: hosted LLM
        |
 frontend (React + TS, minimal)

 drift-monitor: reads stream aggregates -> Prometheus metrics -> Alertmanager
 metadata DB (Postgres): cameras, zones, model versions, query log
 observability: OpenTelemetry -> collector -> Prometheus + Tempo -> Grafana
 experiment tracking (Tracker interface): W&B (training runs, sweeps, reports),
                                          MLflow (eval results, model registry, champion)
 artifacts: DVC (datasets, adapters, fine-tuned weights) -> local remote | S3
 cloud: AWS (S3, ECR, IAM, Budgets, EC2 spot GPU + k3s), all via Terraform
 [stretch, D10: orchestrator (Prefect | Airflow) for the retraining loop]
```

## 4. Components

### 4.1 loadgen
- Replays public video clips as N virtual cameras at a fixed FPS, with per-camera clock, sequence numbers and timestamps.
- Pre-decodes and pre-encodes frames so the generator is never the bottleneck. Its own max rate is measured and reported.
- Drift injection with a schedule: darkening (night), blur, noise, camera shift, occlusion, class-mix change (switch clip). Onset times are logged so drift detection can be scored.
- Reports sent, acknowledged, rejected and dropped frames plus send latency.

### 4.2 ingest-gateway
- Phase 1: REST multipart. Phase 6: gRPC client streaming **(decided by benchmark, D4)**. The public REST API is unaffected.
- Writes the JPEG to the object store (`frames/{camera_id}/{yyyy-mm-dd}/{frame_id}.jpg`), records frame metadata, and publishes a small message (frame ref) to the queue.
- `frame_id` is a ULID (time sortable, unique). Writes are idempotent on `frame_id`.
- Backpressure: when queue lag exceeds a threshold, reject with `RESOURCE_EXHAUSTED` (or HTTP 429). Load shedding policy is configurable and measured.

### 4.3 queue
- Decouples ingest from GPU work and absorbs bursts. Candidates: Redis Streams, Apache Kafka (KRaft mode, single broker for local runs and benchmarks) **(decided by benchmark, D3)**.
- **Keyed or partitioned by `camera_id`** so a camera's frames are processed in order by one consumer at a time. Tracking depends on this. It is also the main constraint on how workers scale.
- At-least-once delivery. Messages are acked only after results are durably written. Poison messages go to a dead-letter stream.
- Consumer lag is exported as a metric and drives worker autoscaling (KEDA).
- Event stream (part of D3): whether detections and events should also flow through a Kafka topic that feeds the detection store (ClickHouse Kafka engine or a consumer) and the drift monitor, for decoupling and replay. Decided separately from the frame queue, so Redis Streams can win the frame queue while the event stream still uses Kafka, or neither does. Kafka is adopted only with benchmark evidence; "Redis Streams suffices at this scale" is a valid outcome.

### 4.4 inference-worker
- Pulls frame refs, fetches JPEGs, decodes and preprocesses, and calls Triton with client-side batching across cameras.
- Runs multi-object tracking per camera (ByteTrack) and derives zone events (enter, exit, loiter) from per-camera zone polygons.
- Computes cheap input statistics per frame (brightness, contrast, Laplacian variance for blur) for drift monitoring.
- Writes detections, track updates, events and embeddings (per the sampling policy) in batches, directly to the stores or through the event topic if D3 adopts one.
- Phase 1 runs models in-process with PyTorch at batch size 1. Phase 5 moves inference to Triton.

### 4.5 Triton Inference Server
- Model repository under `models/triton/` (configs and export scripts are versioned; weights are fetched by script, never committed).
- Models: `yolo` (detector), `clip_image`, `clip_text`. Pre- and post-processing can move to GPU (ensemble, DALI or TensorRT NMS plugin) if profiling shows CPU-bound stages.
- Backend and precision ladder: PyTorch eager -> ONNX Runtime CUDA -> TensorRT FP16 -> TensorRT INT8 **(decided by benchmark, D5)**. Dynamic batching and instance counts are tuned with `perf_analyzer`.
- TensorRT engines are built on the target GPU by script.

### 4.6 Storage

| Store | Holds | Candidates |
|---|---|---|
| Metadata DB | cameras, zones, model registry, query log, mutable track state | Postgres |
| Detection store | detections, events (append-heavy, analytical reads) | Postgres partitioned, TimescaleDB, ClickHouse **(D1)** |
| Vector store | frame embeddings with camera/time filters | pgvector HNSW, Qdrant **(D2)** |
| Object store | JPEG frames | SeaweedFS locally (D12), S3 in cloud |

Logical schema (types simplified; all timestamps UTC, each camera has an IANA timezone for "this morning" style questions):

```
cameras(camera_id, name, site, timezone, created_at)
zones(zone_id, camera_id, name, polygon)
frames(frame_id, camera_id, ts, object_key, width, height, brightness, blur_var, model_version)
detections(ts, camera_id, frame_id, track_id, class_name, confidence, x1, y1, x2, y2, model_version)
tracks(track_id, camera_id, class_name, first_seen, last_seen, n_frames)
events(event_id, ts, camera_id, track_id, zone_id, event_type)       -- zone_enter, zone_exit, loiter
embeddings(frame_id, camera_id, ts, vector[512], model_version)
query_log(query_id, ts, mode, engine, question, query_plan, compiled_sql, latency_ms, tokens, success, error_type, feedback)
```

Notes that shape D1: column stores excel at time-range aggregates and compression but are poor at updates, so mutable track state may stay in Postgres. Detection and event tables are append-only. Because the LLM emits a `QueryPlan`, not SQL, model accuracy does not depend on the SQL dialect. What D1 weighs instead is compiler complexity and plan-to-SQL result equivalence across backends: each viable backend needs a compiler, with equivalence tests showing the same plan returns the same result set on every backend.

### 4.7 query-service
FastAPI, async. Public endpoints in section 6.

Structured queries go through a `StructuredQueryEngine` interface (question plus context in, structured result out), so engines can be swapped and compared on the same eval set **(decided by eval, D11)**. The typed-plan engine is the default until D11 is accepted.

**Structured pipeline (default: typed query plan):**
1. Build context: the `QueryPlan` JSON schema, a pruned schema summary (tables, columns, known class names, zone names and event types), camera timezones, rules, and few-shot (question, plan) examples retrieved by similarity from an example bank (never from the test set).
2. The LLM planner emits a `QueryPlan` under constrained decoding: vLLM structured outputs when self-hosted; provider structured outputs or tool calling for the hosted fallback; JSON-schema structured outputs through the OpenAI-compatible client locally (e.g., Ollama).
3. Validate the plan: Pydantic schema plus semantic checks (known cameras, zones and classes, sane time range, limit within cap). On failure, one repair attempt that feeds the validation error back to the planner.
4. If the intent is `abstain`, return the abstain reason without executing.
5. Compile the plan deterministically to parameterized SQL for the active storage backend (one compiler per backend).
6. Defense in depth: the `sqlglot` validator checks the compiled SQL (parse to AST, single `SELECT`, table and column allowlist, function denylist, `LIMIT` injected or capped).
7. Execute on a read-only DB role with a statement timeout and row cap.
8. Stream a short natural-language answer, structured results, the plan JSON and the compiled SQL over SSE.

`QueryPlan` (Pydantic model, JSON schema exported for constrained decoding):

```
schema_version
intent:         count | aggregate | list | trend | top_k | dwell | frames | abstain
filters:        cameras, zones, classes, min_confidence, event_types,
                time_range (absolute, or relative resolved in the camera timezone)
aggregation, group_by, order, limit
visual_query:   optional text for CLIP ranking (set => hybrid)
abstain_reason: ambiguous | unanswerable | unsupported   (when intent = abstain)
```

Why a typed plan by default: the output space is bounded, so validation is exhaustive and values are always bound as parameters; model accuracy is independent of the SQL dialect; training targets are short and checkable. The cost is expressiveness: questions outside the schema must abstain with `unsupported`, and plan coverage of the eval set is measured and reported.

**Alternative engines (behind the same interface):**
- Free-form SQL engine (task 7.2): pruned schema DDL prompt, the LLM writes SQL, `sqlglot` validation, one self-correction attempt on execution error.
- Agent loop (optional prototype for D11): tool calling over the plan executor and CLIP search.

**Semantic pipeline:** CLIP text embedding via Triton -> ANN top-k with optional camera and time filters -> frames with presigned URLs.

**Hybrid pipeline:** a `QueryPlan` with `visual_query` set, so one planner serves structured and hybrid queries. The executor runs either filtered ANN search or a compiled SQL prefilter followed by vector rerank, chosen by estimated filter selectivity (thresholds measured).

Cross-cutting: Redis cache keyed by normalized question plus data watermark; circuit breaker with fallback to a hosted LLM; every query logged to `query_log` with engine, plan JSON, compiled SQL and stage timings.

### 4.8 llm-service
- vLLM with an OpenAI-compatible API, prefix caching on (the schema prompt is long and shared) and structured outputs for `QueryPlan` JSON; the constrained-decoding overhead is measured.
- Small open model chosen by eval; a LoRA adapter whose target is `QueryPlan` JSON (hybrid uses the same plan, so one adapter covers both). Optionally a second adapter with SQL targets for the D11 comparison. If both exist they are served on one quantized base (AWQ or GPTQ) with multi-LoRA; otherwise adapter vs merged overhead is measured and reported.
- Choice of prompting vs LoRA vs hosted **(decided by benchmark, D7)** on structured-query execution accuracy, p95 latency and cost per 1,000 queries.

### 4.9 drift-monitor
- Per camera, compares a rolling window against a reference window captured during a calibration period.
- Signals: class-mix PSI, confidence distribution (KS or Wasserstein), detections per frame, input statistics (brightness, blur), embedding drift (MMD on sampled embeddings or centroid shift).
- Exports gauges to Prometheus; alert rules in Alertmanager. Thresholds are calibrated on clean runs to a target false-alarm rate.
- Validated with loadgen drift injection: detection delay and false-alarm rate per drift type, plus the measured mAP drop under each drift to justify why it matters.
- Response: the detector is fine-tuned on a clean plus photometrically drifted mix (task 9.5) and must pass the eval gate before becoming champion. A stretch orchestrated retraining loop (task 10.5, D10) chains drift alert -> frame sampling -> fine-tune -> eval gate -> registry.

### 4.10 Evaluation and experiment tracking
- `eval/` holds versioned datasets and the harness:
  - Structured suite (`make eval SUITE=structured`): structured-query execution accuracy. Gold entries store the expected result set, per-dialect gold SQL where needed, a gold `QueryPlan` where expressible and an `expressible_by_plan` flag. Category `unsupported` covers valid analytic questions outside the plan schema (the plan engine must abstain with reason `unsupported`). Reports accuracy over all questions and over expressible ones, plus plan coverage, with plan error classes (invalid plan, plan failed validation, wrong filter, wrong aggregation, wrong group_by) alongside the SQL error classes.
  - Retrieval recall@k and nDCG; detection mAP on a fixed labeled subset (clean and drifted).
- Dev and test splits; the test split is frozen and only used for reported numbers.
- Experiment tracking goes through a `Tracker` protocol in `vms.common` with MLflow, W&B and no-op implementations selected by config. W&B records training runs, sweeps and reports (LoRA, detector fine-tune). MLflow (self-hosted in compose) records eval results and holds the model registry and champion tracking. W&B defaults to disabled or offline and is never needed for the demo. Result JSON files in the repo remain the source of truth for reported numbers and record tracker run IDs or URLs.
- DVC tracks large artifacts we produce (synthetic datasets, training data, adapters, fine-tuned weights, large eval snapshots); small eval JSONL and configs stay in git. See section 10.
- An eval gate in CD blocks a model or prompt change that regresses against the recorded champion.

### 4.11 Observability
- OpenTelemetry in every service. A frame's trace spans loadgen -> ingest -> queue -> worker -> Triton -> DB write. A query's trace spans every pipeline stage.
- Metrics naming: `vms_<component>_<metric>_<unit>`. Key metrics: ingest frames/s, rejected and dropped frames, queue lag, end-to-end freshness (frame timestamp to queryable), worker batch size and latency, Triton queue and compute time, GPU utilization and memory (DCGM exporter), query latency by mode and stage, LLM tokens and cost.
- Structured JSON logs with trace IDs. Dashboards as code in `deploy/grafana/`.

### 4.12 frontend
Minimal React + TypeScript (Vite): query box with mode selector, result tables and charts, frame grid, debug panel showing the plan JSON and compiled SQL, thumbs up/down feedback to `query_log`.

### 4.13 Shared library (`vms.common`)
Every service builds on the same small modules, each behind an interface so tests inject fakes:
- **Settings:** one pydantic-settings class per service. Values are layered from code defaults, then `configs/common.yaml`, `configs/<service>.yaml` and `configs/<service>.<env>.yaml` (deep-merged, `env` from `VMS_ENV`), then `.env`, then `VMS_*` environment variables (`__` for nesting), then explicit arguments. Validation is strict, so a typo in YAML fails at startup.
- **Observability:** `init_observability(settings)` configures JSON logs that carry the active trace and span IDs, OpenTelemetry tracing (OTLP to the collector; a no-op tracer when disabled) and a Prometheus registry. Metric names are assembled and validated as `vms_<component>_<metric>_<unit>`.
- **IDs and time:** a monotonic ULID generator and a `Clock` protocol (system and fake implementations). Both take an injected clock and randomness source, so tests are deterministic.
- **Tracking:** a `Tracker` protocol with no-op, MLflow and W&B backends. `build_tracker` reads the backends from config and returns a `RunTracker` that sends each call to every enabled backend, returning one run reference per backend for the results JSON. Backend libraries are imported only when enabled. Artifacts are logged by reference, never uploaded.

## 5. Data flows and guarantees

**Write path:** loadgen -> ingest (store JPEG, insert frame row, publish ref) -> queue -> worker (batch, infer, track, write) -> ack.
- Delivery: at-least-once; idempotent writes keyed on `frame_id` (and `(frame_id, detection_idx)`), so redelivery causes no duplicates.
- Ordering: per camera, via queue keying.
- Overload: backpressure at ingest, measured shedding above rated capacity, never silent loss.
- Shutdown: workers drain in-flight batches before exit.

**Read paths:** query-service -> (LLM planner -> plan validation -> compiler -> SQL validator -> detection store) | (Triton CLIP text -> vector store) | hybrid (plan with visual query -> filtered ANN or prefilter + rerank) -> object store presigned URLs for frames.

## 6. Interfaces

Internal (gRPC, in `proto/`):

```proto
service Ingest {
  rpc StreamFrames(stream Frame) returns (IngestSummary);
}
message Frame {
  string camera_id = 1;
  int64  ts_unix_ms = 2;
  uint64 seq = 3;
  bytes  jpeg = 4;
}
message IngestSummary { uint64 accepted = 1; uint64 rejected = 2; }
```

Triton is called over its native gRPC API.

Public (REST):

| Endpoint | Purpose |
|---|---|
| `POST /v1/query` | Structured natural-language query, SSE stream |
| `POST /v1/search/semantic` | CLIP search with optional filters |
| `POST /v1/search/hybrid` | Hybrid search |
| `GET /v1/frames/{frame_id}` | Frame metadata and presigned URL |
| `POST /v1/feedback` | Thumbs up/down on a query |
| `GET /healthz`, `GET /readyz`, `GET /metrics` | Health and metrics on every service |

## 7. Decisions to be made by benchmark

Results and rationale go in `docs/DECISIONS.md` (seeded from this table in task 0.2, all `proposed`), raw runs in `docs/EXPERIMENTS.md`.

| ID | Question | Candidates | Decided by | Phase |
|---|---|---|---|---|
| D1 | Detection store | Postgres partitioned + BRIN, TimescaleDB, ClickHouse | Ingest rows/s, query p95 by class, bytes/row at 10M/100M (1B optional), compiler complexity and plan-to-SQL result equivalence across backends | 4 |
| D2 | Vector store | pgvector HNSW, Qdrant | Recall@10 vs latency and QPS at 1M/10M, filtered search across selectivities | 4 |
| D3 | Queue and event stream | Redis Streams, Apache Kafka (KRaft, single broker); separately, whether detections and events flow through a Kafka topic | Throughput, per-key ordering, replay, rebalancing, operational cost; for the event stream, decoupling and replay value vs added ops | 6 |
| D4 | Ingest transport | REST multipart, gRPC streaming | Frames/s per core, p99, CPU per frame | 6 |
| D5 | Inference backend and precision | PyTorch, ORT CUDA, TRT FP16, TRT INT8; batching config | img/s, latency, GPU memory, mAP and recall delta | 5 |
| D6 | Service boundaries | Which components deploy separately | Measured resource and scaling profiles | 6 |
| D7 | LLM strategy | Hosted, base model few-shot, LoRA (plan targets), LoRA quantized | Structured-query execution accuracy, p95, cost per 1k queries | 8 |
| D8 | Embedding sampling | Every frame, fixed rate, scene change | Retrieval recall vs GPU time and storage | 6 |
| D9 | Provider for long GPU benchmark runs (AWS is the default for all infra) | AWS (g5/g6 spot), cheaper GPU clouds | Cost and availability | 2 |
| D10 | Orchestrator for the retraining loop (stretch) | Prefect, Airflow | Operational weight on compose and k3s, DAG expressiveness, effort | 10 |
| D11 | Structured query interface | Typed plan (default), free-form SQL, agent loop (optional prototype) | Execution accuracy, abstention correctness, p95, tokens and cost, plan coverage, red-team attack success rate | 7 |

## 8. SLOs (targets, validated in benchmarks)

| SLO | Target |
|---|---|
| Freshness: frame timestamp to detections queryable | p95 < 2 s at rated capacity |
| Frame loss at or below rated capacity | 0 |
| Structured query latency (incl. LLM) | p95 < 1.5 s |
| Semantic search latency | p95 < 300 ms |
| Hybrid query latency | p95 < 2 s |
| Structured-query execution accuracy | Baseline measured in Phase 3; must improve and never regress past the gate |

Targets may be revised with evidence; record any change in `docs/DECISIONS.md`.

## 9. Deployment

- **Local:** `docker compose` CPU profile with small models; GPU profile when a GPU is present. The stack (`deploy/compose/docker-compose.yml`) runs the infrastructure every service needs, each with a health check: Postgres 17 with pgvector (application DB and MLflow backend DB), SeaweedFS as the S3-compatible object store (D12; buckets for frames and MLflow artifacts are created at startup), Redis (AOF on), the OpenTelemetry collector (OTLP in; traces to Tempo, OTLP metrics re-exposed for Prometheus), Prometheus, Tempo, Grafana with provisioned Prometheus and Tempo datasources (fixed UIDs `prometheus` and `tempo`, files in `deploy/grafana/provisioning/`) and MLflow (Postgres backend, artifacts proxied by the server into the object store, so clients need no S3 credentials). Infrastructure services have no profile and always start. Application services that come in CPU and GPU variants carry `profiles: [cpu]` or `profiles: [gpu]`; `make up` activates `cpu`, `make up PROFILES=gpu` activates `gpu`. Kafka is not in the stack; it is added only if D3 benchmarks justify it (task 6.1). W&B is hosted and not in compose.
- **Kubernetes:** one Helm chart with per-service values. `kind` in CI for smoke tests. GPU benchmarks run on an EC2 spot GPU instance with single-node k3s. EKS only if a decision record justifies it and the user approves the cost. KEDA scales workers on queue lag; HPA scales query-service on CPU or RPS. Triton gets GPU resource requests.
- **Cloud (AWS by default):** Terraform under `deploy/terraform/` provisions, in the project's single configured Region: S3 (frames, DVC remote, artifacts), ECR, IAM with least-privilege roles (GPU instance role limited to the project bucket and ECR pull; CI role limited to ECR push), AWS Budgets alerts, and EC2 spot GPU instances (g5 or g6) running k3s. Every resource is tagged `project=vms` and `session=<id>`. Every apply has a scripted destroy; after destroy, `scripts/gpu_session.sh` lists remaining tagged resources through the Resource Groups Tagging API and fails loudly if any remain. D9 decides only whether long GPU benchmark runs move to a cheaper provider.
- **CI/CD (GitHub Actions):** lint, typecheck, unit tests, integration tests on the compose CPU stack, image builds, Helm lint and kind smoke test, eval gate on a small dev subset.

## 10. Repository layout

```
.
├── AGENTS.md
├── README.md
├── Makefile
├── pyproject.toml, uv.lock
├── .dvc/, *.dvc          # DVC config and pointers for large artifacts
├── configs/              # YAML config per service and environment
├── proto/                # gRPC definitions
├── src/vms/
│   ├── common/           # config, logging, telemetry, ids, clock, Tracker, shared schemas
│   ├── loadgen/
│   ├── ingest/
│   ├── worker/           # inference client, tracking, zones, input stats
│   ├── query/            # QueryPlan, planner, per-backend compilers, engines, sql validator, semantic, hybrid
│   ├── drift/
│   ├── storage/          # clients, repositories, migrations
│   └── bench/            # benchmark runner and result schema
├── models/               # export scripts, Triton configs (no weights)
├── eval/                 # datasets (small, versioned), harness, results/
├── training/             # LoRA data generation and training, detector fine-tune
├── bench/                # scenarios/*.yaml, results/*.json
├── scripts/              # data download, provisioning helpers
├── deploy/
│   ├── compose/
│   ├── helm/
│   ├── terraform/
│   └── grafana/
├── frontend/
├── tests/                # unit/, integration/, load/
└── docs/
```

One Python package with several entrypoints; each deployable is a Dockerfile target, so splitting or merging services is a deployment change, not a rewrite.

Data and artifacts: small files (eval JSONL, configs, few-shot banks) stay in git. Large artifacts we produce (synthetic datasets, training data, LoRA adapters, fine-tuned weights, large eval snapshots) are tracked with DVC: a local remote by default (`make data-pull`), an S3 remote configured by Terraform in the cloud. Public third-party datasets and pretrained weights are still fetched by script with checksums and licenses. The CPU demo needs no DVC remote access; it uses `make data-mini`.

## 11. Security

- The query-service DB role is read-only, with statement timeouts and row limits.
- User questions are untrusted input. Plan engine: LLM output must pass `QueryPlan` schema and semantic validation, and the compiler emits parameterized SQL, so user values never become SQL text. Both the plan and free-form SQL engines run the `sqlglot` validator on the final SQL as defense in depth, and all SQL runs on the read-only role.
- Rate limiting on the public API.
- Secrets live only in `.env` (gitignored) or a secret manager; `.env.example` documents every variable.
- A red-team question set (DDL, injection, exfiltration, resource exhaustion) is part of the eval suite, with attack success rate reported.

## 12. Cost model

- GPUs are rented per session through scripts: provision -> run -> collect -> destroy, with teardown on failure. AWS EC2 spot (g5 or g6) is the default; D9 may move long benchmark runs to a cheaper provider.
- AWS Budgets alerts are mandatory; every resource carries `project=vms` and `session=<id>` tags, which also drive per-session cost attribution and the post-teardown leftover check.
- Every cloud benchmark and training run records its cost.
- Headline efficiency metric: **cost per camera-hour** = instance $/hour / rated capacity.
