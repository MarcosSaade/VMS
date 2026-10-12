# Architecture

> Location in repo: `docs/ARCHITECTURE.md`. This file describes the target design. Items marked **(decided by benchmark)** are hypotheses until the matching decision in `docs/DECISIONS.md` is accepted. Update this file whenever the design changes.

**Current scope (2026-10-06 re-scope, `PLAN.md`).** The plan builds a subset of this design: loadgen, REST ingest, the object store, a worker with YOLO11 and tracking that claims frames from Postgres (`SKIP LOCKED`, D3 default), the structured query path (typed plan, compiler, validator), a minimal UI, observability and dashboards, the evaluation harness, a precision ladder with the worker on the best backend in-process (no Triton), drift detection, and short scripted GPU sessions on AWS. Deferred parts of the design are marked with a scope note in their sections: the queue comparison and event stream, gRPC ingest, Triton, CLIP with semantic and hybrid search, vLLM and LoRA, storage benchmarks, Kubernetes and autoscaling, Terraform and CD. Decisions about deferred parts are accepted defaults (`DECISIONS.md`), not benchmark results.

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

The diagram shows the target design; the current scope (top of this file) builds a subset of it.

```
 loadgen: N virtual cameras, fixed FPS, drift injection
        |  gRPC client stream (JPEG + metadata)          [REST baseline in Phase 1]
        v
 ingest-gateway ------------------------------> object store (SeaweedFS / S3): frames
        |  frame refs (not images)
        v
 queue keyed by camera_id (in scope: Postgres SKIP LOCKED; deferred: Redis Streams | Kafka)
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
 eval and benchmark results: results JSON files in the repo (no experiment tracker, D16)
 artifacts: DVC (datasets, adapters, fine-tuned weights) -> local remote | S3
 cloud: AWS (S3, ECR, IAM, Budgets, EC2 spot GPU + k3s), all via Terraform
 [stretch, D10: orchestrator (Prefect | Airflow) for the retraining loop]
```

## 4. Components

### 4.1 loadgen
- Replays public video clips as N virtual cameras at a fixed FPS, with per-camera clock, sequence numbers and timestamps. Clips come from `make data` (`vms.datasets`, D15): normalized JPEG frames at most 720 px high and 30 fps under `data/clips/<id>/`, each with a `clip.json` (fps, size, scene labels, license, content hash). Sources, checksums and licenses are pinned in `configs/data/sources.yaml`; every benchmark uses the same clips.
- Pre-decodes and pre-encodes frames so the generator is never the bottleneck. Its own max rate is measured and reported.
- Drift injection with a schedule: darkening (night), blur, noise, camera shift, occlusion, class-mix change (switch clip). Onset times are logged so drift detection can be scored.
- Reports sent, acknowledged, rejected and dropped frames plus send latency.
- **v0 (task 1.2, `vms.loadgen`, D17):** async, one event loop. Frames of the assigned clips are resized to 1280x720 and re-encoded at quality 85 once at startup and shared by all cameras playing a clip. Each camera has a seeded start frame and a seeded phase in [0, 1/fps), so sends spread over the frame interval, and it steps through its clip in real time (clip_fps/fps frames per send, looping). Pacing is open loop: frame k is due at `phase + k/fps` whatever the server does, and a frame due while the camera has `max_in_flight_per_camera` sends outstanding is dropped and counted. Schedule lag (send start minus due time) is the generator's health signal; a run whose lag p99 exceeds `max_schedule_lag_p99_ms` is flagged `generator_ok: false`. `unpaced` mode (closed loop, as fast as the in-flight limit allows) and `selftest` (unpaced, `NullTransport`, no I/O) measure the generator's maximum rate. Transport is a multipart POST per frame (`vms.common.ingest_api`) over pooled keep-alive connections, with optional bounded retries (full jitter; never on 429/503). Metrics `vms_loadgen_{sent,accepted,rejected,dropped}_frames_total`, `vms_loadgen_failed_frames_total{reason}`, `vms_loadgen_send_duration_seconds`, `vms_loadgen_schedule_lag_seconds`, `vms_loadgen_in_flight_requests` on `:9105/metrics`; 1% of frames start a trace (`traceparent` header). Each run writes a results JSON (`vms.bench.results`) with per-repetition steady-state numbers and an aggregate. Drift injection is not built yet (task 9.2). A stub ingest (`python -m vms.loadgen stub`) implements the same contract without storage.

### 4.2 ingest-gateway
- Phase 1: REST multipart. Phase 6: gRPC client streaming **(decided by benchmark, D4)**. The public REST API is unaffected.
- Writes the JPEG to the object store (`frames/{camera_id}/{yyyy-mm-dd}/{frame_id}.jpg`), records frame metadata, and publishes a small message (frame ref) to the queue.
- `frame_id` is a ULID (time sortable, unique). Writes are idempotent on `frame_id`.
- Backpressure: when queue lag exceeds a threshold, reject with `RESOURCE_EXHAUSTED` (or HTTP 429). Load shedding policy is configurable and measured.
- **v0 (task 1.3, `vms.ingest`, D18):** FastAPI on uvicorn, one process, run on the host (`make ingest`). `POST /v1/frames` (contract in `vms.common.ingest_api`) validates the form (safe-character `camera_id`/`run_id`, JPEG magic bytes, size cap enforced from a required Content-Length before parsing, capture time at most 5 min in the future), then writes the object at `frames/{camera_id}/{yyyy-mm-dd}/{frame_id}.jpg` (capture date, UTC) and only then inserts the `frames` row with status `pending`, `ON CONFLICT (camera_id, run_id, seq) DO NOTHING`. 201 for a new frame; 200 with the original `frame_id` and `duplicate: true` for a repeat, whose freshly written object is deleted; 500 when the store or database fails, after deleting the object (a failed delete is counted in `vms_ingest_orphaned_objects_total`). A row never points at a missing object. Postgres through a psycopg async pool (statement timeout 2 s, acquire timeout 2 s); the object store through boto3 in a 16-thread pool (connect 2 s, read 5 s, 3 attempts with jittered backoff). Metrics `vms_ingest_frames_total{outcome}`, `vms_ingest_stage_duration_seconds{stage=object_put|db_insert|cleanup}`, `vms_ingest_capture_to_receive_seconds`, `vms_ingest_request_duration_seconds{route,status}`, `vms_ingest_received_jpeg_bytes_total`; spans continue loadgen's `traceparent`. `/readyz` checks Postgres and the bucket. No backpressure yet (deferred with 6.3). Runs on the host (`make ingest`) or as a container of the application image (`make demo`).

### 4.3 queue
*Scope note: deferred. In this plan the `frames` table is the queue (D3 default); no Redis, no Kafka. Workers lease cameras in `camera_leases` (TTL, renewed by the holder) and claim pending frames of their leased cameras with `FOR UPDATE SKIP LOCKED`, oldest first. Completion is fenced on `claimed_by`; a dead worker's cameras and in-flight frames are taken over when its lease expires; frames fail after `max_attempts` (D20). The backlog is exported as `vms_worker_backlog_frames`.*
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
- **v0 as built (task 1.5, D20):** `vms.worker` leases up to `max_cameras` cameras, claims `claim_batch` frames per query, and per frame: fetches the JPEG, then decodes, computes brightness, contrast and Laplacian variance, and runs Ultralytics YOLO11 (`yolo11s` by default, weights pinned by sha256 in `configs/models.yaml`) in one inference thread; then writes detections and marks the frame done in one transaction fenced on its claim. `model_version` is `<weights>@<sha256 prefix>`. Undecodable frames fail at once; object-store, model and database errors retry up to `max_attempts`. Late frames (captured before one already processed for the camera) are stored and counted. Probes on `/healthz`, `/readyz`, `/metrics` (port 8090); SIGTERM finishes the current frame and hands back leases and claimed frames. In the demo it runs as a container of the application image (yolo11n, weights mounted read-only). Model-only CPU latency is benchmarked by `make bench-model`.
- **Tracking and zones as built (task 1.6, D21):** per leased camera the worker keeps a ByteTrack tracker (Ultralytics implementation) and a zone state machine. Detector threshold 0.1 feeds ByteTrack's low-confidence round; detections below 0.25 are stored only when they belong to a track. Only `tracker.classes` (people and vehicles) are tracked. Track ids are `<camera_id>:<session>:<n>`, with a new session per tracker, so ids never repeat across workers, restarts or pauses; a pause longer than `tracker.buffer_s` starts a new session. A track's anchor is its box's bottom centre; zone enter and exit need `zones.min_frames` consecutive frames on the new side and are stamped at the first of them; loiter fires once per visit after `zones.loiter_s`; a track unseen for `buffer_s` while inside exits at its last sighting. Frames that are late or retried are stored without tracking. Track sightings (upsert: `last_seen`, `n_frames`) and events are written in the frame's fenced transaction. The per-frame tracking step (`vms.worker.pipeline.track_frame`) is shared with `python -m vms.worker render`, which draws it on clips for visual review.

### 4.5 Triton Inference Server
*Scope note: deferred. The precision ladder (PyTorch, ONNX Runtime, TensorRT FP16 and INT8) is measured in-process for YOLO11 only, and the worker runs the chosen backend in-process (D5).*
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
zones(zone_id, camera_id, name, polygon)       -- zone_id '<camera_id>:<name>'; polygon [[x, y], ...] normalized 0..1
frames(frame_id, camera_id, run_id, seq, ts, received_at, object_key, size_bytes, status, width, height, brightness, contrast, blur_var, model_version, claimed_by, claimed_at, attempts, processed_at, error)   -- UNIQUE (camera_id, run_id, seq); status pending|processing|done|failed
camera_leases(camera_id, worker_id, acquired_at, expires_at)   -- which worker processes a camera (D20)
detections(frame_id, det_idx, ts, camera_id, track_id, class_name, confidence, x1, y1, x2, y2, model_version)   -- PK (frame_id, det_idx); box in pixels
tracks(track_id, camera_id, class_name, first_seen, last_seen, n_frames)
events(event_id, ts, camera_id, track_id, zone_id, event_type)       -- zone_enter, zone_exit, loiter
embeddings(frame_id, camera_id, ts, vector[512], model_version)   -- deferred with CLIP; not migrated
query_log(query_id, ts, mode, engine, question, query_plan, compiled_sql, latency_ms, tokens, success, error_type, feedback, timings, row_count, llm)
```

Schema v1 is migrations 0001 to 0003 (0004 adds the worker's claim columns, a partial index on pending frames and `camera_leases`; 0005 adds per-stage `timings`, `row_count` and `llm` to `query_log`) in `src/vms/storage/migrations/versions/` (D19). Cameras and zones come from `configs/cameras.yaml` (synced by `python -m vms.storage bootstrap`, run by `make migrate`); zone polygons are drawn on frames of each camera's demo clip with the zone editor (`make zones`, `vms.zones`, D24), which rewrites that file through the same validation. Pipeline-written tables (frames, detections, tracks, events) carry no foreign keys; only zones reference cameras. Each time-series table has one `(camera_id, ts)` index (tracks: `(camera_id, last_seen)`), the deliberate baseline. The query service reads through the `vms_query_reader` role, which holds SELECT on exactly the validator's allowlisted columns (section 11).

Notes that shape D1: column stores excel at time-range aggregates and compression but are poor at updates, so mutable track state may stay in Postgres. Detection and event tables are append-only. Because the LLM emits a `QueryPlan`, not SQL, model accuracy does not depend on the SQL dialect. What D1 weighs instead is compiler complexity and plan-to-SQL result equivalence across backends: each viable backend needs a compiler, with equivalence tests showing the same plan returns the same result set on every backend.

### 4.7 query-service
FastAPI, async. Public endpoints in section 6.

Structured queries go through a `StructuredQueryEngine` interface (question plus context in, structured result out), so engines can be swapped and compared on the same eval set **(decided by eval, D11)**. The typed-plan engine is the default until D11 is accepted.

**Structured pipeline (default: typed query plan):**
1. Build context: the `QueryPlan` JSON schema, a pruned schema summary (tables, columns, known class names, zone names and event types), camera timezones, rules, and few-shot (question, plan) examples retrieved by similarity from an example bank (never from the test set). The bank is versioned JSONL of (question, gold plan, gold SQL) triples (`eval/fewshot/structured.v<N>.jsonl`), and prompts are versioned templates (`configs/prompts/*.v<N>.txt`), both referenced from `configs/query.yaml`.
2. The LLM planner emits a `QueryPlan` under constrained decoding: vLLM structured outputs when self-hosted; provider structured outputs or tool calling for the hosted fallback; JSON-schema structured outputs through the OpenAI-compatible client locally (e.g., Ollama).
3. Validate the plan: Pydantic schema plus semantic checks (known cameras, zones and classes, sane time range, limit within cap). On failure, one repair attempt that feeds the validation error back to the planner.
4. If the intent is `abstain`, return the abstain reason without executing.
5. Compile the plan deterministically to parameterized SQL for the active storage backend (one compiler per backend).
6. Defense in depth: the `sqlglot` validator (`vms.query.sql_validator`, D14) checks the compiled SQL on its AST: one `SELECT` or set operation; no DML or DDL anywhere in the tree (including data-modifying CTEs), no `SELECT INTO`, locking clauses or recursive CTEs; FROM sources are allowlisted tables or subqueries, resolved per scope; every column resolves to the allowlist in `configs/query.yaml`; function denylist by name and prefix; integer-literal `LIMIT`, injected or capped. It returns SQL regenerated from the checked tree (comments dropped, `*` expanded to allowlisted columns), and only that SQL is executed.
7. Execute on a read-only DB role with a statement timeout and row cap.
8. Stream a short natural-language answer, structured results, the plan JSON and the compiled SQL over SSE.

`QueryPlan` (`vms.query.plan`, Pydantic model, JSON schema exported for constrained decoding; draft v0.1 from task 0.6, finalized in 1.7; D13):

```
schema_version: "0.1"
intent:         count | aggregate | list | trend | top_k | dwell | frames | abstain
target:         tracks | detections | events | frames      (what is counted or listed)
filters:        cameras, zones, classes (names), event_types, min_confidence,
                time_range: absolute (camera-local wall clock) | last N units | named period
measure:        {op: avg | min | max | sum, field: confidence | n_frames | dwell_seconds}
group_by:       up to 2 of camera | zone | class_name | event_type
bucket:         hour | day | week                           (trend only)
order:          {by: count | value | time, direction}
limit
visual_query:   optional text for CLIP ranking (frames only; set => hybrid)
abstain_reason: ambiguous | unanswerable | unsupported | disallowed   (when intent = abstain)
```

The model validates field combinations (for example: `trend` needs `bucket`, `dwell` needs tracks and `dwell_seconds`, `event_types` needs target events) and reports every error at once, so one repair attempt can fix them all. Each plan maps to a result kind (scalar, table, time series, frames, abstention) that drives the answer template and the frontend view. Times in a plan are always camera-local; the compiler resolves them with the injected clock.

Why a typed plan by default: the output space is bounded, so validation is exhaustive and values are always bound as parameters; model accuracy is independent of the SQL dialect; training targets are short and checkable. The cost is expressiveness: questions outside the schema must abstain with `unsupported`, and plan coverage of the eval set is measured and reported.

**Alternative engines (behind the same interface):**
- Free-form SQL engine (task 7.2): pruned schema DDL prompt, the LLM writes SQL, `sqlglot` validation, one self-correction attempt on execution error.
- Agent loop (optional prototype for D11): tool calling over the plan executor and CLIP search.

*Scope note: semantic and hybrid search are deferred with CLIP; `visual_query` is rejected as unsupported.*

**Semantic pipeline:** CLIP text embedding via Triton -> ANN top-k with optional camera and time filters -> frames with presigned URLs.

**Hybrid pipeline:** a `QueryPlan` with `visual_query` set, so one planner serves structured and hybrid queries. The executor runs either filtered ANN search or a compiled SQL prefilter followed by vector rerank, chosen by estimated filter selectivity (thresholds measured).

Cross-cutting (deferred: cache and circuit breaker): a cache keyed by normalized question plus data watermark; circuit breaker with fallback to a hosted LLM. In scope: every query logged to `query_log` with engine, plan JSON, compiled SQL and stage timings.

**v0 as built (task 1.7, D22):** `python -m vms.query serve` (`make query`, port 8070), structured mode only, one JSON response per question (no SSE, no cache). Modules: `catalog` (cameras and zones read from Postgres through the read-only login, cached for `catalog_ttl_s`, plus the 80 COCO class names from `configs/query.yaml`), `llm` (an `LlmClient` interface: an OpenAI-compatible client over aiohttp with `response_format` `json_schema`, `json_object` or `none`, timeout and bounded full-jitter retries; and `CannedLlm`, which answers stored plans and abstains on anything else, used when `LLM_MODEL` is empty), `planner` (renders `planner.v1.txt` with the catalog, the current local time per camera timezone from the injected clock, the plan schema and all few-shot examples; parses, validates structurally and semantically, repairs once with every error), `semantic` (names exist, zones on the plan's cameras, time range at most `max_time_range_days` and not in the future, limit within the cap), `timerange` and `compiler` (below), the existing `sql_validator`, `store` (two pools: the read-only executor and the query-log writer), `answer` (templated answers, no second LLM call), `engine` (`StructuredQueryEngine` and `PlanEngine`, which times each stage and logs every question) and `app` (FastAPI). Stages and error types: catalog, plan (`planner_unavailable` 503, `invalid_plan` 422), abstention (200, `status: abstained`; a plan with `visual_query` abstains as `unsupported` until CLIP), compile (`compile_error` 500), validate (`sql_rejected` 500: the compiler emitted something the validator refuses, blocked before execution), execute (`timeout` 504, `execution_error` 500). The response carries the answer, columns and rows, result kind, the compact plan, the validated SQL with `%(pN)s` placeholders and its parameters, stage timings, tokens and the planner's model. Frames results add `image_url` (`GET /v1/frames/{id}/image`, served by the service from the object store).

**Compiler semantics (`vms.query.compiler`, Postgres).** SQL text comes only from constants in the compiler; every plan value is a parameter, and LIMIT is inlined as a validated integer (D14). Every query is restricted to catalog cameras (the plan's cameras, else its zones' cameras, else all). Time ranges resolve per camera timezone (`vms.query.timerange`: named periods, last N units and local absolute times, DST-correct, from the injected clock), with one OR branch of UTC bounds per timezone group. Detections, events and frames match on `ts`; a track matches if visible during the range (`last_seen >= start AND first_seen < end`). Tracks filtered or grouped by zone are the tracks that entered the zone in the range, counted once per zone. Events by class use their track's class; frames by class or confidence are frames with at least one such detection. Dwell is `last_seen - first_seen` without zones, and first enter to last exit per track and zone with zones (visits without an exit are left out). Trend buckets are camera-local and tracks fall in the bucket of `first_seen`. Ordering is total. A test compiles every valid plan shape (more than 300), validates it, and executes it on Postgres as the read-only login.

### 4.8 llm-service
*Scope note: deferred. The planner calls an OpenAI-compatible endpoint (Ollama locally or a hosted API) with few-shot prompting (D7 default).*
- vLLM with an OpenAI-compatible API, prefix caching on (the schema prompt is long and shared) and structured outputs for `QueryPlan` JSON; the constrained-decoding overhead is measured.
- Small open model chosen by eval; a LoRA adapter whose target is `QueryPlan` JSON (hybrid uses the same plan, so one adapter covers both). Optionally a second adapter with SQL targets for the D11 comparison. If both exist they are served on one quantized base (AWQ or GPTQ) with multi-LoRA; otherwise adapter vs merged overhead is measured and reported.
- Choice of prompting vs LoRA vs hosted **(decided by benchmark, D7)** on structured-query execution accuracy, p95 latency and cost per 1,000 queries.

### 4.9 drift-monitor
- Per camera, compares a rolling window against a reference window captured during a calibration period.
- Signals: class-mix PSI, confidence distribution (KS or Wasserstein), detections per frame, input statistics (brightness, blur). Embedding drift (MMD or centroid shift) is deferred with CLIP.
- Exports gauges to Prometheus. Detection is sequential change detection: a fixed threshold baseline and CUSUM on standardized statistics, each calibrated on clean runs and compared on detection delay against false-alarm rate (task 9.3). Alert rules use the chosen operating point.
- Validated with loadgen drift injection: detection delay and false-alarm rate per drift type, plus the measured mAP drop under each drift to justify why it matters.
- Response: the detector is fine-tuned on a clean plus photometrically drifted mix (task 9.5); before/after mAP on drifted and clean sets is reported. The eval gate and the retraining loop (10.2, 10.5) are deferred.

### 4.10 Evaluation and experiment tracking
- `eval/` holds versioned datasets and the harness:
  - Structured suite (`make eval SUITE=structured`): structured-query execution accuracy. Gold entries store the expected result set, per-dialect gold SQL where needed, a gold `QueryPlan` where expressible and an `expressible_by_plan` flag. Category `unsupported` covers valid analytic questions outside the plan schema (the plan engine must abstain with reason `unsupported`). Reports accuracy over all questions and over expressible ones, plus plan coverage, with plan error classes (invalid plan, plan failed validation, wrong filter, wrong aggregation, wrong group_by) alongside the SQL error classes.
  - Detection mAP on a fixed labeled subset (clean and drifted). Retrieval recall@k and nDCG are deferred with CLIP.
  - Every headline metric is reported with a bootstrap 95% confidence interval, and each metric's documentation says what it does and does not measure.
- Dev and test splits; the test split is frozen and only used for reported numbers.
- Results JSON files in the repo are the only record of runs and the source of truth for reported numbers; comparing two runs is a diff of two results files. There is no experiment tracker: MLflow and W&B were removed in task 0.7 because no kept task needs them (D16).
- DVC tracks large artifacts we produce (synthetic datasets, training data, adapters, fine-tuned weights, large eval snapshots); small eval JSONL and configs stay in git. See section 10.
- (Deferred, 10.2) An eval gate in CD blocks a model or prompt change that regresses against the recorded champion.

### 4.11 Observability
- OpenTelemetry in every service. A frame's trace spans loadgen -> ingest -> queue -> worker -> Triton -> DB write. A query's trace spans every pipeline stage.
- Metrics naming: `vms_<component>_<metric>_<unit>`. Key metrics: ingest frames/s, rejected and dropped frames, queue lag, end-to-end freshness (frame timestamp to queryable), worker batch size and latency, Triton queue and compute time, GPU utilization and memory (DCGM exporter), query latency by mode and stage, LLM tokens and cost.
- Structured JSON logs with trace IDs. Dashboards as code in `deploy/grafana/`.

### 4.12 frontend
Minimal Vite + TypeScript UI: a camera wall with the latest frame per camera and its detection boxes; a query box with the answer, a table or chart chosen from the result kind, and a panel with the plan JSON and compiled SQL; clear states for abstention and validator rejection. Built to make the demo and write-up visual. Semantic mode and feedback buttons are deferred.

**As built (task 1.8, D23):** `frontend/`, vanilla TypeScript with no UI framework or component library (Vite 8, TypeScript 6; 10 KB of JS). The query service serves the built files at `/` (`ui_dir`, default `frontend/dist`), so the browser talks to one origin: no CORS, no proxy container, and the same setup works with `make query` on the host and with the containerized service (`make app`). Camera wall: polls `GET /v1/live` every 1.5 s (per catalog camera: the latest frame with a model version, its detections and the zone polygons), preloads each new image before swapping it in, and draws boxes (colored by class, labeled with class and track number) and zones in an SVG overlay in frame coordinates; toggles hide boxes or zones. Query box: example chips from `GET /v1/examples` (the canned questions); views chosen from the response (`view.ts`): a big number for scalars, an inline SVG bar chart for small single-key groupings and line or bar charts for time series (with the table under both), a thumbnail grid for frames, amber callouts for abstentions and red ones per error type ("Blocked by the SQL validator" for `sql_rejected`, the plan errors listed for `invalid_plan`), and a collapsible panel with the plan, the validated SQL, its parameters, the model, tokens and stage timings. All text is inserted with `textContent`. Pure view logic is unit-tested with vitest; `npm run screenshots` takes the README screenshots with playwright-core.

### 4.13 Shared library (`vms.common`)
Every service builds on the same small modules, each behind an interface so tests inject fakes:
- **Settings:** one pydantic-settings class per service. Values are layered from code defaults, then `configs/common.yaml`, `configs/<service>.yaml` and `configs/<service>.<env>.yaml` (deep-merged, `env` from `VMS_ENV`), then `.env`, then `VMS_*` environment variables (`__` for nesting), then explicit arguments. Validation is strict, so a typo in YAML fails at startup.
- **Observability:** `init_observability(settings)` configures JSON logs that carry the active trace and span IDs, OpenTelemetry tracing (OTLP to the collector; a no-op tracer when disabled) and a Prometheus registry. Metric names are assembled and validated as `vms_<component>_<metric>_<unit>`.
- **IDs and time:** a monotonic ULID generator and a `Clock` protocol (system and fake implementations). Both take an injected clock and randomness source, so tests are deterministic.

## 5. Data flows and guarantees

**Write path:** loadgen -> ingest (store JPEG, insert frame row, publish ref) -> queue -> worker (batch, infer, track, write) -> ack.
- Delivery: at-least-once; idempotent writes keyed on `frame_id` (and `(frame_id, detection_idx)`), so redelivery causes no duplicates.
- Ordering: per camera, via camera leases (D20): one worker holds a camera and claims its frames in capture order. Frames that reach ingest late are processed out of order and counted (`vms_worker_late_frames_total`).
- Overload: backpressure at ingest, measured shedding above rated capacity, never silent loss.
- Shutdown: workers finish the current frame, then hand back claimed frames and leases before exit.

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
| `POST /v1/frames` | Ingest one frame: multipart `camera_id`, `run_id`, `seq`, `ts_unix_ms`, `jpeg`. 2xx accepted (duplicates included), 429/503 shed, anything else failed (`vms.common.ingest_api`) |
| `POST /v1/query` | Structured natural-language query; v0 returns one JSON response (SSE deferred) |
| `GET /v1/catalog` | Cameras, zones and classes the planner knows (query service) |
| `GET /v1/live` | Camera wall: latest processed frame per camera, its detections and zones (query service) |
| `GET /v1/examples` | Example questions for the UI: the canned planner's questions (query service) |
| `GET /v1/frames/{frame_id}/image` | A stored frame's JPEG, read by the query service from the object store |
| `POST /v1/search/semantic` | CLIP search with optional filters |
| `POST /v1/search/hybrid` | Hybrid search |
| `GET /v1/frames/{frame_id}` | Frame metadata (deferred; v0 serves images directly, see above) |
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

- **Local:** `docker compose` CPU profile with small models; GPU profile when a GPU is present. The stack (`deploy/compose/docker-compose.yml`) runs the infrastructure every service needs, each with a health check: Postgres 17 with pgvector (application DB), SeaweedFS as the S3-compatible object store (D12; the frames bucket is created at startup), the OpenTelemetry collector (OTLP in; traces to Tempo, OTLP metrics re-exposed for Prometheus), Prometheus, Tempo, Grafana with provisioned Prometheus and Tempo datasources (fixed UIDs `prometheus` and `tempo`, files in `deploy/grafana/provisioning/`). Redis and MLflow were removed in task 0.7 (D16). Infrastructure services have no profile and always start. The application image (`deploy/docker/Dockerfile`: a Node stage builds the UI, then Python 3.12 slim with the locked environment from `uv.lock`, CPU PyTorch, about 1 GB compressed) runs the query service and UI under profile `app` (`make app`, after `make up` has migrated the database); `make demo` also starts ingest and one worker from the same image (D24). Application services that come in CPU and GPU variants carry `profiles: [cpu]` or `profiles: [gpu]`; `make up` activates `cpu`, `make up PROFILES=gpu` activates `gpu`. Kafka is not in the stack; it is added only if D3 benchmarks justify it (task 6.1, deferred).
- **Kubernetes (deferred):** one Helm chart with per-service values. `kind` in CI for smoke tests. GPU benchmarks run on an EC2 spot GPU instance with single-node k3s. EKS only if a decision record justifies it and the user approves the cost. KEDA scales workers on queue lag; HPA scales query-service on CPU or RPS. Triton gets GPU resource requests.
- **Cloud (AWS by default):** in this plan, `scripts/gpu_session.sh` launches one tagged EC2 spot GPU instance with the AWS CLI, runs scenarios with docker compose and verifies teardown through the Resource Groups Tagging API; an AWS Budgets alarm guards spend (task 2.4). Deferred target design (10.1): Terraform under `deploy/terraform/` provisions, in the project's single configured Region: S3 (frames, DVC remote, artifacts), ECR, IAM with least-privilege roles (GPU instance role limited to the project bucket and ECR pull; CI role limited to ECR push), AWS Budgets alerts, and EC2 spot GPU instances (g5 or g6) running k3s. Every resource is tagged `project=vms` and `session=<id>`. Every apply has a scripted destroy; after destroy, `scripts/gpu_session.sh` lists remaining tagged resources through the Resource Groups Tagging API and fails loudly if any remain. D9 decides only whether long GPU benchmark runs move to a cheaper provider.
- **CI (GitHub Actions):** lint, typecheck, unit tests, integration tests on the compose CPU stack, and the eval harness on a tiny subset with a fake LLM. Image builds, Helm lint, the kind smoke test and the CD eval gate are deferred.

## 10. Repository layout

```
.
├── AGENTS.md
├── README.md
├── Makefile
├── pyproject.toml, uv.lock
├── .dvc/, *.dvc          # DVC config and pointers for large artifacts
├── configs/              # YAML config per service and environment; cameras.yaml (camera catalog)
├── proto/                # gRPC definitions
├── src/vms/
│   ├── common/           # config, logging, telemetry, ids, clock, shared schemas
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

- The query service logs in as `POSTGRES_READONLY_USER`, a member of `vms_query_reader`: SELECT on exactly the columns in the validator's allowlist (column-level grants, so `query_log`, `frames.status` and any new table are unreadable), no write privilege, `default_transaction_read_only` on, and no TEMPORARY on the database. The allowlist is thus enforced by the validator before execution and by Postgres during it; tests check the grants, the config and the migrated schema agree. The query service's reader pool also sets `default_transaction_read_only`, `statement_timeout` (5 s) and `work_mem` (16 MB) on every connection and fetches at most `sql.max_limit` rows; query_log is written through a separate owner connection that only runs one parameterized INSERT (D22).
- User questions are untrusted input. Plan engine: LLM output must pass `QueryPlan` schema and semantic validation, and the compiler emits parameterized SQL, so user values never become SQL text. Both the plan and free-form SQL engines run the `sqlglot` validator on the final SQL as defense in depth, and all SQL runs on the read-only role.
- Rate limiting on the public API.
- Secrets live only in `.env` (gitignored) or a secret manager; `.env.example` documents every variable.
- A red-team question set (DDL, injection, exfiltration, resource exhaustion) is part of the eval suite, with attack success rate reported.

## 12. Cost model

- GPUs are rented per session through scripts: provision -> run -> collect -> destroy, with teardown on failure. AWS EC2 spot (g5 or g6) is the default; D9 may move long benchmark runs to a cheaper provider.
- AWS Budgets alerts are mandatory; every resource carries `project=vms` and `session=<id>` tags, which also drive per-session cost attribution and the post-teardown leftover check.
- Every cloud benchmark and training run records its cost.
- Headline efficiency metric: **cost per camera-hour** = instance $/hour / rated capacity.
