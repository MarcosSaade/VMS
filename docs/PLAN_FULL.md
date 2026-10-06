# Plan (full)

> Location in repo: `docs/PLAN_FULL.md`. Each task has a **prompt** to give the coding agent, **done when** criteria, and where relevant **considerations**. Each phase has an exit gate. `AGENTS.md` rules (tests, experiment log, docs, reproducibility) apply to every task and are not repeated here. If evidence shows a better path, change the plan and record why in `docs/DECISIONS.md`.

## Global considerations

- **Order matters.** The naive baseline (Phase 1) and its measurements (Phase 2) come before any optimization so every later improvement has a before/after number. Don't optimize earlier "while you're there"; log the idea in `EXPERIMENTS.md` instead.
- **Every decision should be defensible in an interview:** context, options, evidence, trade-offs. `DECISIONS.md` is the study guide.
- **Headline numbers** to work toward: rated capacity (cameras at 5 FPS per GPU), freshness p95, query p95 per mode, inference throughput ladder, structured-query execution accuracy, retrieval recall@10, cost per camera-hour, drift detection delay.
- **Honesty:** label synthetic data and self-measured results, report negative results, and never claim "production-ready" unless the checklist in `RUNBOOK.md` (task 10.4) is met.
- **Cost:** GPU work is batched into planned sessions on AWS (the default provider). Prepare scenarios and scripts on CPU first so rented time is spent measuring, not debugging.
- **Optional services stay optional:** W&B, AWS and DVC remotes are never required for `make setup && make up && make demo`.

---

## Phase 0: Foundation

**Goal:** a clean, tested, observable skeleton that makes every later task cheap.
**Exit gate:** fresh clone runs `make setup && make check && make up` successfully; CI green; docs skeleton in place.

### 0.1 Repo scaffold

**Prompt:** Set up the repository per `docs/ARCHITECTURE.md` section 10. Use uv with Python 3.12 and a single package `vms` under `src/` with subpackages per component. Add Makefile targets: `setup, fmt, lint, typecheck, test, test-integration, check, up, down, logs, demo, data, data-mini, data-pull, bench, eval`. Configure ruff (lint + format), mypy strict on `src/vms`, pytest with markers `integration`, `gpu`, `slow` (non-unit markers skipped by default), and pre-commit. Add `.gitignore` covering data, weights, results artifacts above a size threshold, and `.env`. Run `dvc init` with a local remote (a gitignored directory configured in `.dvc/config`) and wire `make data-pull` to `dvc pull`, treating a missing or empty remote as a no-op with a clear message. Add `.env.example` with every variable, including optional `WANDB_API_KEY` and `WANDB_MODE` (default `disabled`). Add one trivial test.
**Done when:** `make setup && make check` passes on a fresh clone; pre-commit runs ruff and mypy; `make data-pull` succeeds without any remote access.

### 0.2 Docs skeleton

**Prompt:** Create every file in the fixed documentation structure from `AGENTS.md` section 6 with headings and short placeholders. `EXPERIMENTS.md` gets the entry format at the top. `DECISIONS.md` gets the template and D1 to D11 from `ARCHITECTURE.md` section 7 with status `proposed`, with Context and Options filled from the matching architecture sections (D3 includes the event-stream question from 4.3; D9 is limited to where long GPU benchmark runs execute, since AWS is the default for all infra; D10 is the stretch orchestrator choice; D11 is the structured query interface, with the typed plan as default until accepted). `RUNBOOK.md` gets sections: Setup, Data, Running locally, Benchmarks, GPU sessions, Deployment, Operations, Troubleshooting, Teardown, Production checklist.
**Done when:** all files exist; no other Markdown files exist in the repo.

### 0.3 Shared libraries
**Prompt:** Implement `vms.common`: settings via pydantic-settings that load `configs/<service>.yaml` with environment overrides; a JSON structured logger that includes trace and span IDs; an OpenTelemetry init function (OTLP exporter, no-op in tests) plus a Prometheus metrics helper enforcing the `vms_<component>_<metric>_<unit>` naming; ULID generation; an injectable `Clock` protocol with a real and a fake implementation; a `Tracker` protocol (start run, log params, log metrics, log artifact reference, end run, return run ID or URL) with MLflow, W&B and no-op implementations selected by config. W&B defaults to disabled or offline mode and its import is lazy, so nothing needs a W&B account. Write unit tests for each; Tracker tests use the no-op implementation and a recording fake, never a real backend.
**Done when:** tests pass; a sample service can log, trace and expose metrics with three lines of setup; switching tracker is a config change.
**Considerations:** roles are fixed by convention: W&B for training runs, sweeps and reports (LoRA, detector fine-tune); MLflow for eval results, model registry and champion tracking. Both may be enabled for one run; the results JSON records every tracker run ID or URL.

### 0.4 CI

**Prompt:** Add a GitHub Actions workflow running `make check` on push and pull request with uv caching. Add a placeholder integration job that will run against the compose stack later. Fail on any lint, type or test error.
**Done when:** CI is green on main and shows a failing run when a test is broken deliberately on a branch.

### 0.5 Compose stack

**Prompt:** Create `deploy/compose/` with Postgres (pgvector image), MinIO (with bucket bootstrap), Redis, OpenTelemetry collector, Prometheus, Tempo, Grafana (provisioned datasources) and MLflow (Postgres backend, MinIO artifacts). Add health checks. `make up` starts the stack and waits until healthy; `make down` stops it; `make logs` tails logs. Use a CPU profile by default and a `gpu` profile for later services.
**Done when:** `make up` reaches healthy on a laptop; Grafana shows Prometheus and Tempo datasources; MLflow UI loads.
**Considerations:** Kafka is not part of this stack; it is added in task 6.1 only if benchmarked. W&B is a hosted service and is not in compose.

### 0.6 Query assets (built from scratch)

**Prompt:** Build the query assets from scratch; the old smart-vision prototype is not used (user decision, 2026-10-06). Draft the `QueryPlan` Pydantic model in `vms.query` (finalized in 1.7). Author few-shot examples as (question, gold `QueryPlan`, gold SQL) triples covering every intent, time-range kind and abstain reason; questions the plan schema cannot express carry an `unsupported` plan plus their SQL, which also serves the free-form engine and result checks. Implement the SQL validator on `sqlglot` ASTs (single SELECT, table/column allowlist, function denylist, LIMIT injection) with thorough tests, including injection attempts, multiple statements, comments and DDL; it guards compiled SQL from the plan engine and generated SQL from the free-form engine. Write versioned prompt templates for the planner and the free-form SQL engine. Store few-shots and prompts as versioned data under `eval/` or `configs/`, not hardcoded strings, and reference them from config.
**Done when:** validator has comprehensive unit tests passing; few-shot triples validate against the `QueryPlan` schema and their SQL passes the validator; assets are in place and referenced from config.
**Considerations:** these few-shot examples must later be kept out of the eval test split (task 3.1). Few-shot camera and zone names (entrance, checkout, parking_lot, loading_dock) are illustrative; align the demo camera config (task 1.4) with them or update the bank. The validator's table allowlist in `configs/query.yaml` must match the migrations (task 1.4).

---

## Phase 1: Naive end-to-end baseline

**Goal:** a deliberately simple system that works end to end. Simplicity is the point: it is the "before" for every optimization.
**Exit gate:** `make demo` runs end to end on a CPU laptop; integration tests pass; `ARCHITECTURE.md` notes what v0 actually does.

### 1.1 Datasets

**Prompt:** Write a fetcher (`python -m vms.datasets`, invoked by `make data`) that fetches public video suitable for virtual cameras, verifies checksums and extracts clips into `data/` (gitignored). Candidates: MOT17/MOT20 sequences and the VIRAT Video Dataset; check current availability and licenses and record them in `RUNBOOK.md` under Data. Produce a clip manifest (YAML) with clip ID, source, duration, resolution, scene type (indoor/outdoor, day/night) and license. Aim for variety so drift scenarios and retrieval queries are possible later.
**Done when:** `make data` is reproducible and idempotent; manifest committed; licenses documented.
**Considerations:** use the same clips for every benchmark. A small `make data-mini` subset keeps CI and laptops fast. Done 2026-10-06: sources, checksums and licenses in `configs/data/sources.yaml`, measured manifest in `configs/data/manifest.yaml`; VIRAT is opt-in because its agreement must be signed by each user (D15).

### 1.2 Loadgen v0

**Prompt:** Implement `vms.loadgen`: async, N virtual cameras, each looping assigned clips at a configured FPS and resolution, sending JPEG frames with camera ID, timestamp and sequence number to the ingest REST endpoint. Pre-decode and pre-encode frames into memory at startup so the generator stays cheap. Config: cameras, FPS, resolution, JPEG quality, duration, clip assignment, seed. Expose metrics for sent, accepted, rejected and failed frames and send latency. Add a self-test mode that measures the generator's own max rate without a server.
**Done when:** 20 cameras at 5 FPS run stably against a stub server on a laptop; self-test rate recorded in `EXPERIMENTS.md`.

### 1.3 Ingest v0

**Prompt:** Implement `vms.ingest` as a FastAPI service: accept multipart frames, write the JPEG to the object store (SeaweedFS locally, S3 in the cloud; D12) at `frames/{camera_id}/{date}/{frame_id}.jpg`, insert a `frames` row with status `pending`. Idempotency on `(camera_id, seq)` or `frame_id`. Health, readiness and metrics endpoints.
**Done when:** integration test sends frames and finds objects and rows; duplicate sends create no duplicates.

### 1.4 Schema v1

**Prompt:** Implement the logical schema from `ARCHITECTURE.md` section 4.6 in Postgres with Alembic migrations, including pgvector for embeddings, camera timezones and zone polygons. Load cameras and zones from YAML config. Create a read-only role for the query service. Keep indexes simple for now (this is the baseline).
**Done when:** `alembic upgrade head` and downgrade work on a fresh DB; repository layer has tests.

### 1.5 Worker v0

**Prompt:** Implement `vms.worker` v0: poll `frames` for `pending` rows, fetch the JPEG, run YOLO11 (start with `s`) and CLIP ViT-B/32 (open_clip) in-process with PyTorch at batch size 1, write detections and an embedding for every frame, compute input statistics (brightness, contrast, Laplacian variance), mark the frame done. Device configurable (CPU or CUDA). Record model versions. Use injected model interfaces so unit tests run with fakes.
**Done when:** frames flow from loadgen to stored detections and embeddings; unit tests use fakes; an integration test with tiny models passes on CPU.
**Considerations:** polling the DB is intentionally naive; it becomes the "before" for the queue in Phase 6.

### 1.6 Tracking and zone events

**Prompt:** Add per-camera ByteTrack tracking in the worker (a maintained implementation such as the one in `supervision` is fine), write `track_id` on detections, maintain the `tracks` table, and emit zone enter, exit and loiter events from zone polygons. Make loiter threshold configurable. Test with synthetic trajectories.
**Done when:** tracks and events appear for demo clips; unit tests cover enter/exit/loiter edge cases (track lost and re-found, boundary jitter).
**Considerations:** tracking requires in-order processing per camera. This constrains worker scaling and queue design later (task 6.1); note it in `ARCHITECTURE.md`.

### 1.7 Query service v0

**Prompt:** Implement `vms.query` with FastAPI per `ARCHITECTURE.md` section 4.7. Define the `StructuredQueryEngine` interface and implement the typed-plan engine as primary: the `QueryPlan` Pydantic model with exported JSON schema and `schema_version`; a planner prompt with the plan schema, a schema summary, camera timezones, rules and static (question, plan) few-shots; an OpenAI-compatible LLM client (configurable base URL so Ollama, vLLM and hosted APIs all work) using constrained decoding (JSON-schema structured outputs, or tool calling where that is what the provider supports); plan validation (schema plus semantic checks) with one repair attempt using the validation error; abstention with reason; a deterministic Postgres compiler to parameterized SQL; the sqlglot validator on the compiled SQL as defense in depth; execution on the read-only role with statement timeout and row cap; SSE streaming of answer, structured results, plan JSON and compiled SQL. Semantic mode: CLIP text embedding in-process, pgvector cosine top-k with optional camera and time filters, presigned frame URLs. Log every query to `query_log` with engine, plan, compiled SQL and stage timings.
**Done when:** unit tests with a fake LLM cover planning, invalid plans and repair, abstention, compilation (golden SQL per intent, parameters never inlined), validator rejection and SSE framing; integration test answers a handful of real questions against demo data.
**Considerations:** relative time ranges ("this morning") resolve in the camera timezone inside the compiler, not the LLM, so they are testable with the injected clock.

### 1.8 Minimal frontend

**Prompt:** Build a small Vite + React + TypeScript app: query box with mode selector (structured, semantic), streaming answer, result table or frame grid, a debug panel showing the plan JSON and compiled SQL, thumbs up/down posting to a feedback endpoint. Keep styling minimal.
**Done when:** usable against the local stack; served by compose.

### 1.9 Demo

**Prompt:** Add `make demo`: start the stack, run loadgen with 4 cameras for a few minutes, then print answers to a few example structured and semantic queries. Add an integration test in CI that runs a shortened version on CPU with tiny models and a fake or tiny LLM (the fake returns canned plans), with no W&B, AWS or DVC remote access.
**Done when:** `make demo` works on a fresh clone with only Docker and uv installed; the CI integration job runs it.
**Considerations:** the demo uses `make data-mini` (about 180 MB from motchallenge.net and Wikimedia Commons). Cache `data/` in CI keyed on the hash of `configs/data/sources.yaml`, so CI does not hit those hosts on every run.

---

## Phase 2: Observability and baseline load test

**Goal:** see inside the system and produce the baseline numbers every optimization will be compared against.
**Exit gate:** `BENCHMARKS.md` has a baseline section (CPU and one GPU type) with rated capacity, freshness, query latency per mode and a written bottleneck analysis.

### 2.1 Instrumentation

**Prompt:** Instrument all services with OpenTelemetry so one frame's trace spans loadgen, ingest, worker, model inference and DB writes, and one query's trace spans every pipeline stage. Add metrics: ingest frames/s, rejected and dropped, processing backlog, end-to-end freshness histogram (frame timestamp to detections written), worker inference and write latency, query latency by mode and stage, LLM tokens. Add the DCGM exporter to the GPU profile.
**Done when:** a demo run shows complete traces in Tempo and all metrics in Prometheus; tests assert key metrics are emitted.

### 2.2 Dashboards

**Prompt:** Create Grafana dashboards as code in `deploy/grafana/`, auto-provisioned: Pipeline (ingest rate, backlog, freshness, drops), Inference (latency, batch size, GPU utilization and memory), Query (latency by mode and stage, errors, tokens), Resources (CPU, memory per service).
**Done when:** dashboards load automatically on `make up` and display live data during the demo.

### 2.3 Benchmark framework

**Prompt:** Implement `vms.bench`: scenarios in `bench/scenarios/*.yaml` describing ingest load (cameras, FPS, ramp steps, hold time), query load (mix, concurrency, duration) and warmup. The runner orchestrates loadgen and a query load generator (k6 or Locust), scrapes Prometheus over the steady-state window, and writes a results JSON with the fields required by `AGENTS.md` section 7. Add a command that renders results into Markdown tables for `BENCHMARKS.md`. Wire `make bench SCENARIO=...`.
**Done when:** a small scenario runs locally and produces a valid results file; the result schema has a test.
**Considerations:** check that the load generators are not the bottleneck; record their CPU.

### 2.4 GPU bench environment

**Prompt:** AWS is the default provider. Write Terraform in `deploy/terraform/gpu-bench/`, in the project's single configured Region, that provisions: an S3 bucket (artifacts and the DVC remote, configured as a DVC remote by this task), an ECR repository, IAM with least-privilege roles (GPU instance profile limited to the project bucket and ECR pull), AWS Budgets alerts, and one EC2 spot GPU instance (g5 or g6) with Docker, NVIDIA drivers and single-node k3s. Tag every resource `project=vms` and `session=<id>`. Add `scripts/gpu_session.sh` that provisions, syncs code, runs a named list of scenarios, copies results back and destroys everything, with teardown in a trap so it runs on failure; after destroy it lists resources still tagged with the session ID through the Resource Groups Tagging API and exits non-zero, loudly, if any remain. Then write D9, which decides only whether long GPU benchmark runs move to a cheaper provider, presenting cost per hour and spot availability for g5/g6 in the project Region vs alternatives. Ask the user for confirmation before the first real apply.
**Done when:** one dry run and one real short session complete with automatic teardown verified by the tag check; cost recorded; `RUNBOOK.md` GPU sessions section written.
**Considerations:** before the first apply, confirm the account plan allows GPU instances and that the spot vCPU quota for G instances in the project Region is above zero; request an increase early because approval can take days. EKS is used only if a decision record justifies it and the user approves the cost.

### 2.5 Baseline load tests

**Prompt:** Write scenarios for (a) an ingest ramp: step cameras through 1, 2, 4, 8, 16, ... at 5 FPS, hold each step long enough for steady state, and record freshness p95, backlog growth and drops to find rated capacity; (b) mixed query load at several concurrency levels while ingest runs at about 70% of rated capacity. Run on CPU locally and on one GPU type via a GPU session.
**Done when:** results files exist for both environments; rated capacity determined per the definition in `ARCHITECTURE.md` section 2.

### 2.6 Baseline report

**Prompt:** Profile the worker and query service under load (py-spy flame graphs, trace breakdowns, GPU utilization). Write the baseline section of `BENCHMARKS.md`: tables, hardware, methodology, and a ranked list of bottlenecks with evidence. Use this list to sanity-check the order of Phases 4 to 6, and propose plan changes if the evidence suggests a different order.
**Done when:** baseline section complete; bottleneck list links to traces or profiles saved as artifacts.

---

## Phase 3: Evaluation harness

**Goal:** trustworthy quality measurement, so speed work can't silently hurt quality and LLM work has a real target.
**Exit gate:** `make eval` reports structured-query execution accuracy (dev and test, over all and over plan-expressible questions, plus plan coverage), retrieval metrics and detection mAP for the baseline, logged to MLflow and recorded in `BENCHMARKS.md`.

### 3.1 Structured-query dataset

**Prompt:** Build a frozen eval DB snapshot generated deterministically (seeded replay of demo clips plus synthetic additions) so gold answers never change; track the snapshot with DVC and keep the generator script and seed in git. Draft 300+ questions across categories: count, aggregate, time window (with camera timezones), joins across tracks/events/zones, trend, top-k, dwell time, comparison, ambiguous (expected behavior: ask or abstain), unanswerable (abstain), unsupported (a valid analytic question outside the plan schema; the plan engine must abstain with reason `unsupported`), adversarial (refuse). Include paraphrases. Each entry stores gold SQL (per dialect where needed), the expected result set from executing it, a gold `QueryPlan` where expressible, and an `expressible_by_plan` flag; check that each gold plan compiles and returns the expected result set. Use a strong model to draft candidates, but flag that the user must review a random sample of dev and all of test before freezing. Split 60/40 dev/test stratified by category; record the test file hash. Store as JSONL under `eval/structured/`.
**Done when:** dataset passes schema validation; every gold SQL and every gold plan executes and matches the expected result set; the user has reviewed and approved; test hash recorded in `DECISIONS.md` or `BENCHMARKS.md`.
**Considerations:** few-shot examples and any future training data must not overlap the test set (exact or near duplicates). Storing expected results, not only SQL or plans, keeps the set valid after a change of backend in Phase 4. Keep `unsupported` honest: these questions should be ones a user could plausibly ask, so they measure the real cost of the plan schema.

### 3.2 Scorer and taxonomy

**Prompt:** Implement the structured eval harness: run each question through any `StructuredQueryEngine` (selected by config), compare result sets with the stored expected results (order-insensitive unless the question implies order, numeric tolerance, column-name agnostic), and classify failures. Plan error classes: invalid plan, plan failed validation, wrong filter, wrong aggregation, wrong group_by. SQL error classes: invalid SQL, rejected by validator, wrong table or column, wrong filter, wrong time handling, wrong aggregation, wrong join. Shared: should have abstained, abstained wrongly, wrong abstain reason. Report accuracy over all questions and over plan-expressible ones, plan coverage, abstention correctness, and accuracy per category, plus latency, tokens and cost. Include a red-team subset (DDL, injection, exfiltration, resource exhaustion) and report attack success rate. Log runs through the `Tracker` (MLflow) and write results JSON with the tracker run ID. Parallelize with a concurrency limit.
**Done when:** `make eval SUITE=structured SPLIT=dev` runs; scorer edge cases are unit tested; red-team results reported.

### 3.3 Retrieval eval

**Prompt:** Create 60 to 100 natural-language visual queries relevant to the clips (e.g., people carrying bags, cyclists, crowded scenes, vehicles at night). Build a small labeling CLI that pools candidates (CLIP top-50 plus random frames) and lets the user mark relevant frames quickly. Compute recall@10, nDCG@10 and MRR. Store labels under `eval/retrieval/`.
**Done when:** labels exist for all queries; `make eval SUITE=retrieval` reports metrics.
**Considerations:** pooling from CLIP biases the labels toward CLIP; adding random and detection-filtered candidates reduces this. Note the limitation in the write-up.

### 3.4 Detection quality

**Prompt:** Fix a labeled evaluation subset for detection (e.g., 500 images from COCO val2017, plus MOT ground truth for person boxes if useful) with a frozen file list. Implement mAP@50 and mAP@50-95 measurement for any inference backend behind the worker's model interface.
**Done when:** `make eval SUITE=detection` reports baseline mAP for PyTorch YOLO11.

### 3.5 Baseline eval

**Prompt:** Run all suites (`structured`, `retrieval`, `detection`) on the baseline system, record results in `BENCHMARKS.md` and `EXPERIMENTS.md` (tag `query-eval` for the structured suite), and add a CI job that runs the harness on a tiny subset with a fake LLM and the no-op tracker to keep it from breaking.
**Done when:** baseline quality numbers published; CI harness job green.

---

## Phase 4: Storage

**Goal:** choose the detection and vector stores with evidence (D1, D2).
**Exit gate:** D1 and D2 accepted in `DECISIONS.md` with linked evidence; system migrated; eval re-run with no unexplained regression.

### 4.1 Synthetic detection generator

**Prompt:** Fit distributions from real Phase 1 replay data (class mix per camera and hour, detections per frame, track lengths, event rates, diurnal pattern) and implement a deterministic, seeded generator that writes Parquet chunks at 10M, 100M and optionally 1B detection rows with matching frames, tracks and events.
**Done when:** generated data matches fitted distributions within tolerance (tested); generation is reproducible by seed; generated Parquet is tracked with DVC (or regenerated by seed, with the choice stated).

### 4.2 Storage workload

**Prompt:** Define the storage benchmark: (a) sustained ingest at several batch sizes and concurrencies, (b) a query suite of about 20 representative queries derived from the structured eval categories (gold plans and SQL) (point lookup by frame, time-range scan, aggregation by class and hour, top-k, joins to tracks and events, dwell time), (c) mixed reads during ingest. Specify cold and warm runs and how caches are dropped.
**Done when:** workload defined as scenarios with a runner that works against any candidate store through an adapter interface.

### 4.3 Detection store benchmark

**Prompt:** Benchmark three candidates on the same machine with equal resource limits and a fair, documented tuning effort for each: Postgres 16 with daily partitioning, BRIN on time and B-tree on (camera_id, ts); TimescaleDB hypertables with compression and continuous aggregates; ClickHouse MergeTree with partition by day, a sort key such as (camera_id, class_name, ts), LowCardinality columns and suitable codecs. Measure ingest rows/s, query p50/p95 per query class (cold and warm), bytes per row, CPU and memory, at 10M and 100M rows (1B on a rented machine only if the user approves the cost).
**Done when:** results files and a comparison table in `BENCHMARKS.md`; per-store tuning documented.
**Considerations:** column stores handle updates poorly, so mutable track state may stay in Postgres (a hybrid is a valid outcome). Make sure the harness, not the database, isn't the bottleneck.

### 4.4 Vector store benchmark

**Prompt:** Benchmark pgvector (HNSW, sweeping m, ef_construction, ef_search) against Qdrant (HNSW with payload indexes) at 1M and 10M 512-dimensional vectors built from real CLIP embeddings (augment if needed and say so). Measure recall@10 against exact search, p95 latency and QPS, and repeat with camera and time filters at several selectivities. Measure index build time and memory.
**Done when:** recall vs latency curves and the filtered-search comparison are in `BENCHMARKS.md`.
**Considerations:** filtered ANN behavior is the interesting part for hybrid search; highly selective filters can collapse HNSW recall in some engines.

### 4.5 Decide and migrate

**Prompt:** Write D1 and D2 with evidence, including operational complexity and compiler complexity: each viable detection store needs a plan-to-SQL compiler, with equivalence tests showing the same plan returns the same result set on every backend (run the structured dev split's gold plans through each compiler). Model accuracy no longer depends on dialect, but report the free-form SQL path's per-dialect gold SQL status where it matters for D11. Migrate the system to the winners, update schema, migrations, compilers and few-shots, and re-run all eval suites and the ramp test.
**Done when:** decisions accepted; compiler equivalence tests green for each viable backend; integration tests green; eval and ramp results compared with baseline in `BENCHMARKS.md`.

---

## Phase 5: Inference optimization

**Goal:** a measured optimization ladder for YOLO and CLIP, ending with the worker on Triton (D5).
**Exit gate:** ladder table (throughput, latency, GPU memory, mAP or recall) in `BENCHMARKS.md`; worker uses Triton; ramp re-run shows the end-to-end effect.

### 5.1 Export and Triton repository

**Prompt:** Write export scripts for YOLO11 to ONNX with dynamic batch, and for CLIP image and text encoders as separate ONNX models. Create the Triton model repository under `models/triton/` with `config.pbtxt` files; weights are fetched by script. Add parity tests comparing ONNX outputs with PyTorch within tolerance on fixed inputs. Add Triton to compose (GPU profile) and Helm.
**Done when:** Triton serves all three models; parity tests pass.

### 5.2 Precision ladder

**Prompt:** On one rented GPU type, measure each step: PyTorch eager, ONNX Runtime CUDA, TensorRT FP16, TensorRT INT8 (calibrate on a representative frame set that is disjoint from the eval sets). Use `perf_analyzer` for throughput and latency at several batch sizes, and record GPU memory. For each step, run detection mAP (3.4) for YOLO and retrieval recall (3.3) for CLIP. Build TensorRT engines on the target GPU by script.
**Done when:** ladder table complete with quality columns; INT8 kept only if the quality drop is within a threshold you state in `DECISIONS.md` beforehand.

### 5.3 Batching sweep

**Prompt:** Sweep Triton dynamic batching (max batch size, preferred sizes, max queue delay) and instance group counts, and plot the throughput vs latency frontier for each model. Choose an operating point that meets the freshness SLO with headroom.
**Done when:** frontier plots saved as artifacts and summarized in `BENCHMARKS.md`; chosen config committed.

### 5.4 Pre- and post-processing

**Prompt:** Profile the full worker path under load. If decode, resize or NMS dominate, move them to the GPU (Triton ensemble, DALI, TensorRT NMS plugin, or a small custom C++/CUDA implementation as a custom Triton backend or TensorRT plugin) and measure end to end. Attempt the custom C++/CUDA option only if profiling shows preprocessing dominates, and benchmark it against the DALI path. If they don't dominate, record that and stop.
**Done when:** decision and measurements logged; end-to-end worker throughput recorded.

### 5.5 Integrate and decide D5

**Prompt:** Switch the worker to Triton over gRPC with client-side batching across cameras. Re-run the ingest ramp on the same GPU type as the baseline and report rated capacity before vs after. Write D5.
**Done when:** D5 accepted; `BENCHMARKS.md` shows model-only and end-to-end improvement separately.
**Considerations:** the end-to-end number (cameras per GPU) is the headline; model-only speedups are supporting detail.

---

## Phase 6: Scaling the pipeline

**Goal:** a horizontally scalable, failure-tolerant pipeline on Kubernetes with autoscaling, with transport and queue choices justified (D3, D4, D6, D8).
**Exit gate:** rated capacity measured on the GPU cluster; chaos tests show no frame loss below capacity; autoscaling demonstrated; decisions accepted.

### 6.1 Queue

**Prompt:** Replace DB polling with a queue carrying frame refs, keyed or partitioned by camera so per-camera ordering is preserved for tracking. Add Apache Kafka (KRaft mode, single broker) to compose under a profile, and benchmark Redis Streams against Kafka for the realistic message size and rate, ordering guarantees, consumer rebalancing behavior, replay and operational cost. Implement the chosen one with consumer groups, ack-after-write and a dead-letter stream. Also evaluate whether detections and events should flow through a Kafka topic that feeds the detection store (for example the ClickHouse Kafka engine or a consumer) and the drift monitor, for decoupling and replay: measure write throughput and freshness against direct writes and weigh the added operations. Decide both inside D3; if Redis Streams wins the frame queue, still record whether the event stream benefits from Kafka.
**Done when:** D3 accepted with numbers for both questions; ramp test shows the effect vs polling.
**Considerations:** how tracker state moves when a camera is reassigned to another worker during rebalancing must be handled explicitly (e.g., reset with a logged discontinuity, or state handoff). Do not adopt Kafka without benchmark evidence; "Redis Streams suffices at this scale" is a valid outcome, and if Kafka is not adopted it is removed from compose.

### 6.2 gRPC ingest

**Prompt:** Define `proto/ingest.proto` per `ARCHITECTURE.md` section 6, generate code via a Makefile target, implement client-streaming ingest in the gateway and loadgen, and benchmark against REST multipart: frames/s per core, p99 latency, CPU per frame. Keep the public API on REST. Write D4.
**Done when:** D4 accepted with numbers; both transports tested.

### 6.3 Reliability

**Prompt:** Implement and test: idempotent writes for redelivered messages, backpressure at ingest when queue lag passes a threshold (gRPC `RESOURCE_EXHAUSTED`), configurable shedding policy above capacity, dead-letter handling for poison frames, graceful shutdown that drains in-flight batches, timeouts and bounded retries everywhere. Add fault-injection integration tests (kill a worker mid-batch, DB unavailable briefly, object store slow).
**Done when:** fault tests pass and show zero loss and zero duplicates below capacity.

### 6.4 Helm and kind

**Prompt:** Create one Helm chart under `deploy/helm/` with per-service values, resource requests and limits, GPU requests for Triton, probes, config maps and secrets wiring. Add `make k8s-smoke` that creates a kind cluster, installs the chart in the CPU profile, runs a short demo and tears down; run it in CI.
**Done when:** CI kind smoke test green; chart passes `helm lint`.

### 6.5 Autoscaling

**Prompt:** Add a KEDA ScaledObject scaling workers on queue lag, and an HPA for the query service on CPU or request rate. Under a ramp scenario on the GPU cluster, record time to scale up, lag recovery time and scale-down behavior. Consider how worker count relates to Triton capacity (GPUs are the hard limit) and document the policy.
**Done when:** autoscaling plots and numbers in `BENCHMARKS.md`.

### 6.6 Capacity and chaos on the GPU cluster

**Prompt:** Run the full ramp and mixed query load on the GPU cluster to measure rated capacity and SLOs. Run chaos tests: kill worker pods, restart Triton, inject DB latency, build a backlog and measure drain rate. Using measured CPU, GPU and memory profiles per component, write D6 on which components deploy separately and why (merging is a valid outcome).
**Done when:** D6 accepted; chaos results in `BENCHMARKS.md`.

### 6.7 Embedding sampling

**Prompt:** Compare embedding every frame, a fixed rate (e.g., 1 FPS) and scene-change triggering (frame difference or perceptual hash threshold): retrieval recall@10 (3.3), GPU time and storage. Write D8 and apply the chosen policy.
**Done when:** D8 accepted with the trade-off table.

---

## Phase 7: Search quality

**Goal:** a strong hybrid search mode, measurable structured-query gains, and an evidence-based choice of structured query interface (D11).
**Exit gate:** all eval suites re-run on test with improvements attributed to specific changes; D11 accepted.

### 7.1 Hybrid search

**Prompt:** Implement hybrid queries on the shared `QueryPlan`: the same planner emits a plan with `visual_query` set (no separate hybrid schema); execute filtered ANN, or a compiled SQL prefilter followed by vector rerank, choosing by estimated selectivity with thresholds measured on the vector benchmark from 4.4. Add 40+ labeled hybrid queries to the eval and report metrics.
**Done when:** hybrid mode in API and frontend; hybrid eval results recorded.

### 7.2 Query engine improvements and D11

**Prompt:** On the dev split only, ablate each change to the plan engine independently: few-shot retrieval from an example bank by embedding similarity, schema and plan pruning (only relevant tables, enums and plan fields in the prompt), repair variants, and a result cache keyed by normalized question plus data watermark. Implement the free-form SQL engine behind the `StructuredQueryEngine` interface (pruned schema DDL prompt, sqlglot validation, one self-correction attempt). Optionally prototype an agent loop (tool calling over the plan executor and CLIP search). Compare engines on the dev split by execution accuracy, abstention correctness, p95 latency, tokens and cost, expressiveness coverage (share of questions expressible by the plan schema) and red-team attack success rate, then run the chosen configurations once on test. Write D11.
**Done when:** ablation table and engine comparison in `BENCHMARKS.md`; cache hit rate and latency effect measured; D11 accepted.
**Considerations:** the agent loop is strictly optional. The typed plan stays the default until D11 is accepted; a hybrid outcome (plan by default, SQL fallback for `unsupported`) is valid if the evidence supports it and the red-team rate stays acceptable.

### 7.3 Error analysis

**Prompt:** Re-run all suites, then analyze remaining failures by category with examples. Turn real failures from `query_log` feedback into new dev examples (never test).
**Done when:** error analysis section in `BENCHMARKS.md`.

---

## Phase 8: LLM serving and LoRA

**Goal:** efficient self-hosted LLM serving and an evidence-based choice between prompting, fine-tuning and hosted models (D7).
**Exit gate:** D7 accepted with accuracy, latency and cost per 1,000 queries for each option.

### 8.1 vLLM serving

**Prompt:** Serve candidate small open models with vLLM behind the OpenAI-compatible interface; check which current small models are strongest at structured JSON output and SQL before choosing candidates. Enable prefix caching (the schema prompt is long and shared) and measure its effect. Measure constrained-decoding overhead: structured outputs on vs off for `QueryPlan` generation (tokens/s, time to first token, p95, and plan validity rate). Load test concurrent requests for tokens/s, time to first token and p95, and compare with llama.cpp or Ollama on the same GPU. Add the llm-service to compose and Helm.
**Done when:** serving comparison in `BENCHMARKS.md`; query service runs against vLLM.

### 8.2 Training data

**Prompt:** Generate training (question, `QueryPlan`) pairs with a strong model over the schema and varied synthetic DB states. Filter: the plan is schema-valid, compiles successfully, executes, and the result sets of two independent generations agree. Deduplicate and run leakage checks against the test set: exact match, near-duplicate by embedding similarity and template overlap. Target 2k to 10k pairs; record generation cost and filter pass rates. If a SQL adapter is planned for D11 (8.3), keep the compiled or independently generated SQL for each pair as well.
**Done when:** dataset tracked with DVC (generation script, seed and filter config in git); leakage report shows no overlap.

### 8.3 LoRA fine-tune

**Prompt:** Fine-tune the chosen base with LoRA (PEFT or Unsloth) on a rented GPU via the session script, with plan JSON as the target. Log training runs and sweeps to W&B and eval results to MLflow through the `Tracker`. Ablate data size (e.g., 500, 2k, all) and rank on the dev split; evaluate the best once on test. Optionally train a second adapter with SQL targets on the same questions for the D11 comparison.
**Done when:** ablation table and final test numbers recorded; adapter weights tracked with DVC; candidate registered in the MLflow registry.

### 8.4 Quantization and multi-LoRA

**Prompt:** Quantize the base model (AWQ or GPTQ) and measure the accuracy and latency effect. If both a plan adapter and a SQL adapter exist, serve them on one base with vLLM multi-LoRA and measure overhead vs merged single models. If only the plan adapter exists (hybrid shares the plan, so there is no separate hybrid adapter), measure adapter vs merged serving overhead and state that multi-LoRA was not needed.
**Done when:** results recorded; serving config committed.

### 8.5 Decide D7

**Prompt:** Compare hosted model, base model few-shot, LoRA, and quantized LoRA on structured-query execution accuracy (test), p95 latency under load and cost per 1,000 queries (including amortized GPU cost). Write D7 honestly, including the case where fine-tuning loses.
**Done when:** D7 accepted; default model in config matches the decision.

---

## Phase 9: Drift detection

**Goal:** demonstrate that the system notices when its inputs or outputs change, with measured detection delay and false-alarm rate, and that the detector can be recovered on drifted data.
**Exit gate:** drift detection results table in `BENCHMARKS.md`; alerts and runbook procedure in place; drift fine-tune results table in `BENCHMARKS.md`.

### 9.1 Drift statistics

**Prompt:** Implement `vms.drift`: per camera, compare rolling windows with a reference window from a calibration period. Signals: class-mix PSI, confidence distribution (KS and Wasserstein), detections per frame, input statistics, embedding drift (MMD on sampled embeddings and centroid shift). Export gauges to Prometheus. Unit test each statistic on synthetic distributions with known shifts.
**Done when:** statistics tested; gauges visible during a demo run.

### 9.2 Drift injection

**Prompt:** Add scheduled transforms to loadgen: gradual darkening, blur, noise, camera shift, occlusion and class-mix change (switching clips). Log onset times in the results. Measure detection mAP under each transform on the labeled subset so the impact of each drift is quantified.
**Done when:** scenarios exist for each drift type; mAP impact table recorded.

### 9.3 Alerts and scoring

**Prompt:** Calibrate thresholds on clean runs to a target false-alarm rate, add Alertmanager rules, then run drift scenarios and measure detection delay per drift type, false alarms per camera-day, and which signals catch which drift.
**Done when:** results table in `BENCHMARKS.md`.

### 9.4 Dashboard and runbook

**Prompt:** Add a Drift dashboard and a `RUNBOOK.md` procedure: what each alert means, how to confirm it, and the response options (recalibrate reference, flag the camera, route frames for labeling, retrain).
**Done when:** dashboard provisioned; runbook section written.

### 9.5 Detector fine-tune on drifted data

**Prompt:** Using the mAP-under-drift table from 9.2, take the worst photometric drifts (darkening, blur, noise). Photometric transforms keep bounding boxes valid, so apply them to a labeled set disjoint from the eval subsets. Fine-tune YOLO11 on a clean plus drifted mix. Report mAP before and after on the drifted held-out eval AND on the clean eval (forgetting check), plus training time and cost. Run a small sweep over learning rate and mix ratio with W&B. Store weights with DVC.
**Done when:** results table in `BENCHMARKS.md`; weights tracked with DVC; limitation stated that augmentation-based drift is not real domain shift.
**Considerations:** geometric drifts (camera shift) need box adjustment, so skip them or handle them explicitly. Pseudo-labeling with a larger model is an alternative source of labels. The resulting detector becomes a candidate champion for the 10.2 eval gate.

---

## Phase 10: Cloud and continuous delivery

**Goal:** reproducible cloud deployment, a CD pipeline with an eval gate, and the final full-scale numbers.
**Exit gate:** a tagged release deploys through CD to a provisioned cluster; final benchmark published; production checklist complete or gaps stated.

### 10.1 Cloud stack

**Prompt:** Extend the AWS Terraform from 2.4 into a full environment in the project Region: EC2 spot GPU node(s) (g5 or g6) running k3s, S3 (frames, artifacts, DVC remote), ECR, IAM with least-privilege roles (instance, CI push, read-only where possible), secrets (AWS Secrets Manager or SSM Parameter Store) and AWS Budgets alerts, with every resource tagged `project=vms` and `session=<id>`. Use EKS only if a decision record justifies it and the user approves the cost. Keep a single command each for up and down; down ends with the tagged-resource check from 2.4. Confirm cost with the user before applying.
**Done when:** environment comes up and down by script; the tag check reports zero leftovers; `RUNBOOK.md` Deployment and Teardown sections updated.

### 10.2 CD with eval gate

**Prompt:** Add a release workflow: on a version tag, build and push images, then deploy with `helm upgrade` behind a manual approval (the cluster is ephemeral). Add an eval gate that runs the structured dev subset (about 100 questions) and detection mAP (clean and drifted) against the candidate model, adapter or prompt and fails if results drop more than a stated threshold below the champion recorded in `eval/champion.json`. Log gate runs to MLflow; track candidates and the champion in the MLflow model registry, with `eval/champion.json` mirroring the champion's metrics in git so the gate works from the repo alone. Promote the candidate to champion only on pass. If no LLM key is available in CI, skip with a clear warning.
**Done when:** a deliberately degraded prompt is blocked by the gate; a normal change passes; the 9.5 detector has gone through the gate as a candidate.

### 10.3 Full-scale benchmark

**Prompt:** Run the final ingest ramp and mixed query load on the cloud environment. Report rated capacity, freshness p95, query p95 per mode and cost per camera-hour, with all settings and hardware.
**Done when:** final section in `BENCHMARKS.md`; resources torn down and cost logged.

### 10.4 Production checklist

**Prompt:** Write a production readiness checklist in `RUNBOOK.md`: SLOs defined and measured, alerts and dashboards, load and chaos tested, backup and retention, security (read-only role, secrets, rate limits, red-team results), rollback tested, runbooks for likely incidents. Mark each item met or not met with evidence links.
**Done when:** checklist complete with honest status.

### 10.5 Orchestrated retraining loop (stretch)

**Prompt:** Build a DAG: trigger (manual or drift alert webhook), sample frames for labeling or pseudo-labeling, fine-tune using the 9.5 script, run the 10.2 eval gate, register the candidate in the MLflow registry, optionally deploy. Choose the orchestrator (Prefect or Airflow) by operational weight on compose and k3s, DAG expressiveness and effort, and record D10.
**Done when:** a drift alert (or manual trigger) runs the DAG end to end on the CPU stack with tiny models; D10 accepted.
**Considerations:** stretch task, below polish in the cut lines and not scheduled in the week estimates. Keep it out of the `make demo` path.

---

## Phase 11: Packaging

**Goal:** make the work legible to a hiring manager in two minutes and to an engineer in twenty.
**Exit gate:** README, write-up and video published; fresh-clone reproduction verified.

### 11.1 README

**Prompt:** Rewrite `README.md`: one-paragraph pitch, headline numbers (each linked to its results and hardware), architecture diagram, optimization ladder summary, quickstart (`make setup && make up && make demo`), and links into the docs. Keep it short.
**Done when:** a reader can grasp the system and its results without opening other files.

### 11.2 Write-up

**Prompt:** Write `docs/WRITEUP.md`: problem and workload, architecture, each major decision with its evidence, the optimization journey from baseline to final with tables, what did not work and why, limitations (synthetic data, self-measured numbers, eval label biases), and future work.
**Done when:** every claim links to evidence; limitations section is candid.

### 11.3 Demo

**Prompt:** Script a 3-minute demo video: load ramp in Grafana, structured, semantic and hybrid queries, a drift alert firing, autoscaling. Optionally set up a cost-capped live demo (CPU stack with hosted LLM fallback, auto-shutdown) only if the user approves the cost.
**Done when:** video recorded; demo instructions in `RUNBOOK.md`.

### 11.4 Reproduction test

**Prompt:** On a clean machine or VM, follow only the README and RUNBOOK from a fresh clone. Fix every gap found.
**Done when:** quickstart works unaided; fixes committed.
