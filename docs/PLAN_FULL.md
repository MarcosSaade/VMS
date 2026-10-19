# Plan (full)

> Location in repo: `docs/PLAN_FULL.md`. Each task has a **prompt** to give the coding agent, **done when** criteria, and where relevant **considerations**. Each phase has an exit gate. `AGENTS.md` rules (tests, experiment log, docs, reproducibility) apply to every task and are not repeated here. If evidence shows a better path, change the plan and record why in `docs/DECISIONS.md`.

## Scope (2026-10-06 re-scope)

The plan was cut to a finishable core of about 6 part-time weeks. `PLAN.md` holds the scope lock, checkpoint C1, the order of work, an "Understand it" question per task and the interview-defense list. Here, kept tasks are unchanged unless marked **Status: shrunk**, and deferred tasks are marked **Status: deferred** with their original text kept for reference. Task IDs are never renumbered.

## Global considerations

- **Order matters.** The naive baseline (Phase 1) and its measurements (Phase 2) come before any optimization so every later improvement has a before/after number. Don't optimize earlier "while you're there"; log the idea in `EXPERIMENTS.md` instead.
- **Every decision should be defensible in an interview:** context, options, evidence, trade-offs. `DECISIONS.md` is the study guide.
- **Headline numbers** to work toward: rated capacity (cameras at 5 FPS, CPU and one GPU), freshness p95, structured-query p95, the precision ladder (latency against mAP), structured-query execution accuracy, drift detection delay at a stated false-alarm rate. Every quality number carries a bootstrap 95% confidence interval.
- **Honesty:** label synthetic data and self-measured results, report negative results, and never claim "production-ready" (the production checklist, task 10.4, is deferred).
- **Cost:** GPU work is batched into planned sessions on AWS (the default provider). Prepare scenarios and scripts on CPU first so rented time is spent measuring, not debugging.
- **Optional services stay optional:** AWS and DVC remotes are never required for `make setup && make up && make demo`.
- **Every tool earns its place:** a technology stays only if a kept task measures a need for it (D16).

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

**Status:** done; the `Tracker` part was removed in 0.7 (2026-10-06 re-scope, D16).
**Prompt:** Implement `vms.common`: settings via pydantic-settings that load `configs/<service>.yaml` with environment overrides; a JSON structured logger that includes trace and span IDs; an OpenTelemetry init function (OTLP exporter, no-op in tests) plus a Prometheus metrics helper enforcing the `vms_<component>_<metric>_<unit>` naming; ULID generation; an injectable `Clock` protocol with a real and a fake implementation; a `Tracker` protocol (start run, log params, log metrics, log artifact reference, end run, return run ID or URL) with MLflow, W&B and no-op implementations selected by config. W&B defaults to disabled or offline mode and its import is lazy, so nothing needs a W&B account. Write unit tests for each; Tracker tests use the no-op implementation and a recording fake, never a real backend.
**Done when:** tests pass; a sample service can log, trace and expose metrics with three lines of setup; switching tracker is a config change.
**Considerations:** roles are fixed by convention: W&B for training runs, sweeps and reports (LoRA, detector fine-tune); MLflow for eval results, model registry and champion tracking. Both may be enabled for one run; the results JSON records every tracker run ID or URL.

### 0.4 CI

**Prompt:** Add a GitHub Actions workflow running `make check` on push and pull request with uv caching. Add a placeholder integration job that will run against the compose stack later. Fail on any lint, type or test error.
**Done when:** CI is green on main and shows a failing run when a test is broken deliberately on a branch.

### 0.5 Compose stack

**Status:** done; MinIO became SeaweedFS (D12), and Redis and MLflow were removed in 0.7 (2026-10-06 re-scope, D16).

**Prompt:** Create `deploy/compose/` with Postgres (pgvector image), MinIO (with bucket bootstrap), Redis, OpenTelemetry collector, Prometheus, Tempo, Grafana (provisioned datasources) and MLflow (Postgres backend, MinIO artifacts). Add health checks. `make up` starts the stack and waits until healthy; `make down` stops it; `make logs` tails logs. Use a CPU profile by default and a `gpu` profile for later services.
**Done when:** `make up` reaches healthy on a laptop; Grafana shows Prometheus and Tempo datasources; MLflow UI loads.
**Considerations:** Kafka is not part of this stack; it is added in task 6.1 only if benchmarked. W&B is a hosted service and is not in compose.

### 0.6 Query assets (built from scratch)

**Prompt:** Build the query assets from scratch; the old smart-vision prototype is not used (user decision, 2026-10-06). Draft the `QueryPlan` Pydantic model in `vms.query` (finalized in 1.7). Author few-shot examples as (question, gold `QueryPlan`, gold SQL) triples covering every intent, time-range kind and abstain reason; questions the plan schema cannot express carry an `unsupported` plan plus their SQL, which also serves the free-form engine and result checks. Implement the SQL validator on `sqlglot` ASTs (single SELECT, table/column allowlist, function denylist, LIMIT injection) with thorough tests, including injection attempts, multiple statements, comments and DDL; it guards compiled SQL from the plan engine and generated SQL from the free-form engine. Write versioned prompt templates for the planner and the free-form SQL engine. Store few-shots and prompts as versioned data under `eval/` or `configs/`, not hardcoded strings, and reference them from config.
**Done when:** validator has comprehensive unit tests passing; few-shot triples validate against the `QueryPlan` schema and their SQL passes the validator; assets are in place and referenced from config.
**Considerations:** these few-shot examples must later be kept out of the eval test split (task 3.1). Few-shot camera and zone names (entrance, checkout, parking_lot, loading_dock) are illustrative; align the demo camera config (task 1.4) with them or update the bank. The validator's table allowlist in `configs/query.yaml` must match the migrations (task 1.4).

### 0.7 Trim the stack

**Prompt:** Remove what no kept task needs (2026-10-06 re-scope, D16): Redis and MLflow from the compose stack (services, volumes, MLflow image, the MLflow database init script and artifact bucket), the `Tracker` package with its MLflow and W&B backends and settings, the `tracking` dependency group and the Redis client, and every related variable, test, CI setting and doc reference. Results JSON files stay the source of truth for reported numbers.
**Done when:** `make up` reaches healthy without Redis and MLflow; `make check` and `make test-integration` pass; no reference to Redis, MLflow, W&B or the `Tracker` remains outside the history (`EXPERIMENTS.md`, `PLAN.md` revisions, decision records).

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
**Considerations:** Done 2026-10-09 (D17). Open-loop pacing with a per-camera in-flight cap and counted drops; schedule lag flags a saturated generator. Self-test 36.4k frames/s with no I/O (E-009); one process tops out near 1,250 frames/s over multipart HTTP (E-010); 20x5 against the stub was stable in 3 of 3 reps (E-011). The ingest contract adds `run_id`, so 1.3's idempotency key is `(camera_id, run_id, seq)`. Multipart parsing looked expensive in aiohttp (E-010); 1.3 should measure a raw-body variant.

### 1.3 Ingest v0

**Prompt:** Implement `vms.ingest` as a FastAPI service: accept multipart frames, write the JPEG to the object store (SeaweedFS locally, S3 in the cloud; D12) at `frames/{camera_id}/{date}/{frame_id}.jpg`, insert a `frames` row with status `pending`. Idempotency on `(camera_id, run_id, seq)` or `frame_id`. Health, readiness and metrics endpoints.
**Done when:** integration test sends frames and finds objects and rows; duplicate sends create no duplicates.
**Considerations:** Done 2026-10-09 (D18). Object first, then the row with `ON CONFLICT` on `(camera_id, run_id, seq)`; duplicates and races return the original `frame_id`; a failed insert deletes the object. The `frames` table and Alembic were set up here because ingest needs them; 1.4 adds the rest of the schema on top of migration 0001. Multipart parsing is kept as the baseline; whether a raw-body upload is worth it is measured in 2.6 (E-010 hints the multipart parser is costly).

### 1.4 Schema v1

**Status:** shrunk (2026-10-06 re-scope).
**Prompt:** Implement the logical schema from `ARCHITECTURE.md` section 4.6 in Postgres with Alembic migrations: cameras (with IANA timezones), zones (polygons), frames, detections, tracks, events and query_log. No `embeddings` table (CLIP is deferred). Load cameras and zones from YAML config, naming them to match the few-shot bank (entrance, checkout, parking_lot, loading_dock) or updating the bank. Create a read-only role for the query service. Keep indexes simple (this is the baseline), and keep the validator allowlist in `configs/query.yaml` in step with the migrations.
**Note:** Alembic and migration 0001 (`frames`) already exist from 1.3, with a `fresh_db` integration fixture; add the other tables as new revisions.
**Done when:** `alembic upgrade head` and downgrade work on a fresh DB; repository layer has tests; the few-shot SQL integration test runs against the migrated schema instead of temporary tables.
**Considerations:** Done 2026-10-12 (D19). Migrations 0002 (tables, baseline indexes) and 0003 (`vms_query_reader`: column-level SELECT on the allowlist, TEMPORARY revoked from PUBLIC). `make migrate` also runs `python -m vms.storage bootstrap`: the read-only login from `POSTGRES_READONLY_*` and the camera catalog from `configs/cameras.yaml` (cam-001..cam-004 named after the few-shot bank; zone polygons are placeholders to tune in 1.6). Detections are keyed `(frame_id, det_idx)` for idempotent redelivery in 1.5. No foreign keys on pipeline-written tables, so 1.5 must write detections and the frame status in one transaction. Repositories for detections, tracks and events arrive with their writers (1.5, 1.6).

### 1.5 Worker v0

**Status:** shrunk (2026-10-06 re-scope).
**Prompt:** Implement `vms.worker` v0: claim `pending` frames from Postgres with `SELECT ... FOR UPDATE SKIP LOCKED`, fetch the JPEG, run YOLO11 (start with `s`) in-process with PyTorch at batch size 1, write detections, compute input statistics (brightness, contrast, Laplacian variance) into `frames`, mark the frame done. Device configurable (CPU or CUDA). Record model versions. Use an injected model interface so unit tests run with fakes. No CLIP.
**Done when:** frames flow from loadgen to stored detections; unit tests use fakes; an integration test with a tiny model passes on CPU.
**Considerations:** polling Postgres is the queue (D3 default). It is the baseline every later number is measured against. Frames of one camera must be processed in order for tracking, so claim per camera.
**Status note:** Done 2026-10-12 (D20). Per-camera order is kept by camera leases (`camera_leases`, migration 0004) rather than claims alone, because tracker state lives in one worker. Completion is fenced on `claimed_by`. Weights are pinned in `configs/models.yaml` (`make models`); PyTorch is the CPU build. Model-only CPU latency is in E-013 (`make bench-model`). For 1.6: the worker counts late frames but processes them anyway; the tracker must skip or handle frames older than its state; `DetectionRow.track_id` is ready to fill.

### 1.6 Tracking and zone events

**Status:** shrunk (2026-10-06 re-scope).
**Prompt:** Add per-camera ByteTrack tracking in the worker (the tracker built into ultralytics or the one in `supervision`), write `track_id` on detections, maintain the `tracks` table, and emit zone enter and exit events from zone polygons. Loiter is optional: add it only if it costs under an hour. Test with synthetic trajectories.
**Done when:** tracks and events appear for demo clips; unit tests cover enter/exit edge cases (track lost and re-found, boundary jitter).
**Status note:** Done 2026-10-12 (D21). Ultralytics ByteTrack (added `lap`), bottom-centre anchors, 2-frame debounce, loiter included (30 s). Detector threshold lowered to 0.1 for ByteTrack's second round; detections under 0.25 are stored only when tracked. A time gap longer than the tracker buffer starts a new tracker session (bug found in the demo run, E-015). Zone polygons in `configs/cameras.yaml` are still placeholders: redraw them on frame overlays in 1.9.
**Considerations:** tracking requires in-order processing per camera; note it in `ARCHITECTURE.md`.

### 1.7 Query service v0

**Status:** shrunk (2026-10-06 re-scope).
**Prompt:** Implement `vms.query` with FastAPI per `ARCHITECTURE.md` section 4.7, structured mode only. Define the `StructuredQueryEngine` interface and implement the typed-plan engine: finalize the `QueryPlan` (D13); a planner prompt from `configs/prompts/planner.v1.txt` with the plan schema, a schema summary, camera timezones, rules and static few-shots from the bank; an OpenAI-compatible LLM client (configurable base URL so Ollama and hosted APIs work) with JSON-schema structured outputs; plan validation (schema plus semantic checks) with one repair attempt; abstention with reason; a deterministic Postgres compiler to parameterized SQL; the sqlglot validator on the compiled SQL; execution on the read-only role with statement timeout and row cap. Return one JSON response with the answer, result rows, result kind, plan JSON and compiled SQL. Log every query to `query_log` with plan, compiled SQL and stage timings. The `frames` intent lists matching frames with object-store URLs; `visual_query` is rejected as unsupported until CLIP returns.
**Done when:** unit tests with a fake LLM cover planning, invalid plans and repair, abstention, compilation (golden SQL per intent, parameters never inlined) and validator rejection; an integration test answers a handful of real questions against demo data.
**Considerations:** relative time ranges ("this morning") resolve in the camera timezone inside the compiler, not the LLM, so they are testable with the injected clock. No SSE, cache or circuit breaker in this plan.
**Status note:** Done 2026-10-12 (D22; D13 accepted). `make query` on port 8070; `python -m vms.query plan` prints plan and SQL without a database. The canned planner (`configs/demo/canned_plans.v1.jsonl`) runs when `LLM_MODEL` is empty; Ollama and hosted APIs via `LLM_*` (DeepSeek needs `json_object`, E-017). query_log gained stage timings, row count and model (migration 0005). For 1.8: `GET /v1/catalog`, `GET /v1/frames/{id}/image`, and the response shape in RUNBOOK (Query service); the live camera wall still needs an endpoint for the latest frame and its boxes. For 1.9: the canned plans file is where demo questions go.

### 1.8 Minimal frontend

**Status:** shrunk (2026-10-06 re-scope).
**Prompt:** Build a small Vite + TypeScript UI (React optional; no heavy component libraries), served by compose. Views: a camera wall with the latest frame per camera and its detection boxes, refreshed every second or two; a query box that shows the answer, a table or chart chosen from the result kind (scalar, table, time series, frames), and a collapsible panel with the plan JSON and compiled SQL; clear states for abstention, validator rejection and errors. Design it to look good in screenshots (consistent spacing, readable at 1280 px wide).
**Done when:** usable against the local stack during `make demo`; screenshots of both views saved for the README.
**Considerations:** keep it small: the UI exists to make the demo and write-up visual, not as a product. No feedback buttons and no semantic mode.
**Status note:** Done 2026-10-12 (D23). Vanilla TypeScript served by the query service at `/`; compose runs it with `make app` (profile `app`, image `deploy/docker/Dockerfile`). Screenshots of both views and of each result state were taken at 1280 px with `npm run screenshots` against a live pipeline (4 cameras, yolo11n) and handed to the user; they are not committed, because `AGENTS.md` forbids committing images and the frames are CC BY-NC-SA video: committing them (with a `docs/images/` exception) is the user's call before 11.1.

### 1.9 Demo

**Prompt:** Add `make demo`: start the stack and the UI, run loadgen with 4 cameras for a few minutes, then print answers to a few example structured queries (also visible in the UI). Add an integration test in CI that runs a shortened version on CPU with tiny models and a fake or tiny LLM (the fake returns canned plans), with no AWS or DVC remote access.
**Done when:** `make demo` works on a fresh clone with only Docker and uv installed; the CI integration job runs it.
**Zones (user request, 2026-10-12):** build a small interface that shows a frame from each demo camera and lets the user draw and edit zone polygons, writing them to `configs/cameras.yaml`; then ask the user to draw the zones (the current ones are placeholders). Check the result with `python -m vms.worker render`.
**Considerations:** the demo uses `make data-mini` (about 180 MB from motchallenge.net and Wikimedia Commons). Cache `data/` in CI keyed on the hash of `configs/data/sources.yaml`, so CI does not hit those hosts on every run.
**Status note:** Done 2026-10-12 (D24, E-020). `make demo` runs ingest, a yolo11n worker and the query service as containers of the application image, loadgen from the host, waits for the backlog to drain and prints the canned planner's answers (`ask --check` fails on errors). CI runs `make demo DEMO_SECONDS=30` after the integration tests, with `data/` and the weights cached. The zone editor is `make zones` (`vms.zones`); by the user's instruction the agent did not wait for zones to be drawn: the placeholders remain until the user draws them locally and checks them with `python -m vms.worker render`.

---

## Phase 2: Observability and baseline load test

**Goal:** see inside the system and produce the baseline numbers every optimization will be compared against.
**Exit gate:** dashboards provisioned; `BENCHMARKS.md` has a baseline section (CPU and one GPU type) with rated capacity, freshness, structured-query latency and a written bottleneck analysis.

### 2.1 Instrumentation

**Status:** shrunk (2026-10-06 re-scope).
**Prompt:** Instrument all services with OpenTelemetry so one frame's trace spans loadgen, ingest, worker, model inference and DB writes, and one query's trace spans every pipeline stage (plan, validate, compile, execute). Add metrics: ingest frames/s, rejected and dropped, processing backlog, end-to-end freshness histogram (frame timestamp to detections written), worker inference and write latency, query latency by stage, LLM tokens. GPU metrics (DCGM exporter) are added only on the GPU instance during 2.4 to 2.6.
**Done when:** a demo run shows complete traces in Tempo and all metrics in Prometheus; tests assert key metrics are emitted.

### 2.2 Dashboards

**Prompt:** Create Grafana dashboards as code in `deploy/grafana/`, auto-provisioned, and designed to be screenshotted for the README and write-up (clear titles, units, sensible ranges, no empty panels): Pipeline (ingest rate, backlog, freshness p50/p95, drops), Inference (latency, throughput, batch size, GPU utilization and memory when present), Query (latency by stage, errors, abstentions, tokens), Resources (CPU, memory per service) and Drift (placeholder panels filled in 9.4).
**Done when:** dashboards load automatically on `make up`, display live data during the demo, and screenshots of each are saved for the write-up.

### 2.3 Benchmark framework

**Prompt:** Implement `vms.bench`: scenarios in `bench/scenarios/*.yaml` describing ingest load (cameras, FPS, ramp steps, hold time), query load (mix, concurrency, duration) and warmup. The runner orchestrates loadgen and a query load generator (k6 or Locust), scrapes Prometheus over the steady-state window, and writes a results JSON with the fields required by `AGENTS.md` section 7. Add a command that renders results into Markdown tables for `BENCHMARKS.md`. Wire `make bench SCENARIO=...`.
**Done when:** a small scenario runs locally and produces a valid results file; the result schema has a test.
**Considerations:** check that the load generators are not the bottleneck; record their CPU.

### 2.4 GPU bench environment

**Status:** shrunk (2026-10-06 re-scope).
**Prompt:** AWS is the default provider. Write `scripts/gpu_session.sh` (AWS CLI, no Terraform) that, in the project's single configured Region: launches one EC2 spot GPU instance (g5 or g6) from a GPU-ready AMI with Docker and the NVIDIA toolkit, tagged `project=vms` and `session=<id>` (instance, volumes, security group); syncs the code; runs named benchmark scenarios with docker compose; copies results back; and terminates everything, with teardown in a trap so it runs on failure. After teardown it lists resources still tagged with the session ID through the Resource Groups Tagging API and exits non-zero, loudly, if any remain. Create an AWS Budgets alarm for the project once. State the expected cost and ask the user before the first real launch.
**Done when:** one real short session completes with automatic teardown verified by the tag check; cost recorded in the results; `RUNBOOK.md` GPU sessions section written.
**As built (2026-10-13, D28):** spot launches are denied by the account's managed SCP, so sessions are on-demand (g6.xlarge); the driver is `vms.cloud` in Python (boto3) behind `scripts/gpu_session.sh`, for testable teardown. Everything except the real session is done; the session waits for the on-demand G and VT quota (L-DB2E81BA, 4 vCPUs requested). Then: `make gpu-preflight`, `make gpu-session` (gpu-smoke and worker-model-gpu-n), log the run, write the cost into BENCHMARKS.
**Rebuilt on Runpod (2026-10-19, D37):** AWS denied the GPU quota for EC2 and SageMaker (D35), Lambda Cloud declined the card and DigitalOcean wanted a ~$250 prepayment, so GPU sessions run on Runpod pods: `scripts/runpod_session.sh` (`vms.cloud.pod_session`, REST v2) creates one pod (`vms-gpu-<id>`, 1x A40 secure, base image by digest, per-session SSH key, no volume), runs `kind: model` and `ladder` scenarios through `scripts/pod_remote.sh` (locked environment with CUDA torch, no Docker), copies results back, deletes the pod and verifies by name and recorded id; the pod deletes itself after 150 minutes. Done when: `make runpod-preflight`, then one real `make runpod-session` (worker-model-gpu-n) with `teardown_verified` true, the time limit armed, the cost logged in EXPERIMENTS and BENCHMARKS, and the RUNBOOK section checked against what happened. The EC2 path stays as built for the CPU sessions.
**Considerations:** before the first launch, confirm the account plan allows GPU instances and that the on-demand vCPU quota for G instances in the project Region is at least the instance's vCPUs (`make gpu-preflight` checks both). If the quota never arrives, Phase 5 and 9.5 use their CPU fallbacks. No S3 bucket, ECR, IAM roles, k3s or DVC S3 remote in this plan.

### 2.5 Baseline load tests

**Status:** shrunk (2026-10-06 re-scope).
**Prompt:** Write scenarios for (a) an ingest ramp: step cameras through 1, 2, 4, 8, 16, ... at 5 FPS, hold each step long enough for steady state, and record freshness p95, backlog growth and drops to find rated capacity; (b) a small structured-query load (a few concurrency levels, fake or local LLM so cost stays zero) while ingest runs at about 70% of rated capacity. Run on CPU locally and once on the GPU instance.
**Done when:** results files exist for both environments; rated capacity determined per the definition in `ARCHITECTURE.md` section 2, with variance across repetitions.
**Also on that session (2026-10-14 revision):** the 3.5 second-LLM dev run on a local model (see 3.5).
**CPU part as built (2026-10-14, E-039, E-040):** `bench/scenarios/ramp-cpu.yaml` (1 to 4 cameras: the CPU edge is between 2 and 3, so powers of two would step over it) and `query-load-cpu.yaml` (ingest at 70% of rated as 2 cameras x 3.5 FPS, concurrency 1 to 8, canned planner). Rated capacity on CPU: 2 cameras. GPU part deferred (2026-10-19, D37): a Runpod pod has no Docker, so the GPU ramp needs the stack on a GPU host another way (Deferred list); 2.5 ticks on its CPU part.

### 2.6 Baseline report

**Prompt:** Profile the worker and query service under load (py-spy flame graphs, trace breakdowns, GPU utilization). Write the baseline section of `BENCHMARKS.md`: tables, hardware, methodology, and a ranked list of bottlenecks with evidence. Use the list to check whether Phase 5 targets the real bottleneck, and propose a swap under the scope lock if it does not.
**Done when:** baseline section complete; bottleneck list links to traces or profiles saved as artifacts.
**CPU part as built (2026-10-14, E-041, E-042):** `kind: profile` scenarios (`vms.bench.profile`, py-spy on the host with the container stopped) for the worker at saturation and the query service; the ranked list is in BENCHMARKS ("Baseline bottlenecks"). Phase 5 targets the top bottleneck (the forward pass); the second (serial per-frame I/O) fits inside 5.5, so no scope swap. GPU part deferred with the GPU ramp (2026-10-19, D37); 2.6 ticks on its CPU part.

---

## Phase 3: Evaluation harness

**Goal:** trustworthy quality measurement, so speed work can't silently hurt quality and LLM work has a real target.
**Exit gate:** `make eval` reports structured-query execution accuracy (dev and test, over all and over plan-expressible questions, plus plan coverage) and detection mAP for the baseline, each with a bootstrap 95% confidence interval, written to results JSON and recorded in `BENCHMARKS.md`.

### 3.1 Structured-query dataset

**Status:** shrunk (2026-10-06 re-scope).
**Prompt:** Build a frozen eval DB snapshot generated deterministically (seeded replay of demo clips plus synthetic additions) so gold answers never change; track the snapshot with DVC and keep the generator script and seed in git. Draft about 120 questions across categories: count, aggregate, time window (with camera timezones), joins across tracks/events/zones, trend, top-k, dwell, ambiguous, unanswerable, unsupported (a valid analytic question outside the plan schema) and adversarial (refuse), with some paraphrases. Each entry stores gold SQL, the expected result set from executing it, a gold `QueryPlan` where expressible, and an `expressible_by_plan` flag; check that each gold plan compiles and returns the expected result set. Use a strong model to draft candidates. Split 50/50 dev/test stratified by category; record the test file hash. Store as JSONL under `eval/structured/`.
**Done when:** dataset passes schema validation; every gold SQL and every gold plan executes and matches the expected result set; the user has reviewed all of the test split and a sample of dev; test hash recorded in `BENCHMARKS.md`.
**Considerations:** about 60 test questions give a 95% interval of roughly plus or minus 12 points at 70% accuracy; state this in the write-up rather than over-reading small differences. Few-shot examples must not overlap the test set (exact or near duplicates; reuse `normalize_question`). Keep `unsupported` honest: questions a user could plausibly ask.

### 3.2 Scorer and taxonomy

**Prompt:** Implement the structured eval harness: run each question through any `StructuredQueryEngine` (selected by config), compare result sets with the stored expected results (order-insensitive unless the question implies order, numeric tolerance, column-name agnostic), and classify failures into a documented error taxonomy. Plan error classes: invalid plan, plan failed validation, wrong filter, wrong aggregation, wrong group_by, wrong time range. Shared: should have abstained, abstained wrongly, wrong abstain reason. Report accuracy over all questions and over plan-expressible ones, plan coverage, abstention correctness and accuracy per category, each with a bootstrap 95% confidence interval (resampling questions, fixed seed, at least 2,000 resamples), plus latency, tokens and cost. For every metric, write one line on what it measures and what it does not (for example, execution accuracy ignores whether the answer was phrased well and can pass by coincidence on small result sets). Include a red-team subset (DDL, injection, exfiltration, resource exhaustion) and report attack success rate. Write a results JSON per run. Parallelize with a concurrency limit.
**Done when:** `make eval SUITE=structured SPLIT=dev` runs; scorer and bootstrap edge cases are unit tested; the taxonomy and metric notes are in `BENCHMARKS.md`; red-team results reported.

### 3.3 Retrieval eval

**Status:** deferred (2026-10-06 re-scope; not planned). Kept unchanged for reference; it returns only through the scope-lock rule in `PLAN.md`.

**Prompt:** Create 60 to 100 natural-language visual queries relevant to the clips (e.g., people carrying bags, cyclists, crowded scenes, vehicles at night). Build a small labeling CLI that pools candidates (CLIP top-50 plus random frames) and lets the user mark relevant frames quickly. Compute recall@10, nDCG@10 and MRR. Store labels under `eval/retrieval/`.
**Done when:** labels exist for all queries; `make eval SUITE=retrieval` reports metrics.
**Considerations:** pooling from CLIP biases the labels toward CLIP; adding random and detection-filtered candidates reduces this. Note the limitation in the write-up.

### 3.4 Detection quality

**Prompt:** Fix a labeled evaluation subset for detection (about 500 images from COCO val2017 for all classes, plus MOT17 ground truth for person boxes on frames disjoint from anything used for calibration or fine-tuning) with a frozen file list. Implement mAP@50 and mAP@50-95 for any inference backend behind the worker's model interface, with bootstrap 95% confidence intervals over images.
**Done when:** `make eval SUITE=detection` reports baseline mAP with CIs for PyTorch YOLO11.
**Considerations (as built, D31):** 500 COCO images give an interval about 5.5 points wide on mAP@50-95; compare two backends with a paired bootstrap over the same resamples (the saved per-run predictions make this cheap) rather than by overlapping intervals. The MOT17 frames are correlated within a sequence, so their interval is optimistic. Calibration (5.2) and fine-tuning (9.5) data must come from outside `eval/detection/subset.v1.jsonl` (COCO train2017, MOT17-05/10/11/13).

### 3.5 Baseline eval

**Prompt:** Run the structured and detection suites on the baseline system. For the structured suite, compare two LLMs on dev (for example a local model through Ollama and one hosted model) and pick the default for D7; run test once for the chosen one. Record results in `BENCHMARKS.md` and `EXPERIMENTS.md` (tag `query-eval`), and add a CI job that runs the harness on a tiny subset with a fake LLM to keep it from breaking.
**Done when:** baseline quality numbers published; CI harness job green.
**Revision (2026-10-14):** the two-LLM comparison is moved to the first GPU session: serve a local open model there (e.g. qwen2.5:7b through Ollama, or vLLM, with at least an 8k context for the ~4.7k-token prompt) and run `make eval SPLIT=dev` against it with the same harness, then compare with the DeepSeek dev run and settle D7. 3.5 is done without it: DeepSeek is the provisional planner and its test split is run once. If the local model wins on dev, it gets its own single test run.

---

## Phase 4: Storage

**Status:** whole phase deferred (2026-10-06 re-scope). D1 and D2 have defaults.

**Goal:** choose the detection and vector stores with evidence (D1, D2).
**Exit gate:** D1 and D2 accepted in `DECISIONS.md` with linked evidence; system migrated; eval re-run with no unexplained regression.

### 4.1 Synthetic detection generator

**Status:** deferred (2026-10-06 re-scope; not planned). Kept unchanged for reference; it returns only through the scope-lock rule in `PLAN.md`.

**Prompt:** Fit distributions from real Phase 1 replay data (class mix per camera and hour, detections per frame, track lengths, event rates, diurnal pattern) and implement a deterministic, seeded generator that writes Parquet chunks at 10M, 100M and optionally 1B detection rows with matching frames, tracks and events.
**Done when:** generated data matches fitted distributions within tolerance (tested); generation is reproducible by seed; generated Parquet is tracked with DVC (or regenerated by seed, with the choice stated).

### 4.2 Storage workload

**Status:** deferred (2026-10-06 re-scope; not planned). Kept unchanged for reference; it returns only through the scope-lock rule in `PLAN.md`.

**Prompt:** Define the storage benchmark: (a) sustained ingest at several batch sizes and concurrencies, (b) a query suite of about 20 representative queries derived from the structured eval categories (gold plans and SQL) (point lookup by frame, time-range scan, aggregation by class and hour, top-k, joins to tracks and events, dwell time), (c) mixed reads during ingest. Specify cold and warm runs and how caches are dropped.
**Done when:** workload defined as scenarios with a runner that works against any candidate store through an adapter interface.

### 4.3 Detection store benchmark

**Status:** deferred (2026-10-06 re-scope; not planned). Kept unchanged for reference; it returns only through the scope-lock rule in `PLAN.md`.

**Prompt:** Benchmark three candidates on the same machine with equal resource limits and a fair, documented tuning effort for each: Postgres 16 with daily partitioning, BRIN on time and B-tree on (camera_id, ts); TimescaleDB hypertables with compression and continuous aggregates; ClickHouse MergeTree with partition by day, a sort key such as (camera_id, class_name, ts), LowCardinality columns and suitable codecs. Measure ingest rows/s, query p50/p95 per query class (cold and warm), bytes per row, CPU and memory, at 10M and 100M rows (1B on a rented machine only if the user approves the cost).
**Done when:** results files and a comparison table in `BENCHMARKS.md`; per-store tuning documented.
**Considerations:** column stores handle updates poorly, so mutable track state may stay in Postgres (a hybrid is a valid outcome). Make sure the harness, not the database, isn't the bottleneck.

### 4.4 Vector store benchmark

**Status:** deferred (2026-10-06 re-scope; not planned). Kept unchanged for reference; it returns only through the scope-lock rule in `PLAN.md`.

**Prompt:** Benchmark pgvector (HNSW, sweeping m, ef_construction, ef_search) against Qdrant (HNSW with payload indexes) at 1M and 10M 512-dimensional vectors built from real CLIP embeddings (augment if needed and say so). Measure recall@10 against exact search, p95 latency and QPS, and repeat with camera and time filters at several selectivities. Measure index build time and memory.
**Done when:** recall vs latency curves and the filtered-search comparison are in `BENCHMARKS.md`.
**Considerations:** filtered ANN behavior is the interesting part for hybrid search; highly selective filters can collapse HNSW recall in some engines.

### 4.5 Decide and migrate

**Status:** deferred (2026-10-06 re-scope; not planned). Kept unchanged for reference; it returns only through the scope-lock rule in `PLAN.md`.

**Prompt:** Write D1 and D2 with evidence, including operational complexity and compiler complexity: each viable detection store needs a plan-to-SQL compiler, with equivalence tests showing the same plan returns the same result set on every backend (run the structured dev split's gold plans through each compiler). Model accuracy no longer depends on dialect, but report the free-form SQL path's per-dialect gold SQL status where it matters for D11. Migrate the system to the winners, update schema, migrations, compilers and few-shots, and re-run all eval suites and the ramp test.
**Done when:** decisions accepted; compiler equivalence tests green for each viable backend; integration tests green; eval and ramp results compared with baseline in `BENCHMARKS.md`.

---

## Phase 5: Inference optimization

**Goal:** a measured precision ladder for YOLO11 (accuracy against latency), ending with the worker on the best backend in-process (D5). No Triton in this plan.
**Exit gate:** ladder table (latency, throughput, GPU memory, mAP with CI) in `BENCHMARKS.md`; ramp re-run shows the end-to-end effect.

### 5.1 ONNX export and parity

**Status:** shrunk (2026-10-06 re-scope).
**Prompt:** Write an export script for YOLO11 to ONNX with dynamic batch; weights are fetched by script. Add parity tests comparing ONNX Runtime outputs with PyTorch on fixed inputs (boxes and scores after NMS, and raw head outputs) within a tolerance you justify. No CLIP export and no Triton repository.
**Done when:** parity tests pass on CPU; tolerance and observed max differences recorded in `EXPERIMENTS.md`.
**As built (2026-10-14, E-037):** `vms.worker.onnx_detector` (export, `OnnxDetector` with numpy pre/post-processing and `detect_batch`), `detector.backend` and the `build_detector` factory, `vms.worker.parity` and `make onnx-parity` (eval subset + data-mini, results JSON). Unit tests use a seeded random-init YOLO11n (offline); the integration test uses the pinned weights. Parity also holds at the mAP level for yolo11n and yolo11s.

### 5.2 Precision ladder

**Prompt:** On one rented GPU type, measure each step for YOLO11: PyTorch eager, ONNX Runtime CUDA, TensorRT FP16, TensorRT INT8 (calibrate on a representative frame set that is disjoint from the eval sets). Measure latency at batch 1 and throughput at a few batch sizes with a small in-process harness, plus GPU memory, and run detection mAP (3.4) with CIs for every step. Build TensorRT engines on the target GPU by script. CPU fallback if no GPU is available: PyTorch, ORT FP32 and ORT INT8 (static quantization) on CPU.
**Done when:** ladder table complete with mAP and CIs; INT8 kept only if its mAP drop is within a threshold stated in `DECISIONS.md` beforehand.
**GPU (2026-10-19, D37, D38):** the GPU rungs run on a Runpod pod (1x A40) as `bench/scenarios/ladder-gpu-n.yaml` through `make runpod-session`, with mAP for every rung and the paired comparisons in the same session (the ladder's `quality:` section); `bench/scenarios/sweep-gpu-trt-{fp16,int8}-n.yaml` are the 5.3 GPU sweep. TensorRT 11 has no FP16/INT8 builder flags (strongly typed networks only), so FP16 comes from an FP16 ONNX export and INT8 from a symmetric Q/DQ model (D38).

### 5.3 Batching sweep

**Status:** shrunk (2026-10-06 re-scope).
**Prompt:** In the same harness, sweep batch size (1, 2, 4, 8, 16, 32) for the best backend from 5.2 and plot throughput against latency. Choose an operating point that meets the freshness SLO with headroom.
**Done when:** plot saved and summarized in `BENCHMARKS.md`; chosen batch size committed in config.
**CPU part as built (2026-10-19, E-057, E-059):** `bench/scenarios/sweep-cpu-ort-n.yaml` (the 5.2 ladder harness, one step, batches 1 to 32) and `python -m vms.worker sweep-plot` (operating point: smallest batch within 5% of the best throughput with p95 under 25% of the SLO; `docs/images/batch-sweep-cpu.png`). Laptop picks 4 (the committed `infer_batch`), AWS c7i.2xlarge picks 8. The GPU sweep runs on Runpod (D37).

### 5.4 Pre- and post-processing

**Status:** deferred (2026-10-06 re-scope; not planned). Kept unchanged for reference; it returns only through the scope-lock rule in `PLAN.md`.

**Prompt:** Profile the full worker path under load. If decode, resize or NMS dominate, move them to the GPU (Triton ensemble, DALI, TensorRT NMS plugin, or a small custom C++/CUDA implementation as a custom Triton backend or TensorRT plugin) and measure end to end. Attempt the custom C++/CUDA option only if profiling shows preprocessing dominates, and benchmark it against the DALI path. If they don't dominate, record that and stop.
**Done when:** decision and measurements logged; end-to-end worker throughput recorded.

### 5.5 Integrate and decide D5

**Status:** shrunk (2026-10-06 re-scope).
**Prompt:** Switch the worker to the best backend in-process (ONNX Runtime with the TensorRT or CUDA provider, or the CPU fallback), batching frames across cameras up to the batch size from 5.3 while keeping per-camera order for tracking. Re-run the ingest ramp on the same hardware as the baseline and report rated capacity before vs after. Write D5.
**Done when:** D5 accepted; `BENCHMARKS.md` shows model-only and end-to-end improvement separately.
**Considerations:** the end-to-end number (cameras per GPU) is the headline; model-only speedups are supporting detail.
**GPU part (2026-10-19, D37):** the GPU ramp is deferred (a Runpod pod has no Docker for the stack), so on GPU only model-only numbers exist; D5's GPU part is decided from 5.2 and 5.3 on the pod and labeled model-only, and the README says cameras per GPU was not measured.

---

## Phase 6: Scaling the pipeline

**Status:** whole phase deferred (2026-10-06 re-scope). D3, D4, D6 and D8 have defaults. Idempotent ingest stays part of 1.3.

**Goal:** a horizontally scalable, failure-tolerant pipeline on Kubernetes with autoscaling, with transport and queue choices justified (D3, D4, D6, D8).
**Exit gate:** rated capacity measured on the GPU cluster; chaos tests show no frame loss below capacity; autoscaling demonstrated; decisions accepted.

### 6.1 Queue

**Status:** deferred (2026-10-06 re-scope; not planned). Kept unchanged for reference; it returns only through the scope-lock rule in `PLAN.md`.

**Prompt:** Replace DB polling with a queue carrying frame refs, keyed or partitioned by camera so per-camera ordering is preserved for tracking. Add Apache Kafka (KRaft mode, single broker) to compose under a profile, and benchmark Redis Streams against Kafka for the realistic message size and rate, ordering guarantees, consumer rebalancing behavior, replay and operational cost. Implement the chosen one with consumer groups, ack-after-write and a dead-letter stream. Also evaluate whether detections and events should flow through a Kafka topic that feeds the detection store (for example the ClickHouse Kafka engine or a consumer) and the drift monitor, for decoupling and replay: measure write throughput and freshness against direct writes and weigh the added operations. Decide both inside D3; if Redis Streams wins the frame queue, still record whether the event stream benefits from Kafka.
**Done when:** D3 accepted with numbers for both questions; ramp test shows the effect vs polling.
**Considerations:** how tracker state moves when a camera is reassigned to another worker during rebalancing must be handled explicitly (e.g., reset with a logged discontinuity, or state handoff). Do not adopt Kafka without benchmark evidence; "Redis Streams suffices at this scale" is a valid outcome, and if Kafka is not adopted it is removed from compose.

### 6.2 gRPC ingest

**Status:** deferred (2026-10-06 re-scope; not planned). Kept unchanged for reference; it returns only through the scope-lock rule in `PLAN.md`.

**Prompt:** Define `proto/ingest.proto` per `ARCHITECTURE.md` section 6, generate code via a Makefile target, implement client-streaming ingest in the gateway and loadgen, and benchmark against REST multipart: frames/s per core, p99 latency, CPU per frame. Keep the public API on REST. Write D4.
**Done when:** D4 accepted with numbers; both transports tested.

### 6.3 Reliability

**Status:** deferred (2026-10-06 re-scope; not planned). Kept unchanged for reference; it returns only through the scope-lock rule in `PLAN.md`.

**Prompt:** Implement and test: idempotent writes for redelivered messages, backpressure at ingest when queue lag passes a threshold (gRPC `RESOURCE_EXHAUSTED`), configurable shedding policy above capacity, dead-letter handling for poison frames, graceful shutdown that drains in-flight batches, timeouts and bounded retries everywhere. Add fault-injection integration tests (kill a worker mid-batch, DB unavailable briefly, object store slow).
**Done when:** fault tests pass and show zero loss and zero duplicates below capacity.

### 6.4 Helm and kind

**Status:** deferred (2026-10-06 re-scope; not planned). Kept unchanged for reference; it returns only through the scope-lock rule in `PLAN.md`.

**Prompt:** Create one Helm chart under `deploy/helm/` with per-service values, resource requests and limits, GPU requests for Triton, probes, config maps and secrets wiring. Add `make k8s-smoke` that creates a kind cluster, installs the chart in the CPU profile, runs a short demo and tears down; run it in CI.
**Done when:** CI kind smoke test green; chart passes `helm lint`.

### 6.5 Autoscaling

**Status:** deferred (2026-10-06 re-scope; not planned). Kept unchanged for reference; it returns only through the scope-lock rule in `PLAN.md`.

**Prompt:** Add a KEDA ScaledObject scaling workers on queue lag, and an HPA for the query service on CPU or request rate. Under a ramp scenario on the GPU cluster, record time to scale up, lag recovery time and scale-down behavior. Consider how worker count relates to Triton capacity (GPUs are the hard limit) and document the policy.
**Done when:** autoscaling plots and numbers in `BENCHMARKS.md`.

### 6.6 Capacity and chaos on the GPU cluster

**Status:** deferred (2026-10-06 re-scope; not planned). Kept unchanged for reference; it returns only through the scope-lock rule in `PLAN.md`.

**Prompt:** Run the full ramp and mixed query load on the GPU cluster to measure rated capacity and SLOs. Run chaos tests: kill worker pods, restart Triton, inject DB latency, build a backlog and measure drain rate. Using measured CPU, GPU and memory profiles per component, write D6 on which components deploy separately and why (merging is a valid outcome).
**Done when:** D6 accepted; chaos results in `BENCHMARKS.md`.

### 6.7 Embedding sampling

**Status:** deferred (2026-10-06 re-scope; not planned). Kept unchanged for reference; it returns only through the scope-lock rule in `PLAN.md`.

**Prompt:** Compare embedding every frame, a fixed rate (e.g., 1 FPS) and scene-change triggering (frame difference or perceptual hash threshold): retrieval recall@10 (3.3), GPU time and storage. Write D8 and apply the chosen policy.
**Done when:** D8 accepted with the trade-off table.

---

## Phase 7: Search quality

**Status:** whole phase deferred (2026-10-06 re-scope). D11 has a default; error analysis lives in 3.2 and 3.5.

**Goal:** a strong hybrid search mode, measurable structured-query gains, and an evidence-based choice of structured query interface (D11).
**Exit gate:** all eval suites re-run on test with improvements attributed to specific changes; D11 accepted.

### 7.1 Hybrid search

**Status:** deferred (2026-10-06 re-scope; not planned). Kept unchanged for reference; it returns only through the scope-lock rule in `PLAN.md`.

**Prompt:** Implement hybrid queries on the shared `QueryPlan`: the same planner emits a plan with `visual_query` set (no separate hybrid schema); execute filtered ANN, or a compiled SQL prefilter followed by vector rerank, choosing by estimated selectivity with thresholds measured on the vector benchmark from 4.4. Add 40+ labeled hybrid queries to the eval and report metrics.
**Done when:** hybrid mode in API and frontend; hybrid eval results recorded.

### 7.2 Query engine improvements and D11

**Status:** deferred (2026-10-06 re-scope; not planned). Kept unchanged for reference; it returns only through the scope-lock rule in `PLAN.md`.

**Prompt:** On the dev split only, ablate each change to the plan engine independently: few-shot retrieval from an example bank by embedding similarity, schema and plan pruning (only relevant tables, enums and plan fields in the prompt), repair variants, and a result cache keyed by normalized question plus data watermark. Implement the free-form SQL engine behind the `StructuredQueryEngine` interface (pruned schema DDL prompt, sqlglot validation, one self-correction attempt). Optionally prototype an agent loop (tool calling over the plan executor and CLIP search). Compare engines on the dev split by execution accuracy, abstention correctness, p95 latency, tokens and cost, expressiveness coverage (share of questions expressible by the plan schema) and red-team attack success rate, then run the chosen configurations once on test. Write D11.
**Done when:** ablation table and engine comparison in `BENCHMARKS.md`; cache hit rate and latency effect measured; D11 accepted.
**Considerations:** the agent loop is strictly optional. The typed plan stays the default until D11 is accepted; a hybrid outcome (plan by default, SQL fallback for `unsupported`) is valid if the evidence supports it and the red-team rate stays acceptable.

### 7.3 Error analysis

**Status:** deferred (2026-10-06 re-scope; not planned). Kept unchanged for reference; it returns only through the scope-lock rule in `PLAN.md`.

**Prompt:** Re-run all suites, then analyze remaining failures by category with examples. Turn real failures from `query_log` feedback into new dev examples (never test).
**Done when:** error analysis section in `BENCHMARKS.md`.

---

## Phase 8: LLM serving and LoRA

**Status:** planned after checkpoint C1 (2026-10-16, user's decision; was deferred in the 2026-10-06 re-scope). C1's scope is unchanged. 8.1 may run early on the first GPU session, next to the 3.5 local LLM run. D7 keeps its default until 8.5.

**Goal:** efficient self-hosted LLM serving and an evidence-based choice between prompting, fine-tuning and hosted models (D7).
**Exit gate:** D7 accepted with accuracy, latency and cost per 1,000 queries for each option.

### 8.1 vLLM serving

**Status:** planned after C1 (2026-10-16); moved before publishing (2026-10-19). Changed: Helm dropped (6.4 is deferred). Runs on a Runpod A40 session (D37); the 3.5 local-LLM dev comparison runs on the same served model. Needs the pod driver to allow an LLM-serving scenario kind (it runs `model` and `ladder` only today).

**Prompt:** Serve candidate small open models with vLLM behind the OpenAI-compatible interface; check which current small models are strongest at structured JSON output and SQL before choosing candidates. Enable prefix caching (the schema prompt is long and shared) and measure its effect. Measure constrained-decoding overhead: structured outputs on vs off for `QueryPlan` generation (tokens/s, time to first token, p95, and plan validity rate). Load test concurrent requests for tokens/s, time to first token and p95, and compare with llama.cpp or Ollama on the same GPU. Add the llm-service to compose (GPU profile).
**Done when:** serving comparison in `BENCHMARKS.md`; query service runs against vLLM.

### 8.2 Training data

**Status:** planned after C1 (2026-10-16); moved before publishing (2026-10-19). CPU only (a hosted LLM drafts pairs), no GPU session needed.

**Prompt:** Generate training (question, `QueryPlan`) pairs with a strong model over the schema and varied synthetic DB states. Filter: the plan is schema-valid, compiles successfully, executes, and the result sets of two independent generations agree. Deduplicate and run leakage checks against the test set: exact match, near-duplicate by embedding similarity and template overlap. Target 2k to 10k pairs; record generation cost and filter pass rates. If a SQL adapter is planned for D11 (8.3), keep the compiled or independently generated SQL for each pair as well.
**Done when:** dataset tracked with DVC (generation script, seed and filter config in git); leakage report shows no overlap.

### 8.3 LoRA fine-tune

**Status:** planned after C1 (2026-10-16); moved before publishing (2026-10-19), on a Runpod A40 (48 GB fits a 7B model in 16-bit plus LoRA). Changed: W&B and the MLflow registry are gone (D16); runs write results JSON, adapters go to DVC.

**Prompt:** Fine-tune the chosen base with LoRA (PEFT or Unsloth) on a rented GPU via the session script, with plan JSON as the target. Each training run and eval writes a results JSON (D16). Ablate data size (e.g., 500, 2k, all) and rank on the dev split; evaluate the best once on test. Optionally train a second adapter with SQL targets on the same questions for the D11 comparison.
**Done when:** ablation table and final test numbers recorded; adapter weights tracked with DVC; the chosen adapter pinned by hash in config.

### 8.4 Quantization and multi-LoRA

**Status:** planned after C1 (2026-10-16).

**Prompt:** Quantize the base model (AWQ or GPTQ) and measure the accuracy and latency effect. If both a plan adapter and a SQL adapter exist, serve them on one base with vLLM multi-LoRA and measure overhead vs merged single models. If only the plan adapter exists (hybrid shares the plan, so there is no separate hybrid adapter), measure adapter vs merged serving overhead and state that multi-LoRA was not needed.
**Done when:** results recorded; serving config committed.

### 8.5 Decide D7

**Status:** planned after C1 (2026-10-16).

**Prompt:** Compare hosted model, base model few-shot, LoRA, and quantized LoRA on structured-query execution accuracy (test), p95 latency under load and cost per 1,000 queries (including amortized GPU cost). Write D7 honestly, including the case where fine-tuning loses.
**Done when:** D7 accepted; default model in config matches the decision.

---

## Phase 9: Drift detection

**Goal:** demonstrate, as sequential change detection, that the system notices when a camera's inputs or outputs change: per-camera statistics, injected drift with known onsets, and a measured trade-off between detection delay and false-alarm rate; plus a small before/after experiment showing the detector can be recovered on drifted data.
**Exit gate:** drift detection results table in `BENCHMARKS.md`; alerts and runbook procedure in place; drift fine-tune results table in `BENCHMARKS.md`.

### 9.1 Drift statistics

**Prompt:** Implement `vms.drift`: per camera, compute streaming statistics over fixed windows and compare them with a reference window from a calibration period. Signals: input statistics (brightness, contrast, blur, noise estimate), detections per frame, mean confidence and the confidence distribution (KS or Wasserstein against the reference), and class-mix PSI. No embedding drift (CLIP is deferred). Export gauges to Prometheus. Unit test each statistic on synthetic distributions with known shifts.
**Done when:** statistics tested; gauges visible during a demo run.

### 9.2 Drift injection

**Prompt:** Add scheduled, seeded transforms to loadgen: gradual and abrupt darkening, blur, noise, camera shift, occlusion and class-mix change (switching clips). Log onset times in the results. Measure detection mAP (with CIs) under each photometric transform on the labeled subset so the impact of each drift is quantified.
**Done when:** scenarios exist for each drift type; mAP impact table recorded.

### 9.3 Alerts and scoring

**Prompt:** Frame alerting as sequential change detection. Implement two detectors on the 9.1 statistics: a fixed threshold on the windowed statistic (baseline) and CUSUM (or Page-Hinkley) on the standardized statistic. Calibrate each on clean runs, then sweep its sensitivity (threshold, or CUSUM drift and decision parameters) and, across seeded drift scenarios and clean runs, measure detection delay after the known onset and false alarms per camera-hour. Plot delay against false-alarm rate for both detectors, with spread across seeds, and state which signals catch which drift. Add Prometheus alert rules for the chosen operating point.
**Done when:** delay vs false-alarm curves and a results table in `BENCHMARKS.md`; the chosen operating point is justified; detector logic is unit tested on synthetic streams with known change points.

### 9.4 Dashboard and runbook

**Prompt:** Fill the Drift dashboard from 2.2 (per-camera statistics against their reference bands, CUSUM state, alerts with onset markers during injected scenarios, designed to be screenshotted) and write a `RUNBOOK.md` procedure: what each alert means, how to confirm it, and the response options (recalibrate the reference, flag the camera, route frames for labeling, retrain).
**Done when:** dashboard provisioned; runbook section written.

### 9.5 Detector fine-tune on drifted data

**Status:** shrunk (2026-10-06 re-scope).
**Prompt:** Using the mAP-under-drift table from 9.2, take the worst photometric drift (darkening, blur or noise). Photometric transforms keep bounding boxes valid, so apply them to a labeled set disjoint from the eval subsets. Fine-tune YOLO11n for a few epochs on a clean plus drifted mix with one fixed configuration. Report mAP before and after, with CIs, on the drifted held-out eval and on the clean eval (forgetting check), plus training time and cost. Store weights with DVC. Runs on the GPU session; on CPU, use a smaller subset and say so.
**Done when:** before/after table in `BENCHMARKS.md`; weights tracked with DVC; limitation stated that augmentation-based drift is not real domain shift.
**Considerations:** no sweep and no experiment tracker in this plan. Geometric drifts (camera shift) need box adjustment, so skip them.
**As built (2026-10-19, D36, E-060, E-061):** `vms.finetune` with a frozen training list v1 (800 COCO train2017 + 767 MOT17 frames, disjoint from the eval and calibration sets), blur as the target (worst for the camera scenes, E-051), a blur-mix run and a no-blur control with one fixed configuration, trained on an AWS c7i.4xlarge (D35, $0.80) and scored on detection subset v1 clean, blur 3/6 and noise 25. Negative: the augmentation is worth +4.4 MOT17 mAP@50-95 at blur 6 over the control, but the recipe loses 2 to 7 points elsewhere; not deployed. Weights DVC-tracked; limitation (synthetic blur) stated with every number.

### 9.6 Fine-tune follow-up (added 2026-10-19)

**Prompt:** 9.5's no-blur control lost 2 to 7 points against the base model on every condition (E-061), so the training recipe hurts by itself. Change exactly one thing in the D36 configuration, either freezing the backbone (`freeze` the first layers, so only the neck and head adapt) or turning off Ultralytics' warmup, and re-run both the blur-mix run and the control on the same frozen training list v1, seed and eval (clean, blur 3 and 6, noise 25; COCO and MOT17). Pick the change before running and say why in the decision record; do not try both and report the better one.
**Done when:** a before/after table next to 9.5's in `BENCHMARKS.md` (base, 9.5 runs, 9.6 runs); D36 updated or a new decision says whether a fine-tuned model is deployed; weights tracked with DVC; cost logged.
**Considerations:** the same AWS CPU path as 9.5 (`make cpu-session INSTANCE=c7i.4xlarge`, `VMS_MAX_SESSION_MINUTES=240`); 9.5's session hit the 1 h scenario timeout, so raise `timeouts.scenario_s` for this one. If neither run beats the base on clean data, the result stays negative and that is reported as such.

---

## Phase 10: Cloud and continuous delivery

**Status:** whole phase deferred (2026-10-06 re-scope). The GPU session from 2.4 is the only cloud work in this plan. D10 has a default.

**Goal:** reproducible cloud deployment, a CD pipeline with an eval gate, and the final full-scale numbers.
**Exit gate:** a tagged release deploys through CD to a provisioned cluster; final benchmark published; production checklist complete or gaps stated.

### 10.1 Cloud stack

**Status:** deferred (2026-10-06 re-scope; not planned). Kept unchanged for reference; it returns only through the scope-lock rule in `PLAN.md`.

**Prompt:** Extend the AWS Terraform from 2.4 into a full environment in the project Region: EC2 spot GPU node(s) (g5 or g6) running k3s, S3 (frames, artifacts, DVC remote), ECR, IAM with least-privilege roles (instance, CI push, read-only where possible), secrets (AWS Secrets Manager or SSM Parameter Store) and AWS Budgets alerts, with every resource tagged `project=vms` and `session=<id>`. Use EKS only if a decision record justifies it and the user approves the cost. Keep a single command each for up and down; down ends with the tagged-resource check from 2.4. Confirm cost with the user before applying.
**Done when:** environment comes up and down by script; the tag check reports zero leftovers; `RUNBOOK.md` Deployment and Teardown sections updated.

### 10.2 CD with eval gate

**Status:** deferred (2026-10-06 re-scope; not planned). Kept unchanged for reference; it returns only through the scope-lock rule in `PLAN.md`.

**Prompt:** Add a release workflow: on a version tag, build and push images, then deploy with `helm upgrade` behind a manual approval (the cluster is ephemeral). Add an eval gate that runs the structured dev subset (about 100 questions) and detection mAP (clean and drifted) against the candidate model, adapter or prompt and fails if results drop more than a stated threshold below the champion recorded in `eval/champion.json`. Log gate runs to MLflow; track candidates and the champion in the MLflow model registry, with `eval/champion.json` mirroring the champion's metrics in git so the gate works from the repo alone. Promote the candidate to champion only on pass. If no LLM key is available in CI, skip with a clear warning.
**Done when:** a deliberately degraded prompt is blocked by the gate; a normal change passes; the 9.5 detector has gone through the gate as a candidate.

### 10.3 Full-scale benchmark

**Status:** deferred (2026-10-06 re-scope; not planned). Kept unchanged for reference; it returns only through the scope-lock rule in `PLAN.md`.

**Prompt:** Run the final ingest ramp and mixed query load on the cloud environment. Report rated capacity, freshness p95, query p95 per mode and cost per camera-hour, with all settings and hardware.
**Done when:** final section in `BENCHMARKS.md`; resources torn down and cost logged.

### 10.4 Production checklist

**Status:** deferred (2026-10-06 re-scope; not planned). Kept unchanged for reference; it returns only through the scope-lock rule in `PLAN.md`.

**Prompt:** Write a production readiness checklist in `RUNBOOK.md`: SLOs defined and measured, alerts and dashboards, load and chaos tested, backup and retention, security (read-only role, secrets, rate limits, red-team results), rollback tested, runbooks for likely incidents. Mark each item met or not met with evidence links.
**Done when:** checklist complete with honest status.

### 10.5 Orchestrated retraining loop (stretch)

**Status:** deferred (2026-10-06 re-scope; not planned). Kept unchanged for reference; it returns only through the scope-lock rule in `PLAN.md`.

**Prompt:** Build a DAG: trigger (manual or drift alert webhook), sample frames for labeling or pseudo-labeling, fine-tune using the 9.5 script, run the 10.2 eval gate, register the candidate in the MLflow registry, optionally deploy. Choose the orchestrator (Prefect or Airflow) by operational weight on compose and k3s, DAG expressiveness and effort, and record D10.
**Done when:** a drift alert (or manual trigger) runs the DAG end to end on the CPU stack with tiny models; D10 accepted.
**Considerations:** stretch task, below polish in the cut lines and not scheduled in the week estimates. Keep it out of the `make demo` path.

---

## Phase 11: Packaging

**Goal:** make the work legible to a hiring manager in two minutes and to an engineer in twenty.
**Exit gate:** README, write-up and video published; fresh-clone reproduction verified.

### 11.1 README

**Prompt:** Rewrite `README.md`: one-paragraph pitch, headline numbers with CIs (each linked to its results and hardware), UI and dashboard screenshots, architecture diagram, precision ladder summary, quickstart (`make setup && make up && make demo`), and links into the docs. Keep it short.
**Done when:** a reader can grasp the system and its results without opening other files.

### 11.2 Write-up

**Prompt:** Write `docs/WRITEUP.md`: problem and workload, architecture, each major decision with its evidence (and which decisions are unbenchmarked defaults), the baseline and the precision ladder, the evaluation methodology with CIs and the error taxonomy, drift detection results, what did not work and why, limitations (synthetic data, self-measured numbers, small eval sets, eval label biases), screenshots, and future work drawn from the deferred list.
**Done when:** every claim links to evidence; limitations section is candid.

### 11.3 Demo

**Status:** shrunk (2026-10-06 re-scope).
**Prompt:** Record a 2-minute screen capture: the UI camera wall, a few structured queries with the plan and SQL panel, the Pipeline dashboard during a load step, and a drift alert firing on the Drift dashboard during an injected scenario.
**Done when:** video linked from the README; demo instructions in `RUNBOOK.md`.
**Considerations:** no live demo in this plan.

### 11.4 Reproduction test

**Prompt:** On a clean machine or VM, follow only the README and RUNBOOK from a fresh clone. Fix every gap found.
**Done when:** quickstart works unaided; fixes committed.
