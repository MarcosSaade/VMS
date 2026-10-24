# Experiments

> Location in repo: `docs/EXPERIMENTS.md`. Append-only log of every attempt that tests a hypothesis or produces a number: benchmarks, eval runs, tuning, training runs, abandoned approaches and bugs that changed results. Failed attempts count. Never edit old entries; corrections are new entries that reference the old ID. Curated current numbers live in `docs/BENCHMARKS.md`.

## Entry format

```
### E-014 | 2026-11-02 | storage
Tried: ClickHouse MergeTree ORDER BY (camera_id, ts), 100M rows, 50k-row batch inserts.
Why: D1 candidate for time-range aggregates.
Result: ingest 410k rows/s; hourly agg p95 38 ms (PG: 1.9 s); 11 B/row. bench/results/2026-11-02-storage-ch.json
Outcome: keep; feeds D1.
```

- IDs increase monotonically (E-001, E-002, ...).
- Tags: `infra, ingest, queue, storage, vector, inference, tracking, query, query-eval, retrieval, llm, lora, drift, k8s, cloud, frontend`.
- At most ~6 lines. Numbers over adjectives. Link the results file. State the hardware and whether the data is synthetic.

## Log

### E-001 | 2026-10-06 | infra
Tried: MinIO as the local object store (task 0.5), pinned to its last release tag.
Why: named in the plan; S3-compatible stand-in for S3.
Result: `minio/minio` and `minio/mc` no longer exist on Docker Hub; `quay.io/minio/*` pulls return unauthorized. MinIO stopped publishing community binaries and images in late 2025.
Outcome: abandoned; replaced by SeaweedFS 4.48 in `weed mini` mode (D12). Only a source build would keep MinIO.

### E-002 | 2026-10-06 | infra
Tried: memory footprint and startup time of the compose CPU stack (i7-1355U, 12 threads, 15 GiB RAM; `docker stats` snapshot at idle, not a benchmark).
Result: MLflow 3.16.1 with defaults (4 uvicorn workers + server job runner, 7 huey consumers) used 2.1 GiB; with `--workers=2` and `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false`, 534 to 565 MiB. Whole stack about 1.0 GiB (Grafana 153, SeaweedFS 142, others < 45 MiB each). `make up` to healthy: 18 s from empty volumes, about 14 s warm (images cached).
Outcome: keep the trimmed MLflow settings; the job runner serves GenAI scorer jobs, which we don't use.

### E-003 | 2026-10-06 | query
Tried: SQL validator on sqlglot 30.21 ASTs (task 0.6, built from scratch, no legacy code), probed with 133 accept/reject cases.
Result: three sqlglot behaviors forced design changes. (1) Known functions get canonical names (`generate_series` -> `EXPLODING_GENERATE_SERIES`, `version` -> `CURRENT_VERSION`), so the denylist matches the name as written, canonical and re-rendered. (2) `unnest(...)` in FROM is not a `Table` node, so FROM sources are allowlisted by node type (named table or subquery). (3) A global CTE-name list let `(WITH pg_user AS ...) s CROSS JOIN pg_user` reach the real `pg_user`; fixed with per-scope resolution (`traverse_scope`), regression test added. Unquoted identifiers are normalized first, so `FROM PG_USER` cannot dodge the allowlist.
Outcome: keep; D14. All 23 few-shot gold SQL statements pass the validator and `EXPLAIN` on Postgres 17 against the allowlisted schema (`tests/integration/test_query_assets.py`).

### E-004 | 2026-10-06 | ingest
Tried: source survey for virtual-camera video (task 1.1).
Result: MOT17/MOT20 zips still downloadable (site front page returns 410), CC BY-NC-SA 3.0, pedestrians only. VIRAT 2.0 videos download anonymously from Kitware with published md5/sha512, but the agreement must be signed by every user. Wikimedia Commons: candidates checked by eye; rejected "PD-CCTV" vendor clips (doubtful authorship, identifiable shoppers) and a rain clip that only showed a storm drain.
Outcome: MOT + one Commons CC BY-SA street clip by default; VIRAT opt-in for vehicles in parking lots (D15).

### E-005 | 2026-10-06 | ingest
Tried: HTTP range reads of single sequences from the 5.9 GB MOT17 and 5.0 GB MOT20 zips (i7-1355U, home connection).
Result: one span request per sequence: full sequences pinned in 10 to 21 s each. Trimmed sequences are interleaved with their other frames, so the first 1.5x span-overhead limit fell back to per-member requests: 132 to 158 s for 300 frames. With a 4x limit, MOT20-03 (750 of 2405 frames) took 53 s. `make data` from empty: 2 min 52 s, 824 MB of clips; `make data-mini`: 42 s; re-runs: under 1 s. Rebuilding a clip reproduced its content hash exactly.
Outcome: keep the span prefetch with a 4x limit.

### E-006 | 2026-10-06 | ingest
Tried: decoding the Commons clip Intersection_Pie-IX-Sherbrooke.webm (VP8) with OpenCV 5.0.
Result: OpenCV reported 30000 fps (ffprobe: 30000/1001), so the 30 fps cap kept 1 frame in 1000: 2 frames out of 1816, and the run still reported success.
Outcome: added `source.fps` to pin a rate in the spec, plus a hard failure on probed rates outside 1 to 120 fps; regression tests added. (The clip was later dropped for camera motion, E-007.)

### E-007 | 2026-10-06 | ingest
Tried: checking that clips come from static cameras: first-vs-last frame difference, phase correlation every 0.5 s, and median sparse optical flow on corners.
Result: Pie-IX pans throughout (68 of 121 half-second steps moved; visible framing change), so it was dropped. Roncesvalles is handheld but steady until about 10.6 s, so it was trimmed to 10 s. Both automatic metrics are unreliable in crowds: they flagged MOT20-01 (26/35 steps) and MOT20-03 because the crowd's motion dominates. MOT17-02/04/08/09 showed 0 moving steps.
Outcome: static-camera checks stay manual (look at frames). MOT20 is kept as static per the benchmark's documentation.

### E-008 | 2026-10-06 | infra
Tried: the compose stack after task 0.7 removed Redis and MLflow (same laptop as E-002; `docker stats` at idle, not a benchmark).
Result: about 365 MiB in total (Grafana 167, SeaweedFS 75, OTel collector 38, Prometheus 38, Postgres 24, Tempo 23 MiB), down from about 1.0 GiB (E-002). `make up` from empty volumes: 18 s. 21 Python packages dropped from the environment (mlflow-skinny, wandb, redis and their dependencies).
Outcome: keep; D16.

### E-009 | 2026-10-09 | ingest
Tried: loadgen self-test (task 1.2): 20 cameras unpaced on one event loop, NullTransport (no I/O), 1280x720 q85 frames, 3 x 15 s after 3 s warmup, tracing at 1%. i7-1355U, 12 threads, 15 GiB; other projects' containers also running. Commit c5cff6c.
Result: 36.4k frames/s (std 1.75k; 34.4k to 37.6k), 27 us CPU per frame, 1.0 core (the single loop is saturated). bench/results/2026-10-09-loadgen-selftest-01M4GK6TJTHH0D2T3JW9KHWK28.json
Outcome: the generator core is ~360x the 20x5 target; the per-frame cost that matters is the HTTP client (E-010).

### E-010 | 2026-10-09 | ingest
Tried: loadgen max rate over real HTTP: unpaced multipart POSTs over loopback to the stub ingest (6 worker processes, so the stub is not the limit), 20 cameras x 4 in flight, 3 x 20 s. Same host as E-009.
Result: 1,252 frames/s (std 44; 1,205 to 1,293), 0.80 ms CPU per frame, 0.985 cores: the generator is the bottleneck. A 1-worker stub saturated first at ~670 frames/s. Ad hoc probe (scratch script, not a reported number): aiohttp server-side multipart parsing took ~1.7 ms per 200 KB frame vs ~0.04 ms for a raw body. bench/results/2026-10-09-loadgen-stub-max-01M4GK893V2X7WP677GQEXG8GW.json
Outcome: one loadgen process tops out near 1,250 frames/s (~250 cameras at 5 FPS); keep offered load below ~70% of that, or shard cameras across processes. Multipart cost is an input to 1.3 and D4: measure raw-body ingest there with a scenario.

### E-011 | 2026-10-09 | ingest
Tried: task 1.2 done criterion: 20 cameras x 5 FPS paced (100 frames/s, 1280x720 q85, ~165 KB) against a 1-worker stub, 3 x 60 s after 5 s warmup. Same host as E-009.
Result: stable: 100.0 frames/s offered and accepted in every rep (16,500 of 16,500 steady-state frames), 0 dropped, 0 failed. Send latency p50 8.7 ms, p95 17.6, p99 24.1. Schedule lag p99 2.3 ms (limit 10; one rep had a 27 ms max). Loadgen 0.16 cores, 1.6 ms CPU per frame, twice the unpaced figure (idle wakeups per frame; not investigated). bench/results/2026-10-09-loadgen-stub-20x5-01M4GKA7GNXHJX3ABMV91AKYJ1.json
Outcome: done criterion met; generator_ok in every rep.

### E-012 | 2026-10-09 | ingest
Tried: the first recording of E-009 to E-011 (commit 48b9814).
Result: two harness bugs. (1) Every results file after the first in a session said dirty: true, because untracked results files counted as changes. (2) SIGTERM to the 6-worker stub left its workers on port 8080; the 1-worker stub then failed to bind, so the "20x5 vs 1-worker stub" run actually hit the leftover 6-worker stub (latency p50 6.4 ms instead of 8.7 ms).
Outcome: both fixed with regression tests (c5cff6c); all three scenarios re-run, earlier files discarded. Benchmark scripts now check that the port is free before starting a stub.

### E-013 | 2026-10-12 | inference
Tried: worker model-only latency (task 1.5), yolo11s, PyTorch 2.14.1 CPU, batch 1, imgsz 640; 150 frames (50 each of the 3 data-mini clips, 1280x720 q85), 10 warmup, 3 reps. i7-1355U, other containers running (load avg ~7). Commit d37b10b.
Result: 9.8 frames/s (std 0.65); infer p50/p95/p99 90/100/~120 ms; decode p50 3 to 6 ms; stats p50 5 to 9 ms; total p95 116 ms; torch used ~7.5 cores; 16.7 detections/frame. bench/results/2026-10-12-worker-model-cpu-01M4QT5GC08N5MNH1CY0WNWS9Q.json
Outcome: one CPU worker with yolo11s serves ~2 cameras at 5 FPS, short of the 4x5 demo load. Input stats cost as much as decode (Laplacian in float64 at full resolution): a cheap 2.6 candidate.

### E-014 | 2026-10-12 | inference
Tried: E-013 with only the weights changed to yolo11n. Same host and commit.
Result: 20.0 frames/s (std 1.07); infer p50/p95 39/50 ms; total p95 65 ms; 13.2 detections/frame (fewer than yolo11s; no mAP yet, task 3.4). bench/results/2026-10-12-worker-model-cpu-n-01M4QT70RZG5SB1GTPH058VAXT.json
Outcome: yolo11n just matches the 20 frames/s demo load model-only, with no headroom once DB and object-store time are added. Input to 1.9 (demo model and FPS) and to the CPU ladder (5.2).

### E-015 | 2026-10-12 | tracking
Tried: demo smoke run (not a benchmark): ingest + one yolo11n worker + loadgen, 4 catalogued cameras x 5 FPS for 30 s on the data-mini clips, dev database, laptop.
Result: tracks lasting hours (mean duration 2.3 to 6.6 h per camera). Cause: 30 frames per camera left pending since a 1.3 smoke run (2026-10-09) were processed first, and ByteTrack's lost-track buffer counts frames, so it joined objects across the 3-day gap. After a reset on any gap longer than buffer_s (regression test added), a re-run on a fresh database gave 329 tracks, max duration 29.8 s; zone events on all three zones (75 enters, 57 exits; visits still open at the end have no exit); the worker drained 20 s after loadgen stopped.
Outcome: keep the time-gap reset (D21). The dev database still holds tracks from the buggy run.

### E-016 | 2026-10-12 | inference
Tried: E-013 and E-014 with only the detector threshold changed, 0.25 -> 0.1 (the tracking default, D21). Same host and scenarios; commit 8d270be.
Result: yolo11s 9.79 frames/s (std 0.22; was 9.81), infer p50 89 ms; yolo11n 18.8 frames/s (std 1.9; was 20.0, std 1.07), infer p50 40 ms. Raw detections per frame 27.8 and 23.2 (were 16.7 and 13.2) before the storage rule. bench/results/2026-10-12-worker-model-cpu-01M4QVF9ZAKC9HXMTFERF164VV.json, bench/results/2026-10-12-worker-model-cpu-n-01M4QVGTE15P83D77JXS71K9VF.json
Outcome: no measurable latency cost from the lower threshold (yolo11n difference inside run-to-run noise on a busy laptop). BENCHMARKS now shows the 0.1 runs.

### E-017 | 2026-10-12 | llm
Tried: planner smoke run, not an eval: `python -m vms.query plan --compare configs/demo/canned_plans.v1.jsonl` with DeepSeek (`deepseek-flash`, `LLM_BASE_URL=https://api.deepseek.com`), planner.v1 prompt with all 29 few-shots, cloud container, commit d98dc94 plus the uncommitted 1.7c work.
Result: `json_schema` response_format rejected by the API ("unavailable"); with `json_object`: 10/10 valid plans, no repairs; 8/10 identical to the canned plans, the other 2 equivalent (zones register_1/2 instead of camera checkout; limit 20 instead of 12). ~4.5k prompt tokens per question; latency 1.2 to 3.6 s (p50 ~1.8 s, n=10). Output not committed (ad hoc smoke).
Outcome: `llm.response_format` made configurable (D22). Exact plan match undercounts correct answers: score execution results in 3.2. The planner alone is above the 1.5 s query p95 SLO with this hosted model.

### E-018 | 2026-10-12 | query
Tried: logging and prompting plans with `model_dump(exclude_defaults=True)` (compact JSON).
Result: a unit test found the time range's `kind` dropped (it has a default, and it is the union discriminator), so the dumped plan did not parse back. It affected the few-shot examples rendered into the planner prompt (they taught a format that fails validation), canned replies with time ranges, and plans in responses and query_log. Found before any planner run.
Outcome: `QueryPlan.compact()` keeps `kind`; a test round-trips every few-shot plan.

### E-019 | 2026-10-12 | frontend
Tried: building the application image (`deploy/docker/Dockerfile`: node 22 UI stage + python 3.12 slim with `uv sync --locked --no-dev`) and the UI, in the cloud container (not a benchmark).
Result: image 974 MB compressed, 4.31 GB unpacked (CPU PyTorch, Ultralytics and their dependencies); cold build 3 min 36 s, rebuild after a source change seconds (dependency layer cached). UI bundle 10.4 KB JS, 4.7 KB CSS. `npm install` with Node 22's npm 10.9.4 crashed in arborist (`Cannot read properties of null (reading 'edgesOut')`) resolving Vite's optional peers; npm 11.20.0 wrote the lockfile, and npm 10's `npm ci` installs from it.
Outcome: keep one image for now (D23); a slim query image is the lever if build time matters. Regenerate frontend/package-lock.json with npm 11 (`npx npm@11 install`).

### E-020 | 2026-10-12 | infra
Tried: `make demo DEMO_SECONDS=40` (not a benchmark): containerized ingest, one yolo11n worker and the query service, loadgen 4 cameras x 5 FPS on the data-mini clips, canned planner; cloud container CPU (shared, unknown model), the dev database already holding an earlier run.
Result: exit 0 in 7 min 4 s, including the app image rebuild. Loadgen 20.0 frames/s offered and accepted, 0 failed. The worker drained at about 7.5 frames/s (backlog 524 when loadgen stopped, empty 70 s later). All 10 example questions answered (7 with rows, 3 abstentions as intended); query latency 1 to 26 ms with the canned planner. 2,708 frames done in total; tracks on all 4 cameras, zone events on the 2 cameras with zones.
Outcome: the default 180 s demo takes ~10 min on this machine, mostly waiting for the worker; acceptable for a demo, and the wall lags during load. Also found and fixed: `python -m vms.worker weights` crashed logging `extra={"name": ...}` (reserved LogRecord attribute), so `make models` failed after the first file (regression test added); the answer template printed Postgres numeric values unrounded.

### E-021 | 2026-10-12 | infra
Tried: `make setup && make demo DEMO_SECONDS=30` on a fresh clone of the 1.9 commit (stack volumes deleted first; Docker image cache kept; EXTRA_CA_CERT added to .env for this environment's TLS proxy).
Result: two failures. (1) Wikimedia Commons answered 429 for the Roncesvalles clip (rate limit on this environment's shared IP, also seen in step 0); the verified clip was copied from the main checkout to continue. (2) Every question failed with planner_unavailable: Docker Compose read `LLM_MODEL=   # comment` from the fresh .env as the value "# comment" (python-dotenv reads it as empty), so the container chose the OpenAI client with a nonsense model. Runs from the older .env never hit it. The same pattern affected LLM_API_KEY, S3_REGION, AWS_REGION and the new EXTRA_CA_CERT (which would have broken the image build in CI).
Outcome: comments for variables that may be empty moved to their own lines in .env.example; a unit test forbids the pattern. Existing local .env files created from the old .env.example still carry it and should be fixed by hand (or recreated).

### E-022 | 2026-10-12 | infra
Tried: the shortened `make demo` in CI (run 38261052259, commit b80f36e), the first run on a machine without the `vms/app:local` image.
Result: failed at step 3: compose tried to pull `vms/app:local` for ingest and worker (which reuse the query service's image and have no build section) before building it ("pull access denied"). Every earlier run here had the image already. Reproduced locally by deleting the image.
Outcome: ingest and worker get `pull_policy: never`, and `make demo` and `make app` build the image explicitly before `up`. Re-run here without the image: exit 0.

### E-023 | 2026-10-12 | infra
Tried: the shortened `make demo` in CI after E-022 (run 38264465076, commit fbc1af8).
Result: failed at step 3 before building: `compose build` (BuildKit bake) refused the build secret's default file `/dev/null` ("additional privileges requested: pass --allow=fs.read=/dev/null"): bake grants no reads outside the project. The local Compose v5.6.0 does not enforce this, even with COMPOSE_BAKE=true, so it passed here.
Outcome: the default secret is an empty file in the repo (deploy/docker/no-extra-ca.pem); a user CA goes under .certs/ (gitignored). Verified only by the next CI run, since the failure does not reproduce locally.

### E-024 | 2026-10-13 | infra
Tried: task 2.1 done check: `make demo DEMO_SECONDS=30` (4 cameras x 5 FPS, yolo11n, 10% of frames traced, DeepSeek planner from the local .env), then `python -m vms.observe`. Laptop of E-013, other containers running; one trace each, not a benchmark.
Result: one frame trace from loadgen to the DB write: send 11.5 ms (multipart parse 0.7, object put 7.0, row insert 1.4), queue wait 20.0 s, worker 84 ms (infer 66, decode 3, stats 3.5, write 5.5, fetch 4, track 1.3). One question: 2.48 s, of which the LLM call 2.43 s, validate 21 ms, execute 4.5 ms. Every required metric present; the ingest, worker and query targets discovered and up.
Outcome: at the demo load, freshness is queueing, not work: the worker is slower than the 20 frames/s offered (E-020), so 2.5 must find the camera count where the queue stays empty. The first span to open when freshness rises is `worker.queue_wait`. Tempo made traces searchable 1 to 2 min after they ended, so the check retries for up to 240 s.

### E-025 | 2026-10-13 | infra
Tried: the first demo runs with the 2.1 instrumentation.
Result: three problems. (1) FastAPI 0.143 now creates HTTP server spans itself; the services' own `traceparent` extraction replaced that context, so each query's `query` span became a separate trace from its `POST /v1/query` span (and ingest's stages hung off loadgen, not the HTTP span). (2) Docker health checks and Prometheus scrapes produced a one-span trace every few seconds per service. (3) The telemetry check passed on a trace from the previous demo run while the new ones were not searchable yet.
Outcome: extract only when no span is active (`remote_parent`), exclude `/healthz`, `/readyz`, `/metrics` from FastAPI telemetry, and search only from the run's start (`--since`); unit tests for each. Re-run: the question trace nests under `POST /v1/query` (E-024).

### E-026 | 2026-10-13 | infra
Tried: dashboard screenshot run (task 2.2), not a benchmark: app containers (one yolo11n worker), loadgen 4 cameras x 2 FPS for 150 s, then x 5 FPS for 150 s, demo questions asked once per phase, then the drain. Laptop of E-013 with other containers running. Read from Prometheus.
Result: at 8 frames/s offered the worker kept up: freshness p95 0.31 s, worker busy 69% of the time, inference p50 63 ms. At 20 frames/s it saturated at the same 8.0 frames/s (busy 98%, 2.9 cores, inference p50 rose to 86 ms): backlog peaked at 1,780 frames, the oldest waited 3.6 min, 0 frames lost. The first screenshots exposed two histogram problems. (1) Freshness buckets jumped from 60 to 300 s, so p95 read 4.8 min when no frame had waited much over 2 min. (2) The worker's stage buckets had nothing between 0.1 and 0.25 s, so inference p95 sat flat at 100 ms. Also, the zero fallback `or vector(0)` added a "none" series next to real data (fixed with `or on()`).
Outcome: buckets made denser (freshness up to 600 s; stages from 25 to 300 ms), worker rebuilt, run repeated; docs/images/grafana-*.png come from the second run. One measured lead for 2.6: one saturated worker does 8 frames/s here, although 69% busy at 8 frames/s suggests about 11.6. Inference slows by a third when the worker never idles (CPU contention or thermal throttling; an earlier run on the same machine saturated at about 12 frames/s).

### E-027 | 2026-10-13 | infra
Tried: task 2.3 done check, `make bench` (bench-smoke: 1 and 2 cameras x 5 FPS, 1 query client, warmup 10 s, hold 20 s, 3 repetitions) against the containerized demo services (yolo11n worker, DeepSeek planner from the local .env), laptop, during the task 2.2 screenshot session (E-026: its worker was rebuilt and its runs repeated around the same time; whether its loadgen overlapped these steps is not known) and with unrelated containers active. Not a capacity measurement.
Result: valid results file; every step drained first (first drain 172 s: 1,436 frames left by a demo). Worker throughput varied 2.9 to 10.1 frames/s across repetitions at the same load (worker CPU 1.3 to 2.9 cores), so the ingest SLO held in 2/3 repetitions at 1 camera and 1/3 at 2 (failing steps: freshness p95 18 to 19 s, backlog slope 1.9 to 7 frames/s); 0 frames lost; loadgen generator_ok and client CPU fine in every step. Query p95 3.0 to 4.2 s with ~4.5k tokens per question (hosted LLM, not the canned planner). bench/results/2026-10-13-bench-smoke-01M4THD6R6QHYQ0C11MDGJ393V.json
Outcome: the runner works end to end, and the backlog-slope check caught every failing step. The variance says a shared laptop cannot rate capacity: 2.5 needs a quiet host (nothing else running) and the canned planner for query load (zero cost, latency not LLM-bound) unless an LLM is the variable under test.

### E-028 | 2026-10-13 | cloud
Tried: task 2.4 preflight against the real account (us-east-2, paid plan): dry-run launches of g6.xlarge from the DLAMI Base OSS Nvidia Driver (Ubuntu 22.04, ami-0f5f372f671b84b92), on-demand and spot; `scripts/gpu_session.sh preflight`; Price List and spot price history; `uv pip compile` of torch 2.14.1+cu126 for the GPU image. No instance launched.
Result: on-demand dry run passes; spot fails with an explicit deny in the managed SCP. On-demand G and VT quota (L-DB2E81BA) is 0 (4 requested, case open), so preflight stops there (exit 4); with the quota check pointed at the spot quota (4) every other check passes. On-demand $/h: g4dn.xlarge 0.526, g6.xlarge 0.8048, g6.2xlarge 0.9776, g5.xlarge 1.006; g6.xlarge spot 0.39 to 0.41 over the week. Session estimate $0.821/h (instance + 100 GB gp3 + IPv4), cap $2.05 (150 min self-terminate). `aws login` credentials need botocore[crt] (added to dev). CUDA wheels: with `--index-url pytorch --extra-index-url pypi`, uv searched PyPI first and found no `+cu126` build; `--index pytorch --default-index pypi` resolves (32 packages).
Outcome: D28 (on-demand g6.xlarge). The first real session waits for the quota. Budget `vms-monthly` created ($20/month, excludes credits).

### E-029 | 2026-10-14 | query-eval
Tried: building the structured eval set (task 3.1, D29): synthetic snapshot v1 (seed 20260331), 129 items with gold SQL and gold plans, `make eval-db` and `make eval-build` against the compose Postgres, laptop of E-013. Not a model eval.
Result: snapshot 332,730 frames, 31,132 tracks, 211,193 detections, 26,781 events (16.4 MB gzipped CSV); generated in ~4 s, regenerated byte for byte (also from scratch in the integration test), loaded in ~6 s. The first build passed every gold-vs-plan agreement but flagged one undeclared near duplicate (sq-016/sq-101, ratio 0.87; reworded). Review of the results found an ill-posed top-4 (tie at the cut, now top-3), an empty frames answer (window moved) and a near-tied "busiest hour" (422 vs 421; moved to the lobby, 227 vs 190). Perturbing 6 gold SQLs: 4 caught; 2 not, both equivalent on this data (first_seen vs overlap for the lobby today; avg ignores visits without exit). Rebuilds changed the test file's hash: Postgres averages differed in the last bits between runs (sq-021) and unordered rows came back in another order; stored results are now canonical (rows sorted unless ordered, floats rounded to 9 places), and 3 rebuilds plus a reload gave the same hash.
Outcome: keep; test split sha256 5560a5fb... recorded in BENCHMARKS.md. Agreement checks consistency, not intent: the user's review of the test split is the remaining gate.

### E-030 | 2026-10-14 | query-eval
Tried: doubling the structured eval set from 129 to 279 items (150 new: 106 plan-expressible across every category, 8 ambiguous, 8 unanswerable, 16 unsupported, 12 adversarial) for a narrower test interval; same snapshot v1, same build.
Why: 67 test items gave about +/-11 points (95%) at 70% accuracy, too wide to separate two planners.
Result: build checks caught 3 near duplicates (one of the few-shot bank, two across groups; reworded), one frames list over the 100-row validator limit (window narrowed) and 3 empty answers (thresholds or windows changed); no boundary ties in ordered items. Test split now 142 items (about +/-7.5 points), sha256 0330b3db... ; the 129 old items were re-split with the new ones (no test number was reported on the old split).
Outcome: keep; BENCHMARKS.md and D29 updated.

### E-031 | 2026-10-14 | query-eval
Tried: the 3.2 harness end to end. (1) `make eval PLANNER=gold` (gold plans as the LLM's replies) on dev, and the same over dev+test in the integration test. (2) A free real-LLM smoke run: `--limit 4` on dev with local Ollama qwen3:1.7b (CPU), concurrency 1, 180 s LLM timeout. Laptop of E-013.
Result: (1) 100% on every metric, 279/279 items, 0 breaches; dev in 2.6 s (p50 12 ms, p95 45 ms per question). (2) No answer after 10 min; stopped. Ollama ran the model with a 4,096-token context, shorter than the ~4,600-token planner prompt, so the prompt would be truncated anyway.
Outcome: the harness, scorer and dataset agree with each other. Real-planner numbers need a hosted model (DeepSeek, as in E-017), which costs money: waiting for the user's go-ahead. A local model needs a larger context and a faster machine.

### E-032 | 2026-10-14 | query-eval
Tried: first real-planner eval, `make eval SPLIT=dev`: plan engine with DeepSeek deepseek-flash (json_object, planner.v1, 29 few-shots, temperature 0), concurrency 4, synthetic snapshot v1, laptop. The first attempt failed every question with HTTP 400 (json_schema unavailable): the harness did not read .env's VMS_LLM__RESPONSE_FORMAT, which compose passes to the service; fixed before this run.
Result: accuracy 94.2% [89.8, 97.8] (n=137), plan-expressible 94.8% [89.6, 99.0], abstention correct 92.7% [82.9, 100], red team refused 12/12 with 0 breaches; p50 1.8 s, p95 4.6 s; 4,715 tokens per question; 95 s wall. 8 errors: 2 counted events for "how many people entered" (gold: distinct tracks), 2 exhaustion attacks refused with the wrong reason, 1 refused a supported zone question, 1 answered "how many classes" with one row per class (4 rows; the gold treats count-distinct as unsupported, a borderline call), 1 invalid plan, 1 unrequested limit 20. 2026-10-14-structured-dev-deepseek-flash-01M4X1CYSJZH1946PW8FJKE5HD.json
Outcome: baseline for D7 and 3.5 (compare a second LLM on dev). Candidate prompt fixes: few-shots for distinct people vs entries and for resource-exhaustion refusals. Not tuned yet: dev is the tuning split, test stays untouched.
### E-033 | 2026-10-14 | inference
Tried: building detection subset v1 (task 3.4, D31): 500 COCO val2017 images (seed 20261014) and MOT17-02/04/09 at one frame per second, read from the official zips with one range request per member; laptop of E-013. Not a model eval.
Result: 573 images, 95 MB, built in 52 s (COCO's annotations member 15 s; 8 concurrent requests). COCO 3,833 boxes + 49 crowd regions; MOT17 1,961 persons + 980 ignore regions. A first MOT labelling with the benchmark's convention (pedestrians only; static persons 7 and persons on vehicles 2 ignored) gave 1,761 positives and 1,180 ignores; switched to every visible person because a person standing in a zone is what the VMS counts. MOT17's flag column is 1 exactly for class 1, so it is not used. A rebuild produced the identical list; deleting the images and running `fetch` restored identical bytes. images.cocodataset.org fails TLS (certificate for another name); the path-style S3 URL works.
Outcome: keep; list sha256 1661781d... in eval/detection/subset.v1.json and BENCHMARKS.md.

### E-034 | 2026-10-14 | inference
Tried: `make eval SUITE=detection` for yolo11s and yolo11n (PyTorch CPU, imgsz 640, conf 0.001, NMS 0.7), 2,000 bootstrap resamples, from clean commit bab0b45; laptop. eval/results/2026-10-14-detection-yolo11s-01M4X1QXXBF2VN2NKXY9AYPBV9.json, eval/results/2026-10-14-detection-yolo11n-01M4X1TQS363MBH227YJQGFH7A.json
Result: COCO mAP@50-95 yolo11s 45.8 [44.1, 49.6], yolo11n 37.9 [36.1, 42.2] (published val2017: 47.0 and 39.5, inside both intervals); MOT17 person mAP@50-95 45.0 [43.1, 47.0] and 40.3 [38.6, 42.0], mAP@50 72.9 and 66.7. Metrics were identical across repeated runs (4 for yolo11s, 3 for yolo11n; deterministic CPU inference). The bootstrap mean sits 1.0 (s) and 1.3 (n) points above the COCO point estimate, 0.1 on MOT: per-class AP bias is +0.022 for classes with 1 to 5 boxes and +0.005 above 80 (running-max envelope). Speed varies with load: 2.3 to 10 img/s for yolo11s on the same laptop (not a benchmark; see E-016).
Outcome: baseline for 5.x; yolo11n's 2x model-only speed (E-016) costs about 8 points of COCO mAP and 5 of MOT person mAP. Compare backends with a paired bootstrap.

### E-035 | 2026-10-14 | query-eval
Tried: after the user's review of the eval set: sq-103 ("how many object classes has the lobby seen") changed from unsupported to a count by class (listing the classes is the answer); every item's split pinned in eval/structured/splits.v1.json (changing one item's category would have moved 36 other items between dev and test). Then the dev run of E-032 repeated with prices configured (peak, cache-miss).
Result: test membership and test file unchanged (sha256 0330b3db...). Dev accuracy 94.2% [89.8, 97.8] again, plan-expressible 94.8% [89.7, 99.0], 0 breaches; 4,840 tokens per question, p50 1.9 s, p95 5.9 s; cost $0.23 upper bound for 137 questions. sq-103 now correct; 3 answers flipped between the two runs at temperature 0 (sq-053 now right; sq-098 and sq-180 now wrong). 2026-10-14-structured-dev-deepseek-flash-01M4X45EXKGJP2055D6A04VPVK.json
Outcome: keep the pins; treat a few points between runs as noise. 3.1 closed (user approved the test split and a dev sample).

### E-036 | 2026-10-14 | query-eval
Tried: task 3.5, the one test-split run of the baseline planner: `make eval SPLIT=test`, typed-plan engine with DeepSeek deepseek-flash (configuration of E-035, nothing tuned on dev), clean commit 4228e11, laptop. Also the gold self-test through the CLI (`make eval PLANNER=gold`, now exit 1 unless 100%; added to CI).
Result: test accuracy 95.8% [92.3, 98.6] (n=142), plan-expressible 95.9% [91.8, 99.0], abstention correct 95.6% [88.9, 100]; red team 13/13 refused, 0 breaches; p50 1.9 s, p95 5.1 s; 4,855 tokens/question; $0.24 upper bound; 97 s wall. 6 errors, the dev failure modes: 2 counted entries for "how many people" (sq-047, sq-166), 2 unrequested limit 20 on frame listings (sq-227, sq-233), 2 wrong abstain reasons (sq-244 mask question unsupported vs unanswerable; sq-274 exhaustion attack). Gold self-test on dev via CLI: 137/137. eval/results/2026-10-14-structured-test-deepseek-flash-01M4X596J9CHNDG20XEMRVQT2F.json
Outcome: reported baseline for D7 (provisional) and the README; dev and test agree within CIs. Top fixes for 7.x or the GPU-session comparison (dev only): few-shots for distinct people vs entries, exhaustion refusals and unlimited frame listings. The test split is now spent for this configuration.

### E-037 | 2026-10-14 | inference
Tried: task 5.1, YOLO11 to ONNX (Ultralytics exporter, opset 18, dynamic batch/H/W, no NMS in the graph, legacy TorchScript path) run on ONNX Runtime 1.31 CPU with our numpy letterbox and NMS; `make onnx-parity` for yolo11n and yolo11s over the detection subset (500 COCO + 73 MOT17) and 150 data-mini frames, at conf 0.1 and 0.001; then `make eval SUITE=detection` on the ONNX backend. Laptop, commit fe9cda2. A stray busy-loop test process of mine held ~1 core during these runs: it affects only the images/s fields, which are not reported.
Result: parity ok on every set. Letterbox byte-identical to Ultralytics on all 1,446 images. Max |diff| over both models: raw boxes 0.021 input px, raw scores 1.1e-4; matched detections boxes 0.018 px, scores 1.6e-5; batch 8 vs one at a time 0.0002 px. Of 68k yolo11n detections at conf 0.001 on COCO, 3 sat on one side only, all explained (2 scores at the threshold, 1 IoU at 0.7); none at conf 0.1. mAP identical to the PyTorch runs to 0.01 points (yolo11n MOT17 40.31 vs 40.32), e.g. yolo11s COCO 45.8 [44.1, 49.6] both. Tolerances set 5x to 50x above these maxima. bench/results/2026-10-14-onnx-parity-yolo11n-01M4XQQGJWF596G49X0GQ06AZY.json, bench/results/2026-10-14-onnx-parity-yolo11s-01M4XSHPCKT13PEV7NGAW18RX5.json, eval/results/2026-10-14-detection-yolo11{n,s}-onnx-*.json
Outcome: keep; the ONNX backend is a drop-in for the 5.2 ladder. The first full run crashed on a COCO image with no detections (empty-list dtype in the matcher; fixed with a regression test). Forward-pass speed not measured here (5.2).

### E-038 | 2026-10-14 | infra
Tried: the first `make bench SCENARIO=bench/scenarios/ramp-cpu.yaml` (task 2.5).
Result: aborted in repetition 1: the laptop's disk filled (858 MB free of 92 GB; the rebuilt app image with onnxruntime added a layer), SeaweedFS stopped assigning volumes ("No writable volumes"), every object put failed and ingest answered errors, so steps showed hundreds of lost frames and no freshness. Pruning the unused Docker build cache freed 7.7 GB (removing 7 untagged old app images freed nothing: shared layers); the object store needed a restart to accept writes again. No results file kept.
Outcome: rerun from scratch (E-039). The runner flagged it correctly (loss_ok false) but kept going for 10 minutes; a free-disk check before each step would fail fast (idea, not planned).

### E-039 | 2026-10-14 | ingest
Tried: task 2.5 (a), `ramp-cpu`: containerized ingest + one yolo11n worker (PyTorch CPU, batch 1) + Postgres + SeaweedFS, data-mini at 5 FPS per camera, 1 to 4 cameras, warmup 20 s, hold 90 s, 3 repetitions; canned planner, no query load. Laptop with the unrelated containers stopped (browser still open), commit 7803f35.
Result: rated capacity 2 cameras (10 frames/s) in 3/3 repetitions: freshness p95 0.24 s, 0 lost. 3 cameras fail every time: the worker tops out at 14.2 frames/s (std 0.3) for 15 offered, backlog +0.85 frames/s, freshness p95 9.4 s; 4 cameras: 13.9 frames/s, p95 37.8 s. Worker 3.5 cores at saturation; ingest 0.15 cores and loadgen 0.04 cores at 20 frames/s; send p95 18 to 20 ms. Stage p95 per frame: infer 73 ms, decode 9.5, fetch 8.5, write 9, stats 8, track 4.5. bench/results/2026-10-14-ramp-cpu-01M4XTS4MEBS8ZM7X2GERGGHS3.json
Outcome: baseline for 5.5. Whole-camera steps hide headroom: the ceiling (14.2 frames/s, 2.8 cameras) is 42% above the rated load, and a gain of about 1.1x (past 15 frames/s with margin) would rate 3 cameras.

### E-040 | 2026-10-14 | query
Tried: task 2.5 (b), `query-load-cpu`: closed-loop clients at concurrency 1, 2, 4, 8 on the demo questions, canned planner (no LLM), ingest at 70% of rated (2 cameras x 3.5 FPS), warmup 15 s, hold 60 s, 3 repetitions; same host and services as E-039.
Result: query p50/p95 16/39 ms at 1 client (58 q/s) rising to 46/98 ms at 8 (159 q/s); the query service pins exactly 1.0 core from 4 clients up, so it is CPU-bound in one process. Ingest SLO held in every step (freshness p95 0.24 s, 0 lost). Client CPU under 0.09 cores. Stage p95 at 1 client: SQL validation 21 ms, execution 24 ms, plan/compile ~1 ms. bench/results/2026-10-14-query-load-cpu-01M4XW866TFS51PVHPXJW25EA0.json
Outcome: everything but the LLM fits the 1.5 s SLO with 15x headroom; with a hosted LLM (p50 1.9 s, E-036) the LLM alone breaks it.

### E-041 | 2026-10-14 | inference
Tried: task 2.6, `profile-worker-cpu`: the worker's code on the host under py-spy (100 Hz, speedscope) with the container's settings, 4 cameras offered (20 frames/s, above the ceiling) for 20 + 60 s. Two earlier runs discarded for harness bugs: weights truncated to 0 by int(), and a `<module>` startup rule that swallowed the main thread (both fixed with tests).
Result: of 65.9 s of sampled Python-thread time: PyTorch forward 65.9%, Ultralytics letterbox 1.8% and NMS/scaling 1.5%, JPEG decode 6.6%, input stats 6.1%, tracking 2.8%, DB 5.0%, object store 2.1%, event loop 4.1%. The inference thread was busy 57 s of the 80 s under load (71%): the rest it waited while the event loop claimed, fetched and wrote frames one at a time. bench/results/2026-10-14-profile-worker-cpu-01M4XXTACAXZPASNPFMDASJC9D.json, bench/profiles/2026-10-14-profile-worker-cpu-01M4XXTACAXZPASNPFMDASJC9D.speedscope.json.gz
Outcome: Phase 5 targets the right stage (forward pass). Serial I/O costs ~29% of the ceiling; pre/post-processing is 3% (5.4 rightly deferred).

### E-042 | 2026-10-14 | query
Tried: task 2.6, `profile-query-cpu`: the query service on the host under py-spy, canned planner, 8 closed-loop clients for 15 + 60 s, no ingest.
Result: 130 q/s, p95 123 ms (with the profiler attached). Of 77 s sampled: SQL validation (sqlglot parse and regenerate) 53.2%, HTTP and pydantic 14.0%, OpenTelemetry 11.6%, execution (psycopg) 10.4%, planning 4.8%, event loop 3.4%, compile 0.7%. bench/results/2026-10-14-profile-query-cpu-01M4XXP40EA6Q9CQ3GV2C73DCB.json, bench/profiles/2026-10-14-profile-query-cpu-01M4XXP40EA6Q9CQ3GV2C73DCB.speedscope.json.gz
Outcome: without an LLM the single-process service is CPU-bound and the validator is half its cost (cache validated SQL per plan, or run several workers). Not worth fixing while the LLM call is ~1.9 s per question.

### E-043 | 2026-10-15 | inference
Tried: task 5.2, INT8 with the whole graph quantized: ORT 1.31 static QDQ (per-channel s8 weights, u8 activations, MinMax) of the yolo11s ONNX export, calibration set v1 (64 COCO train2017 + 64 frames of MOT17-05/10/11/13, built in ~40 s; a rebuild and a fresh fetch gave identical bytes), no excluded nodes; `make eval SUITE=detection`. Clean commit 9b71ac7, laptop.
Why: the naive baseline before excluding anything.
Result: 0 detections on all 573 images, mAP 0.0 on both sources. The output concat joins box coordinates (0 to 640 px) and class scores (0 to 1) in one tensor with one u8 scale (~2.5), so every score dequantizes to 0. eval/results/2026-10-15-detection-yolo11s-int8-noexclude-01M50Q32SZ4STPGDPYXBZRRQEN.json
Outcome: abandoned; the Detect head's decode (DFL, anchor arithmetic, sigmoid, final concat) stays FP32 by default (`configs/quantize.yaml`), its 6 convs stay INT8.

### E-044 | 2026-10-15 | inference
Tried: task 5.2 quality, yolo11s on the detection subset: PyTorch, ORT FP32, and ORT INT8 (E-043 settings with the head decode in FP32: 122 nodes excluded; 87 of 88 convs run in INT8), then paired bootstrap comparisons (`python -m vms.eval.detection compare`, 2,000 resamples, seed 20261014). Clean commit 9b71ac7, laptop.
Result: PyTorch and ORT FP32 identical (paired delta 0.00). INT8 vs ORT FP32, mAP@50-95: COCO 44.77 vs 45.85, delta -1.07 [-1.84, +0.40]; MOT17 45.16 vs 45.03, +0.13 [-0.27, +0.55]. mAP@50: COCO -0.63 [-1.61, +0.98]; MOT17 +0.69 [+0.16, +1.49]. D32 (set before this run): COCO drop 1.07 > 1.0 fails (upper end 1.84 <= 2.0 passes); MOT17 passes; verdict: INT8 not kept. eval/results/2026-10-15-detection-compare-yolo11s-int8-vs-onnx-01M50R9S67XGGJFQ3SYKXE14JT.json, ...-onnx-vs-pytorch-01M50RBNWP219YDN57J11REJDW.json
Outcome: rejected by a margin of 0.07 points while the paired interval still includes 0. Not tuned further here: the subset has no dev split, so trying calibration variants against it would tune on the reported set. A detection dev set (other val2017 images) is the precondition for another INT8 attempt.

### E-045 | 2026-10-15 | inference
Tried: task 5.2 speed, `ladder-cpu`: yolo11s detect_batch (letterbox, forward, NMS) on 300 decoded data-mini frames, batch 1/4/8, PyTorch vs ORT FP32 vs ORT INT8, each step in its own process, 5 repetitions with rotated order, 15 s cooldown, 2 warmup batches; compose stack stopped, AC power, balanced power profile, browser open. Clean commit 9b71ac7, laptop.
Result (images/s, mean (std); batch-1 latency p50/p95): PyTorch 12.9 (0.5), 78/87 ms; ORT FP32 21.3 (1.5), 44/53 ms; ORT INT8 38.2 (0.8), 26/31 ms. Batch 4: 15.0, 22.3, 45.9; batch 8: 14.9, 21.0, 46.1. ORT FP32 is 1.65x PyTorch at batch 1 for no mAP change; INT8 1.8x ORT FP32 at batch 1 and 2.1x at batch 4. ORT uses 10 cores, PyTorch 8. INT8 gives 6% more raw detections per frame at conf 0.1 (30.4 vs 28.7). bench/results/2026-10-15-ladder-cpu-01M50QBZK10K8HA5XBGGFJ5R3G.json
Outcome: the best backend that passes D32 on this CPU is ORT FP32, and batching does not help it here (batch 4 +5%, inside the run-to-run std). The E-039 ramp ran yolo11n, so 5.5 needs the same ladder for yolo11n.

### E-046 | 2026-10-15 | inference
Tried: E-044 and E-045 repeated for yolo11n (the model of the compose worker and the E-039 ramp), scenario `ladder-cpu-n` (only the weights changed); INT8 with the same settings (122 nodes excluded, 87 of 88 convs INT8). Clean commit 5bd2499, laptop, compose stack stopped for the speed run.
Result (images/s at batch 1 / 4 / 8): PyTorch 26.5 (std 3.0) / 37.4 / 39.5; ORT FP32 48.8 (2.9) / 56.5 (0.9) / 54.3; ORT INT8 65.8 (0.3) / 80.3 / 82.2. Batch-1 p50/p95: 37/44, 20/23, 15/16 ms. ORT FP32 vs PyTorch: identical mAP (paired delta 0.00). INT8 vs ORT FP32, mAP@50-95: COCO 36.55 vs 37.85, -1.30 [-1.99, -0.52]; MOT17 38.90 vs 40.31, -1.41 [-1.72, -1.03]; fails D32 on both (drops above 1.0). bench/results/2026-10-15-ladder-cpu-n-01M50TMX23YZAMTNEX1DRAC2S1.json, eval/results/2026-10-15-detection-compare-yolo11n-int8-vs-onnx-01M50V86Q8430P31RD2B1CQ8PZ.json
Outcome: ORT FP32 is the 5.5 backend on CPU (1.84x PyTorch at batch 1); INT8 rejected, and for yolo11n the drop is outside noise on both sources. Unlike yolo11s, batch 4 adds 16% for ORT FP32 here (std 0.9), so 5.3 matters for yolo11n. One ORT FP32 batch-1 repetition ran at 43.6 against ~50 for the other four (rep 4, first in its order).

### E-047 | 2026-10-16 | inference
Tried: task 5.5 ramps, one variable at a time, clean commit 168a613, laptop as in E-039 (browser open): `ramp-cpu-pt` (same-session PyTorch baseline, batch 1, now through the batched worker loop), `ramp-cpu-ort` (ONNX Runtime FP32, batch 1), `ramp-cpu-ort-b4` (ORT, `infer_batch` 4); yolo11n, data-mini at 5 FPS per camera, 1 to 5 cameras, warmup 20 s, hold 90 s, 3 repetitions; each run pinned its worker with `expect_worker`.
Result: rated capacity 2 / 4 / at least 5 cameras (5 was the top step, so a lower bound). Ceiling: PyTorch 14.5 frames/s (std 0.1; E-039: 14.2), ORT 22.7 (0.7). Freshness p95 at the rated load 0.24 / 0.23 / 0.27 s; 0 lost in any step; harness_ok everywhere (loadgen under 0.05 cores). Infer stage p95 at 1 camera 74 vs 25 ms. The ORT worker occupies 9.2 cores at 1 camera (PyTorch 1.4) and 9.9 at saturation. bench/results/2026-10-15-ramp-cpu-pt-01M50WC3BVFQ08BEKDRWPKS9K3.json, 2026-10-15-ramp-cpu-ort-01M50YD7G3ASG4EN12Y9XM5P2A.json, 2026-10-16-ramp-cpu-ort-b4-01M5105QDGP84HKNTTMR82HGNJ.json
Outcome: the batched loop at batch 1 matches the old one (no regression). ORT alone doubles rated capacity; its ceiling is 1.56x PyTorch end to end against 1.84x model-only. Batch 4 never failed, so E-048 extends it.

### E-048 | 2026-10-16 | inference
Tried: `ramp-cpu-ort-b4-ext`, ramp-cpu-ort-b4 continued at 5 to 8 cameras, same worker and host, clean commit be6d039.
Result: rated capacity 5 cameras (3/3; freshness p95 0.28 s, backlog max 5 frames); 6 cameras fail every repetition. Ceiling 25.6 frames/s (std 0.5) at 6 cameras (25.4 / 24.9 at 7 / 8: throughput sags as the backlog grows; commit 42d270f's "25.3" is the mean over 6 to 8), infer p95 110 ms per 4-frame call at saturation; 0 lost. The 5-camera step repeats E-047's (25.0 vs 25.0 frames/s, p95 0.28 vs 0.27 s). bench/results/2026-10-16-ramp-cpu-ort-b4-ext-01M511Z4V0FE0MA4TK3YDAY1SD.json
Outcome: batch 4 adds 13% to ORT's ceiling (model-only: 16%) and 1 rated camera; against PyTorch the ceiling is 1.77x while rated capacity is 2.5x (5 vs 2 cameras), because whole-camera steps round the baseline down. The ceiling ratio is the fair summary.

### E-049 | 2026-10-16 | inference
Tried: ONNX Runtime's idle intra-op threads sleeping instead of busy-waiting (`detector.spin` false, new; `session.intra_op.allow_spinning` 0). First an ad hoc diagnostic (scratch script, yolo11n ORT FP32 batch 1 at 640x384 paced at 5 images/s, 2 runs each): 9.1 cores spinning vs 1.3 not; latency p50 19 vs 43 ms (threads wake per run); saturated 50 vs 51 images/s. Then the reported run: `ramp-cpu-ort-b4-nospin` (ramp-cpu-ort-b4 with spin off), 1, 3, 5, 6 and 7 cameras, 3 repetitions, clean commit 124158a.
Result: rated capacity 5 cameras (3/3), as with spinning; ceiling 26.3 frames/s (std 0.4) at 6 cameras vs 25.6 (0.5); worker cores 1.29 / 2.88 / 4.01 at 1 / 3 / 5 cameras (spinning: 9.17 / 9.49 / 9.85); freshness p95 0.24 / 0.24 / 0.40 s (spinning 0.23 / 0.23 / 0.28); infer p95 at 1 camera 49 vs 25 ms; 0 lost. bench/results/2026-10-16-ramp-cpu-ort-b4-nospin-01M513W7G2ACAK5X5PFR9WNXV8.json
Outcome: keep spin off for the deployed CPU worker (D5): same capacity (ceiling +3%, within the run-to-run spread) for 14% (1 camera) to 41% (5 cameras) of the cores, at +0.12 s freshness p95 at the rated load (SLO 2 s). Spinning only buys per-call latency, and the cores it burns are the ones ingest and Postgres share on one host.

### E-050 | 2026-10-16 | drift
Tried: task 9.1 check: `make demo` (4 cameras x 5 FPS for 180 s, data-mini, yolo11n ORT FP32 batch 4) with the new drift service (window 30 s, reference 60 s from each camera's first frame); maxima over the run read from Prometheus; noise-estimate cost timed ad hoc (300 calls on one 1280x720 frame).
Result: all drift gauges present for cam-001 to cam-004, telemetry check passed. Clean windows, max over the run (including the partly filled windows as the stream ends): |standardized shift| 0.44 (contrast, log_blur and noise on cam-001), class PSI <= 0.009, confidence KS <= 0.05. Inputs on the 300-frame clips (mot17-09-mini, mot17-04-mini) shift by exactly 0 in a full window: the clips loop every 10 s at 5 FPS, so a 30 s window holds the same frames as the reference; commons-roncesvalles (298 frames) |shift| <= 0.1. Noise estimate 0.93 ms per frame (input_stats 6.0 ms in all); the worker's stats stage mean was 4.7 ms in the demo. No results file (demo run).
Outcome: 9.1 done. Looping clips make clean runs unrealistically stable: 9.3 must calibrate false alarms on full-length clips (make data) and say that even those loop (17 to 35 s).

### E-051 | 2026-10-16 | drift
Tried: task 9.2 mAP impact, `make eval-drift` (bench/scenarios/drift-map-cpu.yaml): yolo11n ORT FP32 on detection subset v1, clean and under darken (gain 0.5, 0.25), blur (sigma 3, 6 px at 1280), noise (sigma 10, 25) at full strength on every image; paired bootstrap against clean. Code at 920d05d; the dirty flag is from RUNBOOK and ARCHITECTURE edits during the run (no code change). 13 min, laptop.
Result (mAP@50-95 change, COCO / MOT17): darken 0.5 -0.5 / -1.1; darken 0.25 -2.4 / -3.2; blur 3 -4.5 / -5.2; blur 6 -11.8 [-13.6, -10.0] / -17.1 [-18.6, -15.5]; noise 10 -3.1 / -0.8; noise 25 -12.9 / -6.0. Clean 37.9 / 40.3, equal to the PyTorch baseline (E-034). eval/results/2026-10-16-drift-map-cpu-01M5263733J106GJX458JF9108.json
Outcome: blur hurts the camera scenes most (-42% relative at sigma 6): the 9.5 target. Noise hurts COCO about twice as much as MOT17, because MOT frames are downscaled 2x by the letterbox, which averages noise; noise sigma is not comparable across input sizes.

### E-052 | 2026-10-16 | drift
Tried: task 9.2 end-to-end check, `python -m vms.loadgen run --scenario bench/scenarios/drift-darken.yaml --repeats 1` against the compose stack (ingest, D5 worker, drift monitor); gauges read from Prometheus at fixed offsets. Commit 2ae230c (dirty: BENCHMARKS.md edited during the run, no code change).
Result: onsets 161.0 s (cam 001, abrupt) and 178.4 s (cam 003, 120 s ramp), recorded in the results. Brightness shift: cam 001 +0.2 before, -19.8 after; cam 003 -8.4 at 240 s, -20.5 at 300 s, -26.1 at 355 s; controls 002 and 004 at most |0.33| on any input signal over the run. Darkening also moves the noise signal (-7.7 on cam 001: the estimate scales with the gain); confidence shift stays within 0.06 and detections per frame within -0.44, consistent with gain 0.3 costing little mAP (E-051). 7000/7000 frames accepted, generator_ok, loadgen 0.10 cores; 1,902 frames rendered, render p50 13 ms, p99 30 ms. bench/results/2026-10-16-drift-darken-01M5264GKZ5FKRQ5VBMQVPW0DS.json
Outcome: injection and monitor work end to end. Signals are not one-to-one with drift kinds (darkening also reads as less noise), which 9.3 must keep in mind when stating which signal catches which drift.

### E-053 | 2026-10-16 | drift
Tried: task 9.3 data: `bench/scenarios/drift-{clean,darken,blur,noise,shift,occlusion,classmix}.yaml` (4 cameras x 5 FPS x 360 s x 3 reps each, full MOT17 clips, onset 120 s + U(0, 60) s, camera 0 abrupt, camera 2 120 s ramp) against the compose stack (D5 worker, yolo11n ORT b4), from a clean worktree at ebcdf9f, laptop.
Result: all 7 runs clean (0 failed, generator ok); but the first scoring found no processed frames for shift, occlusion and classmix: the worker held 16 leases (max_cameras) of the earlier runs' idle cameras and never took new ones. Fixed in 5b92a28 (idle cameras released after 60 s); the 64,800 pending frames were then processed by the fixed worker (~22 frames/s). Per-frame statistics and detections do not depend on processing time, so the stored data are what a live run would have stored.
Outcome: runs bench/results/2026-10-16-drift-*-01M52{79Q6,8ASQ1,9BW7N,ACYGQ,BE1GW,CF3CE,DG5JA}*.json; a real worker bug found and fixed (with tests). The first score file (empty streams counted as misses) was discarded; scoring now refuses a camera without processed frames.

### E-054 | 2026-10-16 | drift
Tried: `make drift-score` on E-053 (84 streams, 36 drifted, 4.16 clean camera-hours): fixed threshold (h 2 to 100) vs CUSUM (k 0.5/1/2/4 x h 2 to 300) on calibrated z, leave-one-repetition-out operating points. Clean commit 36adf56 (after the E-055 fix; identical to the score at fec2e94, offline windows are all full).
Result: out of sample, threshold (h=3 in every fold) 36/36 caught, 0 false alarms (< 0.72/cam-h), delay median 4.8 s, p90 18.1, max 33.1; CUSUM (k=2 h=2, k=1 h=3, k=2 h=2) same delays, 2 false alarms (0.48/cam-h). In sample threshold h=2: 489 false alarms. Abrupt delays = gap to the next 5 s update for every kind; gradual: darken 11-13 s, blur 16-18, occlusion 13-17, noise 31-33, shift 2-6. Class mix fired PSI first 6/6. bench/results/2026-10-16-drift-detect-cpu-01M52J2GYHBMPAMRVCQW19GZPA.json
Outcome: threshold kept, deployed at h=10 for margin (0 misses, median 5.6 s, p90 28.1 s) (D34). CUSUM gains nothing on drifts this large; false-alarm rate is a lower bound (looping clips).

### E-055 | 2026-10-16 | drift
Tried: live alert check of the D34 operating point (threshold h=10, calibration in configs/drift.yaml): `python -m vms.loadgen run --scenario bench/scenarios/drift-darken.yaml --repeats 1` against the compose stack; alarm gauges and `ALERTS{alertname="VmsCameraDrift"}` read from Prometheus at 5 s steps. First at 00d74b8's parent tree (dirty: docs), then after the fix, clean 00d74b8.
Result: first run: class PSI alarmed on clean cameras 003 and 004 10 s into the run, and confidence W1 on 004 after the stream stopped: partly filled windows, which calibration never saw (fixed in 36adf56: the detector skips windows spanning < 90% of window_s). Second run: no alarm on the clean cameras; abrupt camera (onset 161.0 s) alarm at 170 s, alert firing at 175 s (14 s after onset); gradual camera (onset 178.4 s, 120 s ramp) alarm and alert at 205 s (27 s). bench/results/2026-10-16-drift-darken-01M52HKEBGAMHHYCZ2R3KD8GX9.json, -01M52J3HTRQP1SXHNAX5YSJPWQ.json
Outcome: alerts work end to end; live onset-to-alert is the offline delay plus settle_s (3 s), the 5 s scrape and the 15 s rule evaluation (about 10 s more for abrupt drift).

### E-056 | 2026-10-16 | drift
Tried: task 9.4 live check of the Drift dashboard (new band, detector-statistic and h gauges; onset annotations from loadgen): `python -m vms.loadgen run --scenario bench/scenarios/drift-darken.yaml --repeats 1` against the compose stack, alarms and `ALERTS` read from Prometheus at 5 s steps, dashboard screenshot. Code at f73ac3b plus the uncommitted 9.4 changes (dirty flag set). Three runs: the first overlapped `make check` on the same laptop, which starved the worker (13 vs 20 frames/s offered, backlog 1,352, windows emptied): discarded as evidence. The second and third ran alone, the third after fixing the alarm counter (below).
Result (third run): 7000/7000 frames accepted, backlog max 31, fewest frames per window 107 to 150; both onsets posted to Grafana (2 posted, 0 failed). Abrupt camera (onset 161.0 s): alarm and alert at 175 s (14 s; second run 9 s alarm, 19 s alert). Gradual camera (onset 178.4 s, 120 s ramp): alarm 210 s (32 s), alert 220 s (42 s). Clean cameras 002 and 004: no alarm. bench/results/2026-10-16-drift-darken-01M52R5FSWXW01XVF1P4QMNAHV.json (starved), -01M52SKB1KFGEKB9F4D4JSPWEJ.json, -01M52T1N0E6YXABBRHZ9D6YG7B.json; docs/images/grafana-drift.png
Outcome: dashboard and markers work end to end; delays in line with E-055. Found: `vms_drift_alarms_total` series were created at the first alarm, so `increase()` missed it ("alarm onsets in range" read 1 for 2 alarming cameras); the counters now start at 0 per channel (test added). A region annotation per alarm shaded every panel for the whole drift; the dashboard now marks alarm onsets only.

### E-057 | 2026-10-19 | inference
Tried: task 5.3 batch-size sweep, `make bench SCENARIO=bench/scenarios/sweep-cpu-ort-n.yaml` (yolo11n, ONNX Runtime FP32, spin off as deployed; batches 1 to 32, 480 data-mini frames, 3 repetitions), laptop i7-1355U with the app containers paused. A first run on a dirty tree (sweep code uncommitted) is discarded, not reported: 43 / 52 / 52 / 50 / 48 / 46 images/s.
Result (clean aa73fac), images/s (std) and batch p50/p95: b1 44.0 (1.1) 22/32 ms; b2 47.5 (4.2) 41/59; b4 50.3 (2.5) 77/101; b8 47.9 (2.9) 168/190; b16 47.4 (1.3) 330/372; b32 44.4 (0.8) 719/757; 6.0 to 7.3 cores. bench/results/2026-10-19-sweep-cpu-ort-n-01M5A24RZY0Q3Z01S0BAXY3JKZ.json
Outcome: throughput plateaus at 2 to 4 (+8 to 14% over batch 1, within run-to-run spread of each other) and falls past 8 (cache pressure); the rule (smallest batch within 5% of the best, p95 under 25% of the 2 s SLO) picks 4, the deployed infer_batch (E-048). Same sweep on AWS next (D35).

### E-058 | 2026-10-19 | cloud
Tried: first Runpod GPU session (task 2.4, D37): `scripts/runpod_session.sh run --yes bench/scenarios/worker-model-gpu-n.yaml`, clean commit 8eb2005. Preflight first: L4 (the planned type) and then A40 had no stock in either cloud; the A40 (user's choice) came back in stock 25 min later and the session launched on the first poll.
Result: 1x A40 secure, EU-SE-1, driver 580.159.04 (host CUDA 13.0); pod created to deleted in 199 s billed (setup ~2.5 min: uv, CUDA torch 2.14.1+cu126, weights, ONNX export), $0.033. Self-delete timer armed with the pod-scoped key (`key=yes`; the delete itself was not exercised); teardown verified, no pods left. yolo11n model-only: 50.1 frames/s (std 0.47), decode/stats/infer p50 3.8/7.6/8.4 ms, total p95 20.9 ms, 23.2 detections/frame (CPU laptop: 18.8 frames/s, infer p50 40 ms; same detections/frame). bench/results/2026-10-19-gpu-session-01M5A8JBB6JEE5V6PA674W0CRT.json, -worker-model-gpu-n-01M5A8QXX03BK69Q5XKAWV959X.json
Outcome: the driver works end to end; 2.4 done. On the GPU the CPU stages (decode + stats, 11.4 ms) outweigh the forward pass, so the GPU ladder (5.2) is reported model-only. Next: session 2 (ladder + sweep).

### E-059 | 2026-10-19 | cloud
Tried: first AWS CPU session (D35) on c7i.2xlarge (Xeon Platinum 8488C, 4 cores / 8 vCPU, 16 GiB, us-east-2), clean commit b349c1b: `ramp-cpu-ort-b4-nospin`, `query-load-cpu`, `ladder-cpu-n` and `sweep-cpu-ort-n`, unchanged from the laptop runs (same stack, worker D5, data-mini).
Result: 71 min billed, $0.43, teardown verified. Rated 3 cameras (3/3; freshness p95 0.24 s), 5 fail; worker ceiling 23.8 frames/s (laptop 26.3) at 1.8 worker cores (laptop 4.0); infer p95 122 ms per 4-frame call (laptop 117). Query load at 7 frames/s ingest: p95 30 / 36 / 53 / 129 ms at 1 / 2 / 4 / 8 clients, 141 q/s max (laptop 159). Ladder (spin on) ORT FP32 49.4 / 60.2 / 59.6 images/s at batch 1 / 4 / 8; PyTorch 31.6 / 48.1 / 53.8; INT8 54.7 / 67.2 / 68.7. Sweep (spin off): b1 34.3, b4 51.0, b8 54.8 (0.3), b16 53.7, b32 51.1. bench/results/2026-10-19-{ramp-cpu-ort-b4-nospin-01M5A488HBYMH8CPJAFX5XQZVJ,query-load-cpu-01M5A64TVCB1TDDT4QPVBSCWHB,ladder-cpu-n-01M5A751S20KDY1WDHSB0K8YQ6,sweep-cpu-ort-n-01M5A7P0RP7ASGPVFVB5V0QB0R,cpu-session-01M5A3X28Z8AS5SBWATZZ3HJR5}.json
Outcome: an 8-vCPU server rates 3 cameras for $0.37/h, i.e. about $0.12 per camera-hour of compute. On 4 physical cores the forward pass leaves less headroom than on the laptop's 10, so batching matters more: the sweep's operating point is 8 here vs 4 on the laptop; spinning costs 30% at batch 1 (49 vs 34 images/s) and 15% at 4. The deployed default stays 4 (laptop and compose, E-048); a c7i deployment would set WORKER_INFER_BATCH=8 (not ramp-tested).

### E-060 | 2026-10-19 | drift
Tried: task 9.5 on AWS c7i.4xlarge (16 vCPU, $0.73/h), `bench/scenarios/finetune-blur-cpu.yaml` (D36), clean commit b440130: two fine-tunes of yolo11n on training list v1 (blur-mix: 787 of 1,567 images blurred, mean sigma 4.5; control: none), 10 epochs, then 12 eval runs.
Result: training 977 s and 1,029 s (16.0 and 15.2 images/s incl. one validation pass, 8 torch threads), $0.20 and $0.21; all 12 evals finished, but the scenario hit the session's 1 h scenario timeout before writing the summary (its raw detections stayed on the instance). Session 66 min billed, $0.80, teardown verified. bench/results/2026-10-19-finetune-blur-cpu-{blur-mix-01M5A9CNW3HAERH1M2GSB1EBH9,control-01M5AACJBSVWAPRAK2XSZZTRW9}.json, bench/results/2026-10-19-cpu-session-01M5A809RPZ20Q13MT08V0EEPX.json, eval/results/2026-10-19-finetune-blur-cpu-*-01M5AA*.json
Outcome: added `python -m vms.finetune evaluate` (score copied-back weights, checked against their records), copy-back of the eval's raw detections, and a RUNBOOK note to raise `VMS_TIMEOUTS__SCENARIO_S` for this scenario.

### E-061 | 2026-10-19 | drift
Tried: the E-060 weights re-scored on `laptop` with `python -m vms.finetune evaluate` (same commit plus the evaluate code, dirty), paired bootstrap (2,000 resamples) on detection subset v1. Weights DVC-tracked (`models/finetune/*.pt.dvc`).
Result: every mAP equals the AWS eval to the last digit (CPU inference is deterministic across hosts). mAP@50-95 change vs base, COCO / MOT17: blur-mix clean -7.0 [-8.2, -5.6] / -2.6 [-3.6, -1.6]; blur 6 -1.5 [-2.9, -0.4] / **+1.6 [+0.7, +2.7]**; noise 25 -8.7 / -3.4. Control clean -5.6 / -3.2; blur 6 -4.7 / -2.8. Blur-mix vs control: blur 6 +3.2 [+1.6, +4.6] / **+4.4 [+3.7, +5.3]**, blur 3 +1.6 / -1.2, clean -1.4 / +0.6. bench/results/2026-10-19-finetune-blur-cpu-01M5ADGCCBWNYCCAZ5RMPVKG17.json
Outcome: negative overall. The fine-tune recipe itself costs 3 to 6 points everywhere (the control, without blur, loses on every condition, including MOT17 clean, where adapting to MOT's box convention was expected to help); blur augmentation buys 4.4 points on the target drift against that control, which leaves blur-mix only +1.6 above the untouched model on MOT17 blur 6 and worse on everything else. Not deployed. Likely causes, untested: 800 COCO images for 80 classes (forgetting), Ultralytics' 3-epoch warmup with bias lr 0.1 on a pretrained head, all layers trainable. Next step if revisited: one recipe change at a time against this control (frozen backbone, no warmup), not a sweep.

### E-062 | 2026-10-19 | inference
Tried: GPU session 2 (D38), `ladder-gpu-n` + both TensorRT sweeps on a Runpod A40, clean commit 2c6ea99.
Result: setup reached the engines in ~6 min (torch 2.14.1+cu130, onnxruntime-gpu 1.31.0 with the CUDA provider, TensorRT 11.3.0.99); the FP16 engine built in ~2 min; the INT8 engine failed to parse: `DequantizeLinear (convertAxis): Axis must be in the range [0, nbDims (0)]` on `/model.10/m/m.0/attn/Constant_6`, a scalar constant (the attention scale) that ORT's quantizer had wrapped in Q/DQ. Setup failed, so no scenario ran; pod deleted, teardown verified, 410 s billed, $0.069. bench/results/2026-10-19-gpu-session-01M5ABS2E4B4PXXTTC3585PWQ1.json
Outcome: the TensorRT INT8 model now quantizes Conv only, with FP32 biases (`int8_tensorrt.op_types: [Conv]`, `quantize_bias: false`): locally 87 int8 conv weights, no scalar or int32 Q/DQ (unit test added). Engines are built one precision at a time and a failing ladder step is recorded without stopping the other steps, so one broken rung no longer costs a session.

### E-063 | 2026-10-19 | cloud
Tried: GPU session 2 again after the E-062 fix, from clean commit 21c9b52 (worktree; the shared checkout had another session's uncommitted work). One launch first raced the stock (preflight passed, the session's own preflight a second later found no A40: nothing created).
Result: the pod reached RUNNING but its 22/tcp never got a public port, so direct SSH never appeared; the driver waited the whole running_s (900 s) before tearing down. Teardown verified, 905 s billed, $0.150. Data center not recorded (it was saved only together with SSH). bench/results/2026-10-19-gpu-session-01M5ACHRPBFSA4CFTHBP2RHJB0.json
Outcome: the driver now records the data center once the pod is RUNNING and gives direct SSH 180 s (`ssh_direct_s`), then fails with the pod's port mappings in the error (unit test added); the relaunch loop retries that failure on another host, at most 3 times. Runpod spend so far 2026-10-19: $0.25.

### E-064 | 2026-10-19 | inference
Tried: GPU session 2 (tasks 5.2, 5.3; D38) on runpod-a40 (EU-SE-1, driver 595.91.07, host CUDA 13.2), clean commit d0d75d9: `ladder-gpu-n` (PyTorch, ORT CUDA, TRT FP16, TRT INT8; batch 1/4/8, 5 reps; mAP per rung and paired compares on the pod) and `sweep-gpu-trt-{fp16,int8}-n` (batch 1 to 32, 3 reps). Both engines built (FP16 ~2 min, INT8 ~3 min); 1,540 s billed, $0.255, teardown verified, self-delete timer armed.
Result: batch 1 images/s: PyTorch 120.0, ORT 172.7, TRT FP16 248.6, TRT INT8 183.7. mAP@50-95 COCO/MOT17: PyTorch 37.9/40.3, ORT identical (delta <0.02), TRT FP16 -0.06 [-0.23, +0.04] / -0.03 [-0.19, +0.05] vs ORT, TRT INT8 -2.15 [-2.91, -1.18] / -2.48 [-2.99, -1.88] vs FP16 (fails D32). Batch 8: PyTorch 386 images/s, numpy rungs 271-309 with batch 4 slower than batch 2 and stds up to 67. Sweep FP16 peak 317 images/s at batch 2. bench/results/2026-10-19-ladder-gpu-n-01M5AE3XVVKDKNGBWASF8NFQJF.json, -sweep-gpu-trt-fp16-n-01M5AEYG19BXE9F9D2VPKFDN14.json, -sweep-gpu-trt-int8-n-01M5AF0S3BSPXAK43X1R0KW6VS.json; eval/results/2026-10-19-ladder-gpu-n-*.json
Outcome: quality is settled: FP16 is free, TensorRT INT8 is rejected (D32). Speed at batch > 1 is suspect for the numpy rungs (non-monotonic, high variance, PyTorch with GPU pre/post scales fine): likely OpenCV/NumPy threads sized to the host's 96 threads inside the pod's CPU quota, which the session did not record. Next: record the pod's vCPUs, size thread pools to them, re-run the batch columns and sweep before choosing a GPU operating point.

### E-065 | 2026-10-19 | inference
Tried: E-064's speed side again with one change, ONNX Runtime and OpenCV thread pools capped at the container's CPU quota (`vms.common.cpus`, clean commit 7aa569c): `ladder-gpu-speed-n` (PyTorch, ORT CUDA, TRT FP16; batch 1/2/4/8; 5 reps) and `sweep-gpu-trt-fp16-n` on runpod-a40 (CA-MTL-1, driver 580.178.04). Pod quota recorded this time: 7 CPUs by cgroup, 9 vCPUs and 50 GB by the API. 852 s billed, $0.141, teardown verified.
Result: same shape as E-064. Images/s at batch 1/2/4/8: PyTorch 108/187/265/366, ORT 141/244/259/269, TRT FP16 234/254/201/257 (batch 4 p50 19.8 ms vs 7.1 at batch 2; std up to 64). Sweep FP16 184/283/190/308/282/204 for batch 1 to 32. Numpy-path rungs use 1.6 to 2.6 cores, under the quota. bench/results/2026-10-19-ladder-gpu-speed-n-01M5AGHWFY2DENF038W8E674B9.json, -sweep-gpu-trt-fp16-n-01M5AGSZ5VGHTJYQP9GR277HC5.json
Outcome: hypothesis rejected: the CPU quota is not what limits batches > 1. What holds across both runs: batch 1 (TRT FP16 p50 3.9/4.0 ms, 2.1x PyTorch), and that rungs doing letterbox and NMS in numpy plateau at 250 to 300 images/s while PyTorch's GPU-side pre/post scales to 366 to 386. The TRT batch-4 dip is unexplained. The quota cap stays (correct sizing, recorded in results) but bought nothing measurable here. Next, if pursued: per-stage timing in detect_batch (letterbox, forward, NMS) on the pod before any more GPU sweeps.

### E-066 | 2026-10-23 | llm
Tried: task 8.1 GPU session, attempt 1 (D40), `bench/scenarios/llm-serve-a40.yaml` on runpod-a40 (CA-MTL-1), clean commit 850aaf2: `setup llm` (vLLM 0.31.0 venv, Ollama 0.40.2) in ~2 min, then Qwen3.5-9B on vLLM.
Result: every vLLM server failed at warmup: FlashInfer JIT-compiles its top-k/top-p sampler and found no nvcc (`Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist`; the pod image has no CUDA toolkit). Seen over SSH in the engine log after two servers failed; session stopped by hand (Ctrl-C) to save the rest, teardown verified, 595 s billed, $0.101. bench/results/2026-10-23-gpu-session-01M5M79GH2QWWJ4GNFRSV4Q29F.json
Outcome: `VLLM_USE_FLASHINFER_SAMPLER=0` for every vLLM server (`configs/serving.yaml` `vllm.env`; unit test); vLLM then uses its PyTorch sampler, irrelevant to greedy decoding. Relaunched.

### E-067 | 2026-10-23 | drift
Tried: task 9.6 (D39, chosen before running): 9.5's recipe with the backbone frozen (`freeze: 11`), `bench/scenarios/finetune-blur-freeze-cpu.yaml` on aws-c7i-4xl, clean commit a05e3b3; blur-mix and control, same list, seed and 12 evals.
Result: training 641 s / 690 s (1.5x faster than 9.5). vs base, COCO / MOT17 mAP@50-95: control clean -3.0 [-3.8, -2.2] / +0.5 [-0.2, +1.3] (9.5: -5.6 / -3.2); blur-mix clean -3.4 / -0.0, blur 6 -1.2 / +0.6 [-0.3, +1.5]. blur-mix vs control, blur 6: +0.5 / +1.2 [+0.5, +1.9] (9.5: +3.2 / +4.4). This control vs 9.5's control, clean: +2.6 [+1.4, +3.9] / +3.7 [+3.0, +4.4]. Session 3,323 s, $0.670, teardown verified. bench/results/2026-10-23-finetune-blur-freeze-cpu-01M5M8XS8CZE80DDDR5ST935H8.json, eval/results/2026-10-23-finetune-control-freeze-vs-9.5-01M5M92VCRS3SEKY1YYBMRB1NF.json
Outcome: freezing removes the MOT17 loss and halves the COCO one, so backbone forgetting was about half of 9.5's problem; it also takes away most of the blur gain. Still negative for deployment (D39 rule); weights DVC-tracked, worker unchanged.

### E-068 | 2026-10-23 | llm
Tried: task 8.1 GPU session (D40) after the E-066 fix, `bench/scenarios/llm-serve-a40.yaml` on runpod-a40 (CA-MTL-1), clean commit ba1940e: 6 servers (Qwen3.5-9B, -9B no prefix cache, -4B, Qwen2.5-7B-Instruct, -7B no prefix cache on vLLM 0.31.0; Qwen2.5-7B FP16 on Ollama 0.40.2), load at concurrency 1/4/16 x 3 reps, structured outputs on/off, 3 dev recordings.
Result: all servers started (vLLM 42 to 196 s incl. download). Qwen3.5-9B baseline: c=1 0.41 req/s, TTFT p50 162 ms, decode 32.7 tok/s; c=16 3.90 req/s, p95 5.9 s. Prefix cache off: TTFT 728 ms, c=16 1.24 req/s, p95 22.4 s (hit rate on: 89%; Qwen2.5 99%). Structured off: same speed, validity 92 to 94% vs 88 to 89%. Qwen3.5-4B: c=16 6.70 req/s. Ollama vs vLLM (Qwen2.5-7B): c=16 1.07 vs 5.38 req/s, validity 52% vs 81%. Client CPU < 6% of a core. 3,376 s billed, $0.573, teardown verified. bench/results/2026-10-23-llm-serve-a40-01M5MB6P60PMMMV2FJX8Z0PP91.json
Outcome: prefix caching kept on by default for any self-hosted planner; structured outputs kept (free, guarantees parseable JSON) though they do not raise semantic validity. Ollama's validity gap unexplained (template or default sampling options).

### E-069 | 2026-10-23 | query-eval
Tried: the 3.5 local-LLM dev comparison (moved there 2026-10-14) by replay: `make eval-replay` of E-068's recordings on vms_eval (laptop), dev split, unchanged harness.
Result: 0 replay misses each. Dev accuracy: Qwen3.5-9B 75.2% [67.9, 82.5], Qwen3.5-4B 75.2% [67.9, 82.5], Qwen2.5-7B 58.4% [50.4, 67.2]; DeepSeek (E-035) 94.2% [89.8, 97.8]. Top errors 9B: invalid plan after repair 13, should have abstained 8, wrong aggregation 7. eval/results/2026-10-23-structured-dev-replay-*.json
Outcome: hosted wins on dev with disjoint intervals; per 3.5's rule no local test run; D7 keeps DeepSeek. Qwen3.5-4B (same accuracy, ~2x faster) is the 8.3 LoRA base.

### E-070 | 2026-10-23 | lora
Tried: task 8.2 (D41): `make traindata-states`, `make traindata-generate` (DeepSeek, $3.00 cap, 15 questions per batch call, then the planner per question), `make traindata-filter`. Commits ba1940e (generation), filter on the same code.
Result: 268 batch calls (4 truncated JSON), 3,902 candidates, 4,528 calls, 95.9% prompt cache hits, $2.989. Kept 2,660 (68%). Largest drops: empty result 377, plan A invalid 280, frames with visual query 182 (no CLIP), results differ 179, leakage 139 (158 near-dup, 12 lexical, 2 template hits; 0 after drop). A = B plan for 88% of kept. bench/results/2026-10-23-traindata-v1-01M5MBBQMPWHDBJ1B1Y13S0SBS.json; train/plans.v1 (DVC)
Outcome: dataset v1 for 8.3. 20 questions per call truncated at 6,000 tokens in a dry run (hence 15 and 8,000). Abstentions are 21% of pairs (the dev split has 29%).

### E-071 | 2026-10-23 | lora
Tried: task 8.3 session 1 (D42), `bench/scenarios/lora-plan-a40.yaml` at commit 33de31c on runpod-a40 (CA-MTL-1): micro-batch 4 x accumulation 4, no gradient checkpointing. Two earlier placements in EU-SE-1 got hosts without a public IP (E-063's failure, $0.031 each) and one found no stock ($0); retried pinned to CA-MTL-1.
Result: all 5 runs failed with CUDA out of memory in the cross-entropy over the 248k-token vocabulary (~42 GB at micro-batch 4 with 2k-token examples); nothing to serve. 454 s billed, $0.075, teardown verified. bench/results/2026-10-23-lora-plan-a40-01M5MQ6ZBK82E5J20HK5YJC08X.json, -gpu-session-01M5MPSE6YBG311PMNXPQ92AJQ.json
Outcome: micro-batch 2 x 8 (same effective batch 16) and a probe that halves the micro-batch on OOM before training (commit c515d1f, unit tested); peak then 31 to 33 GB.

### E-072 | 2026-10-23 | lora
Tried: 8.3 session 1 again (commit c515d1f, CA-MTL-1): data size at rank 16 (500, all 2,527 train pairs, 1,000) under an 80-min training budget; then one vLLM server recording dev for the three adapters and the base on the full and compact prompts.
Result: train_s 449 / 2,272 / 899 s at ~1,970 tokens/s, peak 31.7 GB, no OOM halving; val loss (epoch 0, 2) 0.825 -> 0.0137 / 0.0053 / 0.0085. Rank 64 and 4 skipped as planned (41.8 min predicted, 13.4 left). Dev accuracy (replayed): base-full 75.2%, base-compact 20.4%, n500 74.5%, all-r16 84.0% [77.4, 89.8], n1000 74.5%. 4,848 s billed, $0.802, teardown verified. bench/results/2026-10-23-lora-plan-a40-01M5MW72R48ZCKHG15CY12MGZM.json, -gpu-session-01M5MQKTZ748HYDXP902G61P52.json
Outcome: training on all pairs is the first fine-tune to beat the base on dev; rank ablation continued in E-073.

### E-073 | 2026-10-23 | lora
Tried: 8.3 session 2, `lora-plan-a40-b.yaml` (commit f752784, CA-MTL-1): rank 64 and 4 on all pairs, and n1000-r16 again by accident of the plan (same seed and data: a run-to-run noise check); then all 6 adapters' dev recordings (session 1's pushed as .tar) and both base controls on one server; replayed on the laptop.
Result: dev accuracy: all-r4 84.7% [78.8, 90.5], all-r16 83.9%, all-r64 83.9% [78.1, 89.8], n1000-r16 78.1% (session 1's copy: 74.5%), n500 74.5%, base-full 74.5%, base-compact 19.7%. Re-served session-1 adapters gave the same accuracy as in session 1, base-full 1 question lower. Abstention correct r4/r16/r64 95.0/92.5/87.5%; false abstention 1.0/1.0/4.1%. Trainable params r4 5.3M (21 MB adapter), r64 85M. 6,895 s billed, $1.141, teardown verified. bench/results/2026-10-23-lora-plan-a40-b-01M5N2QY7YSASA28F4JTEGQZ24.json, -gpu-session-01M5MW85GEF097E394EZZFGWX1.json; eval/results/2026-10-23-structured-dev-replay-qwen3.5-4b-01M5N2V5590AR4ANA57HTQ2D6X.json to -01M5NB06FJS9BPP9J2FHW673XD.json (7 files)
Outcome: rank does not matter on dev between 4 and 64 (spread 0.8 points, under the 3.6 points between two identical n1000 runs); all-r4 chosen (best point estimate and abstention, smallest). Data size matters: all pairs +6 to +10 over 500 to 1,000.

### E-074 | 2026-10-23 | lora
Tried: the frozen test split (sha256 0330b3db...) once for the dev pick, `bench/scenarios/lora-plan-test-a40.yaml` (commit 2a9ea15, CA-MTL-1): all-r4 (pinned sha256 21412c6e...) and the base on the full prompt on one vLLM server; replayed on the laptop.
Result: 142 questions, 0 replay misses. all-r4 83.1% [76.8, 88.7] (expressible 79.4%, abstention 91.1%, false abstention 3.1%, invalid plan 11) vs base-full 78.9% [71.8, 85.2] (77.3%, 82.2%, 0%, invalid plan 13); DeepSeek (E-036) 95.8% [92.3, 98.6]. Recorded call p50 1.50 s vs 2.63 s. 671 s billed, $0.111, teardown verified. bench/results/2026-10-23-lora-plan-test-a40-01M5NBRPMC2QS6FSF853KGRA83.json, -gpu-session-01M5NB4JNRN6P3478CK6B086PG.json; eval/results/2026-10-23-structured-test-replay-qwen3.5-4b-01M5NBSJR43JKPQ7AA9KQ0RW5W.json, -01M5NBSNHYDADVTS6G9XY2XK7V.json
Outcome: the adapter's gain shrinks from +10 (dev) to +4.2 points on test, intervals overlapping (no paired test run): a likely but not shown gain, mostly from abstention; 1.75x faster per call through the shorter prompt. Still 13 points under the hosted planner, so D7 keeps DeepSeek pending 8.5. 8.3 Runpod total $2.19.

### E-075 | 2026-10-24 | llm
Tried: task 8.4 on one A40 (EU-SE-1), `bench/scenarios/llm-quant-a40.yaml`, clean commit 08c744e: Qwen3.5-4B BF16, RedHatAI GPTQ W4A16 and W8A8 checkpoints of the same weights (LLM Compressor; attention and MLP projections only), each with the pinned all-r4 adapter, and all-r4 merged into the BF16 base on the pod (FP32 merge, BF16 save, `scripts/lora_merge.py`); vLLM 0.31, prefix caching on, load at concurrency 1/4/16/32 x 3 reps, dev recorded per variant.
Result: all 4 servers and the merge ran (merge 36 s). Adapter, compact prompt, c=1 req/s and p95 / c=32 req/s and p95: BF16 0.92, 1.78 s / 11.20, 4.16 s; merged 1.05, 1.55 s / 12.67, 3.66 s; W4A16 1.37, 1.18 s / 13.06, 3.82 s; W8A8 1.12, 1.43 s / 13.07, 3.66 s. Decode at c=1: 50.5 / 58.6 / 79.1 / 63.4 tok/s. Weights 8.03 / 7.99 / 4.53 / 5.71 GiB; KV cache 788k / 803k / 880k / 849k tokens. Base full prompt c=1 p95: BF16 2.09 s, W4A16 1.04 s, W8A8 1.37 s. Session 2,663 s, $0.445, teardown verified. bench/results/2026-10-24-llm-quant-a40-01M5PZEPSC48GZTPADB095KA06.json, -gpu-session-01M5PX4DGN3R4E9RM0YK0M1YCK.json
Outcome: W4A16 is 1.5x the BF16 adapter's speed at c=1 and +17% at saturation; merging removes vLLM's LoRA overhead (+14% at c=1, +13% at c=32). Accuracy in E-076.

### E-076 | 2026-10-24 | query-eval
Tried: `make eval-replay` of E-075's 7 dev recordings, and the test recording of `llm-quant-test-a40.yaml` (W4A16 + all-r4, chosen for the D7 test run before any result; same session).
Result: 0 replay misses. Dev (n=137): BF16 base-full 75.2%, adapter 84.7% (= E-073); merged 83.9% (146 of 147 replies identical to the adapter's, 1 question differs); W4A16 base-full 70.1%, adapter 81.8%; W8A8 base-full 73.0%, adapter 83.2%. Test (n=142): W4A16 + all-r4 82.4% [76.1, 88.0] (abstention 86.7%, invalid plan 11) vs BF16 adapter 83.1% (E-074). eval/results/2026-10-24-structured-dev-replay-*.json, -structured-test-replay-qwen3.5-4b-w4a16-01M5PZQ3154YACNZBERSBEFZ5B.json
Outcome: 4-bit costs the adapter 2.9 dev points and 0.7 [-3.5, +1.4] paired test points (E-078), inside noise; it costs the base on its full prompt 5.1 dev points. Merged equals the adapter. D43.

### E-077 | 2026-10-24 | llm
Tried: hosted DeepSeek `deepseek-flash` under E-075's load profile from the laptop (`bench/scenarios/llm-load-hosted.yaml`, full prompt, json_object, dev questions), for 8.5.
Result: c=1 0.59 req/s, p50 1.66 s, p95 2.58 s (worst rep 2.77); c=32 11.66 req/s, p95 3.13 s (worst 3.40); 3 empty replies of 384 at c=32 (0.8%), none below. Prompt tokens served from DeepSeek's cache: 96.5%. 2.93M prompt, 116k completion tokens: $0.19 at configs/eval.yaml prices. bench/results/2026-10-24-llm-load-hosted-01M5PXAMV2Q6DSTWHBGB99BFJ6.json
Outcome: hosted latency is flat from 1 to 32 concurrent (the provider scales), but its floor (p95 ~2.6 s, internet path included) is above the 1.5 s query SLO.

### E-078 | 2026-10-24 | llm
Tried: task 8.5, `make bench SCENARIO=bench/scenarios/d7-compare.yaml` (vms.serving.compare, commit 2329705): the 4 D7 options on test accuracy, load and cost; A40 at $0.601/h, 730 h/month.
Result: test: hosted 95.8%, base few-shot 78.9%, LoRA 83.1%, LoRA W4A16 82.4%; paired: hosted - LoRA +12.7 [+7.0, +19.0], LoRA - base +4.2 [-0.7, +9.9]. p95 at c=1: 2.77 / 2.20 / 2.02 / 1.33 s (only W4A16 meets 1.5 s). USD per 1k questions: hosted 0.294 (all-miss 1.60); GPU at saturation 0.022 / 0.016 / 0.014, fixed $439/month. Break-even 1.49M questions/month (~2,000/hour sustained). bench/results/2026-10-24-d7-compare-01M5PZTAAM7FWWA8C5WMB4AYF2.json
Outcome: D7 accepted: hosted. Self-hosting wins on cost only above ~1.5M questions/month and loses 13 points of accuracy at any volume; W4A16 + adapter is the measured fallback (D43).
