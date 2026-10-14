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
