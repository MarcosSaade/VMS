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
