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
