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

_No entries yet._
