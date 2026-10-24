# Roadmap

What is built, with its evidence, and what comes next. Numbers live in [BENCHMARKS.md](BENCHMARKS.md), reasons in [DECISIONS.md](DECISIONS.md), every attempt (including failed ones) in [EXPERIMENTS.md](EXPERIMENTS.md).

## Built

### Foundation
- Repo, tooling and CI: uv, ruff, mypy `--strict`, pytest (unit, integration against the compose stack, GPU-marked), pre-commit, DVC with a local remote, GitHub Actions running lint, typecheck, unit and integration tests and the CPU demo.
- Compose stack: Postgres + pgvector, SeaweedFS (S3 API, D12), OpenTelemetry collector, Prometheus, Tempo, Grafana. Redis, MLflow and W&B were removed for lack of a measured need (D16).
- Shared library: layered YAML + environment config, structured JSON logs with trace IDs, OpenTelemetry setup, ULIDs, an injectable clock.

### End-to-end pipeline (CPU)
- Dataset scripts with pinned checksums and license notes (MOT17, MOT20, Wikimedia Commons; VIRAT opt-in).
- Load generator: N virtual cameras at fixed FPS from replayed clips, with seeded drift injection; its own ceiling measured (E-009 to E-011).
- Ingest: REST, JPEG to the object store, frame rows to Postgres, idempotent on `(camera_id, run_id, seq)` (D17, D18).
- Worker: Postgres queue with camera leases and `SKIP LOCKED`, fenced completion, YOLO11 detection, ByteTrack tracking and zone enter/exit events (D20, D21).
- Query service: natural language to a typed `QueryPlan`, validation and repair, a plan-to-SQL compiler, a sqlglot SQL validator, read-only execution, a query log (D13, D14, D22).
- UI (vanilla TypeScript): live camera wall with boxes and zones, question box with charts, tables and frame thumbnails, plan and SQL panel; a zone editor (D23).
- `make demo`: the whole pipeline on CPU with a canned planner, also run in CI (D24).

### Observability and benchmarking
- Traces across the queue and every query stage; Prometheus metrics for freshness, throughput, backlog, stage latencies, tokens (D25).
- Five Grafana dashboards generated from Python and provisioned at startup (D26).
- Benchmark runner: scenario YAML, warmup excluded, repetitions, steady-state windows judged from Prometheus, results JSON with git SHA, config, hardware and seeds (D27).
- Baseline: rated capacity 2 cameras on the laptop CPU, ranked bottlenecks from py-spy profiles (E-039 to E-042).

### Evaluation
- Structured-query eval set: 279 questions over a synthetic frozen database, two independent golds per question, frozen test split, red-team subset (D29).
- Scorer: execution accuracy with bootstrap intervals, a first-difference error taxonomy, attack success as observed harm, a gold-plan self-test in CI (D30).
- Detection eval: a frozen COCO + MOT17 subset, pycocotools-exact mAP, bootstrap intervals, paired comparisons (D31).
- Baseline: 95.8% [92.3, 98.6] test accuracy with a hosted planner, 0 red-team breaches (E-036).

### Model internals
- ONNX export with parity tests (E-037).
- Precision ladders on CPU (PyTorch, ONNX Runtime FP32, INT8) and GPU (PyTorch, ONNX Runtime, TensorRT FP16, INT8), each with mAP; INT8 rejected by a rule set before the first INT8 number (D32, D38).
- Batch-size sweeps on CPU and GPU.
- Optimized worker (ONNX Runtime FP32, cross-camera batch 4, no thread spinning): rated capacity 2 -> 5 cameras (D5, E-047 to E-049); 3 cameras and $0.12 per camera-hour on an AWS c7i.2xlarge (E-059).

### Drift detection
- Per-camera input and output statistics against a calibrated reference (D33); drift injection with seeded onsets and mAP under each drift (E-051).
- Change detection scored on delay against false-alarm rate, fixed threshold vs CUSUM, out of sample (D34, E-053 to E-055); dashboard and runbook procedure.
- Detector fine-tunes against blur, each against a no-blur control: both negative and not deployed (D36, D39, E-060, E-061, E-067).

### LLM serving and fine-tuning
- vLLM serving benchmarks on a rented A40: prefix caching, structured outputs, Ollama comparison (D40, E-068, E-069).
- Synthetic planner training data with execution-agreement filtering and leakage checks (D41, E-070).
- LoRA on Qwen3.5-4B with data-size and rank ablations (D42, E-071 to E-074).
- Quantized (GPTQ W4A16, W8A8) and merged serving (D43, E-075, E-076).
- Planner strategy decided on test accuracy, p95 under load and cost per 1,000 questions: the hosted model (D7, E-077, E-078).

### Cloud sessions
- Scripted, tagged, self-terminating sessions with verified teardown: AWS EC2 for CPU (D35), Runpod pods for GPU (D37). About $6 of rented compute in total (BENCHMARKS, Cost).

## Next

- **Demo video** (2 minutes): camera wall, a few questions with the plan and SQL panel, the pipeline dashboard under load, a drift alert firing.
- **Fresh-clone reproduction** on a clean machine following only the README and RUNBOOK.
- **GPU end to end**: the stack on a GPU host (a full VM or a single-image variant for pods) for a GPU ingest ramp, a GPU worker profile and rated capacity on GPU. Today the GPU numbers are model-level only.
- **Storage benchmarks**: partitioned Postgres vs TimescaleDB vs ClickHouse at 10M and 100M detections, with plan-to-SQL result equivalence across backends (D1); pgvector vs Qdrant for filtered vector search (D2).
- **Visual search**: CLIP embeddings with a sampling policy (D8), hybrid structured + visual queries sharing the `QueryPlan`, and a retrieval eval (recall@k, nDCG).
- **Queue and transport**: Redis Streams vs Kafka keyed by camera (D3), gRPC ingest (D4), reliability tests (redelivery, backpressure, dead letters).
- **Kubernetes**: Helm chart with a kind smoke test in CI, KEDA autoscaling of workers on backlog, capacity and chaos tests (D6).
- **Cloud at scale**: Terraform for the full AWS stack, a full-scale benchmark and cost per camera-hour.
- **Query engine**: free-form SQL as an alternative engine compared on the same eval (D11), few-shot retrieval, result caching; the self-hosted planner live behind the query service (D43).
- **CD with an eval gate**, and an orchestrated retraining loop for the detector (D10).
- **Pre- and post-processing on the GPU** (letterbox and NMS), which cap the exported GPU backends at 250 to 300 images/s (E-065).

## Task index

Docs, code comments and commit messages cite work by task number. The numbers are stable.

| Task | Work | Task | Work |
|---|---|---|---|
| 0.1 | Repo scaffold and tooling | 3.1 | Structured-query eval set |
| 0.2 | Docs skeleton, decisions D1 to D11 | 3.2 | Scorer: accuracy with CIs, error taxonomy |
| 0.3 | Shared library | 3.4 | Detection eval subset |
| 0.4 | CI | 3.5 | Baseline eval numbers |
| 0.5 | Compose stack | 5.1 | ONNX export and parity |
| 0.6 | Query assets: QueryPlan, SQL validator, few-shots | 5.2 | Precision ladder |
| 0.7 | Stack trim (Redis, MLflow, W&B) | 5.3 | Batch-size sweep |
| 1.1 | Dataset scripts | 5.5 | Optimized worker, re-ramp (D5) |
| 1.2 | Load generator | 8.1 | vLLM serving |
| 1.3 | Ingest | 8.2 | Planner training data |
| 1.4 | Schema and migrations | 8.3 | Planner LoRA |
| 1.5 | Worker | 8.4 | Quantized and merged serving |
| 1.6 | Tracking and zone events | 8.5 | Planner strategy (D7) |
| 1.7 | Query service | 9.1 | Drift statistics |
| 1.8 | UI | 9.2 | Drift injection and mAP under drift |
| 1.9 | `make demo` | 9.3 | Change detection: threshold vs CUSUM |
| 2.1 | Tracing and metrics | 9.4 | Drift dashboard and runbook |
| 2.2 | Dashboards | 9.5 | Detector fine-tune against blur |
| 2.3 | Benchmark framework | 9.6 | Fine-tune follow-up (frozen backbone) |
| 2.4 | Rented GPU sessions | 11.1 | README |
| 2.5 | Ingest ramp and query load | 11.2 | Technical write-up |
| 2.6 | Baseline bottleneck analysis | | |
