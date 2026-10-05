# Agent Instructions

You are building a scalable, production-style Video Management System that demonstrates ML systems engineering: benchmarked design decisions, inference optimization, load testing, observability, drift detection and LLM serving. The work is judged by the quality of its evidence, so reproducibility and honest measurement matter as much as working code.

## 1. Start of every session

Read, in this order:
1. This file.
2. `docs/PLAN.md` for status and the next task.
3. The current phase in `docs/PLAN_FULL.md` (task prompt, done criteria, considerations).
4. The relevant sections of `docs/ARCHITECTURE.md`.
5. The last ~20 entries of `docs/EXPERIMENTS.md` and any related entries (search by tag) so you don't repeat dead ends.
6. `docs/DECISIONS.md` for decisions already made.

## 2. Working loop

1. Take the next unchecked task in `docs/PLAN.md`, or the task the user names.
2. Briefly state your approach before large changes.
3. Implement in small steps. Run `make test` after each meaningful change.
4. Before finishing a task: run `make check` and `make test-integration`, update docs (section 6), log experiments (section 4), tick the task in `docs/PLAN.md`, commit.
5. Report back briefly: what changed, test status, any numbers measured, open questions.

## 3. Testing

- `make test` (unit) after every change set. `make check` = lint + typecheck + unit tests. `make test-integration` before marking any task done.
- Every feature and bug fix comes with tests. For a bug, write the failing test first.
- Never skip, delete or weaken a test, or loosen an assertion, just to get green. If a test is wrong, fix it and say why in the commit body.
- Unit tests use no network, GPU or real LLM: inject fakes (DB, object store, Triton, LLM clients). Integration tests run against the compose CPU stack. GPU tests are marked `@pytest.mark.gpu` and skipped by default.
- Tests are deterministic: fixed seeds, injected clock, no sleeps for synchronization.
- Critical paths need thorough tests: `QueryPlan` validation and plan-to-SQL compilers (including equivalence across backends), SQL validator (including adversarial inputs), idempotent writes and redelivery, backpressure, tracking and zone events, drift statistics, eval scorers.
- Do not mark a task or phase complete with failing tests.

## 4. Experiment log: `docs/EXPERIMENTS.md`

Append-only. Log every attempt that tests a hypothesis or produces a number: benchmarks, eval runs, tuning, training runs, abandoned approaches, and bugs that changed results. Failed attempts are as valuable as successes. Keep entries compact:

```
### E-014 | 2026-11-02 | storage
Tried: ClickHouse MergeTree ORDER BY (camera_id, ts), 100M rows, 50k-row batch inserts.
Why: D1 candidate for time-range aggregates.
Result: ingest 410k rows/s; hourly agg p95 38 ms (PG: 1.9 s); 11 B/row. bench/results/2026-11-02-storage-ch.json
Outcome: keep; feeds D1.
```

- IDs increase monotonically. Tags: `infra, ingest, queue, storage, vector, inference, tracking, query, query-eval, retrieval, llm, lora, drift, k8s, cloud, frontend`.
- At most ~6 lines. Numbers over adjectives. Link the results file.
- Never edit old entries. Corrections are new entries that reference the old ID.

## 5. Decisions: `docs/DECISIONS.md`

Any choice a reviewer or interviewer could reasonably ask "why?" about gets an entry:

```
## D1: Detection store
Status: proposed | accepted | superseded by Dn    Date: YYYY-MM-DD
Context: the problem and constraints.
Options: candidates considered.
Decision: what was chosen.
Evidence: E-ids and key numbers.
Trade-offs and consequences: what we gave up, what to revisit and when.
```

Explain the reasoning, not only the outcome, clearly enough that someone could defend it in an interview. D1 to D11 from `docs/ARCHITECTURE.md` start as `proposed`. Add new decisions as they arise.

## 6. Documentation structure (fixed)

These are the only Markdown files in the repo:

| File | Contents |
|---|---|
| `README.md` | Pitch, headline results (linked to evidence), architecture diagram, quickstart, links into docs |
| `AGENTS.md` | These instructions |
| `CLAUDE.md` | One-line pointer (`@AGENTS.md`) so Claude Code loads these instructions; no other content |
| `docs/ARCHITECTURE.md` | Current design; update whenever the design changes |
| `docs/PLAN.md` | Compact plan with status checkboxes |
| `docs/PLAN_FULL.md` | Detailed plan: task prompts, done criteria, considerations |
| `docs/DECISIONS.md` | Decision records |
| `docs/EXPERIMENTS.md` | Append-only log of everything tried |
| `docs/BENCHMARKS.md` | Curated current numbers: tables, methodology, hardware, how to reproduce each |
| `docs/RUNBOOK.md` | Setup, data, running, deploying, operating, troubleshooting, teardown, production checklist |
| `docs/WRITEUP.md` | Final technical write-up (Phase 11) |

Rules:
- Do not create any other `.md` file: no summaries, notes, phase reports, changelogs, per-folder READMEs or scratch docs. New information goes into a section of the right file above.
- Code-level documentation lives in docstrings and comments.
- `EXPERIMENTS.md` is raw history; `BENCHMARKS.md` is the curated current state. When a new best number lands, update `BENCHMARKS.md` and link the experiment.
- Keep docs current with the code at every commit. Stale docs count as a bug.
- If something truly doesn't fit, ask the user before adding a file.

## 7. Reproducibility

- Dependencies are pinned (`uv.lock`, Node lockfile). Benchmark base images are pinned by digest.
- All configuration lives in versioned YAML under `configs/` with environment overrides (pydantic-settings). No magic numbers in code.
- Seed every source of randomness and record the seed.
- Every reported number comes from a script run through `make` or `python -m vms...`. No results from notebooks or ad hoc shell sessions.
- Every benchmark, eval and training run writes a results JSON with: run ID, git SHA and dirty flag, fully resolved config, hardware (GPU model, driver, CUDA, CPU, RAM), key package versions, seed, start and end time, metrics, cloud cost if any, and the tracker run IDs or URLs (MLflow, W&B) when a tracker is enabled. Small results are committed under `bench/results/` or `eval/results/`; large artifacts are tracked with DVC (below).
- Experiment tracking goes through the `Tracker` interface in `vms.common` (MLflow, W&B or no-op, chosen by config). W&B is for training runs, sweeps and reports; MLflow is for eval results, the model registry and champion tracking. The results JSON in the repo stays the source of truth for reported numbers.
- Data and artifacts: small files (eval JSONL, configs) stay in git. Large artifacts we produce (synthetic datasets, training data, LoRA adapters, fine-tuned weights, large eval snapshots) are tracked with DVC (local remote by default, S3 remote in the cloud). Public third-party datasets and pretrained weights are fetched by scripts with checksums and license notes. Never commit videos, images or weights to git.
- A fresh clone must work with `make setup && make up && make demo` on a CPU-only machine without W&B, AWS or DVC remote access (the demo uses `make data-mini`; W&B defaults to disabled or offline). Keep this true at all times.

## 8. Benchmarking and evaluation rules

- Write the scenario as YAML in `bench/scenarios/` before running it.
- Exclude warmup. Use at least 3 repetitions or a long enough steady state. Report p50, p95, p99, throughput, error rate, resource utilization and variance.
- Change one variable at a time against a named baseline.
- Report quality next to speed: inference changes report mAP and recall@k; LLM and query-engine changes report structured-query execution accuracy.
- Compare only results from the same hardware, and always state the hardware.
- Separate model-only from end-to-end numbers. End-to-end (rated capacity, query p95) is the headline.
- Tune on the dev split only. The test split is frozen and used only for reported numbers; record its hash.
- Make sure the harness itself is not the bottleneck (check loadgen and client CPU).
- Label synthetic data and self-measured results as such. Never overstate a result; negative results get reported too.

## 9. Code standards

- Python 3.12, type hints everywhere, `ruff` for lint and format, `mypy --strict` on `src/vms`.
- Pydantic for API and message schemas, protobuf for gRPC.
- One shared structured JSON logger with trace IDs; no `print` in services.
- OpenTelemetry tracing and Prometheus metrics in every service from the first version. Metric names follow `vms_<component>_<metric>_<unit>`.
- Async for IO-bound paths; never block the event loop with CPU-bound work.
- Clients (DB, object store, queue, Triton, LLM) are injected behind small interfaces so fakes are trivial.
- Every network call has a timeout; retries use backoff with jitter and are bounded.
- Each service exposes `/healthz`, `/readyz` and `/metrics`, and shuts down gracefully.
- Frontend: TypeScript, minimal, no heavy UI frameworks.

## 10. Cloud, GPUs and cost

- AWS is the default provider for all infrastructure (S3, ECR, IAM, AWS Budgets, EC2 spot GPU instances running k3s). All resources go in the project's single configured Region. D9 only decides whether long GPU benchmark runs move to a cheaper provider.
- GPUs are rented per session, only through scripts in `deploy/terraform/` or `scripts/`: provision -> run -> collect results -> destroy. Teardown runs even on failure.
- Tag every cloud resource `project=vms` and `session=<id>`. After teardown, `scripts/gpu_session.sh` lists resources still carrying the session tag (Resource Groups Tagging API) and fails loudly if any remain.
- Before anything that costs money or is irreversible (provisioning, deleting data or volumes, force-pushing, publishing anything), state the expected cost or impact and get the user's confirmation.
- Never leave resources running at the end of a session. Verify teardown and log the cost in the run's results.
- Secrets live only in `.env` (gitignored) or a secret manager. Keep `.env.example` complete, including optional variables such as `WANDB_API_KEY` and `WANDB_MODE` (default `disabled` or `offline`).

## 11. Git

- Conventional commits: `feat:`, `fix:`, `perf:`, `test:`, `bench:`, `docs:`, `refactor:`, `chore:`.
- One logical change per commit; commit after each green task.
- Never commit secrets, datasets, frames or weights. Never rewrite pushed history.

## 12. Initiative and judgment

- `docs/ARCHITECTURE.md` and the plans are a starting hypothesis, not a contract. Refactor, replace or remove code (including earlier work) when there is a better approach. If evidence points elsewhere, change course: log it in `EXPERIMENTS.md`, record or supersede the decision in `DECISIONS.md`, and update `ARCHITECTURE.md` and `PLAN.md`.
- Propose improvements, missing tests, better tools or simpler designs proactively.
- When a choice is non-obvious, explain the reasoning briefly in the decision record or commit body.
- Ask the user only when an action costs money, is irreversible, changes the project's goals or scope, or depends on an ambiguous requirement where a wrong guess is expensive.

## 13. Definition of done (any task)

- Code and tests in place; `make check` and `make test-integration` pass.
- Task-specific done criteria in `docs/PLAN_FULL.md` are met.
- Docs updated: `ARCHITECTURE.md` if the design changed, `RUNBOOK.md` if running or ops changed, `BENCHMARKS.md` if numbers changed, `DECISIONS.md` if a decision was made.
- Anything measured or tried is logged in `EXPERIMENTS.md`.
- Task ticked in `docs/PLAN.md`; work committed.
