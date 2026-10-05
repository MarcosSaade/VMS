# VMS

A scalable, production-style Video Management System built to demonstrate ML systems engineering: benchmarked design decisions, inference optimization, load testing, observability, drift detection and LLM serving. It ingests many simulated camera streams, runs detection, tracking and CLIP embedding on GPUs, and answers questions about the footage through structured, semantic and hybrid queries.

Status: early scaffold. See [docs/PLAN.md](docs/PLAN.md).

## Headline results

_No numbers yet._ Every headline number will link to its evidence in [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

## Architecture

_Diagram placeholder (task 11.1)._ For now, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), section 3.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/), GNU make and Docker.

```bash
make setup   # Python 3.12 env from uv.lock, pre-commit hooks, .env
make check   # lint + typecheck + unit tests
```

Run `make help` to list every target. Details are in [docs/RUNBOOK.md](docs/RUNBOOK.md).

## Documentation

| Document | Contents |
|---|---|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Current design |
| [docs/PLAN.md](docs/PLAN.md) / [docs/PLAN_FULL.md](docs/PLAN_FULL.md) | Plan with status, and detailed task prompts |
| [docs/DECISIONS.md](docs/DECISIONS.md) | Decision records (D1 to D11) |
| [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) | Append-only log of everything tried |
| [docs/BENCHMARKS.md](docs/BENCHMARKS.md) | Curated current numbers and how to reproduce them |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | Setup, running, operating, teardown |
| [docs/WRITEUP.md](docs/WRITEUP.md) | Final technical write-up |
| [AGENTS.md](AGENTS.md) | Instructions for coding agents |
