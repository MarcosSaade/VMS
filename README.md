# VMS

A scalable, production-style Video Management System built to demonstrate ML systems engineering: benchmarked design decisions, inference optimization, load testing, observability, drift detection and LLM serving.

Status: early scaffold. See [docs/PLAN.md](docs/PLAN.md) and [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Docker.

```bash
make setup   # Python 3.12 env from uv.lock, pre-commit hooks, .env
make check   # lint + typecheck + unit tests
```

Run `make help` for every target.
