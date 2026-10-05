# Runbook

## Setup

Prerequisites: [uv](https://docs.astral.sh/uv/) (it downloads Python 3.12 if needed), GNU make, Docker with Compose v2.

```bash
make setup   # uv sync --locked (incl. dev tools: ruff, mypy, pytest, pre-commit, dvc),
             # installs the pre-commit hook, creates .env from .env.example if missing
make check   # ruff lint + format check, mypy --strict on src/vms, unit tests
make help    # all targets
```

- Tests: `make test` runs unit tests only. The `integration`, `gpu` and `slow` markers are deselected by default; `make test-integration` runs `-m integration` (it needs `make up`). Run GPU tests with `uv run pytest -m gpu`.
- pre-commit runs ruff (check + format) and mypy from the project env, so they match `make check`. It also rejects files over 500 KB: large artifacts go to DVC.
- Configuration: `.env` (gitignored) for secrets and endpoints; every variable is documented in `.env.example`. W&B is disabled by default (`WANDB_MODE=disabled`).

## Data

- Large artifacts we produce are tracked with DVC. The default remote is `local`, a gitignored directory `.dvc-storage/` at the repo root (configured in `.dvc/config`). In the cloud, an S3 remote is set in `.dvc/config.local` (not committed).
- `make data-pull` runs `dvc pull` from the default remote. It is a no-op with a message when no remote is configured, the local remote directory is missing or empty, or nothing is DVC-tracked yet, so a fresh clone works offline.
- Push new artifacts with `uv run dvc add <path> && uv run dvc push`, then commit the `.dvc` pointer.
