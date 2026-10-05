# Runbook

> Location in repo: `docs/RUNBOOK.md`. How to set up, run, operate and tear down the system. Update it whenever running or operations change.

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
- CI (`.github/workflows/ci.yml`) runs `make setup && make check` on every push and pull request, with the uv dependency and Python caches keyed on `uv.lock`. Any lint, type or test error fails the run. The `integration` job is a placeholder that runs `make test-integration` without the compose stack until task 0.5 adds it. Actions are pinned by commit SHA; to update one, look up the new release tag's SHA and keep the `# vX.Y.Z` comment in sync. The uv version is pinned in the workflow's `UV_VERSION`. Lint workflow changes locally with `uvx --from actionlint-py actionlint`.
- Configuration: service settings live in `configs/` (`common.yaml`, `<service>.yaml`, optional `<service>.<env>.yaml` chosen by `VMS_ENV`). Override any field without editing files with `VMS_<FIELD>`, using `__` for nesting: `VMS_LOG__LEVEL=DEBUG`, `VMS_TRACKING__BACKENDS='["mlflow"]'`. Priority from highest to lowest: explicit arguments, environment, `.env`, YAML layers, code defaults. `.env` (gitignored) holds secrets and endpoints; every variable is documented in `.env.example`. A `.env` key that matches no field of a service is ignored, because one `.env` is shared by all services.
- Experiment tracking: set `tracking.backends` to any of `noop` (default), `mlflow` and `wandb`. MLflow uses `tracking.mlflow.tracking_uri`, else `MLFLOW_TRACKING_URI`. W&B uses `tracking.wandb.mode`, else `WANDB_MODE`, else `disabled`, so it never needs an account unless you choose `online`. MLflow and W&B are installed by `make setup` (the `tracking` dependency group) and imported only when enabled.

## Data

- Large artifacts we produce are tracked with DVC. The default remote is `local`, a gitignored directory `.dvc-storage/` at the repo root (configured in `.dvc/config`). In the cloud, an S3 remote is set in `.dvc/config.local` (not committed).
- `make data-pull` runs `dvc pull` from the default remote. It is a no-op with a message when no remote is configured, the local remote directory is missing or empty, or nothing is DVC-tracked yet, so a fresh clone works offline.
- Push new artifacts with `uv run dvc add <path> && uv run dvc push`, then commit the `.dvc` pointer.

## Running locally

_Placeholder (tasks 0.5, 1.9)._ `make up` starts the compose CPU stack, `make demo` runs the end-to-end CPU demo on `make data-mini`, `make logs` follows logs and `make down` stops the stack. Service URLs and ports go here.

## Benchmarks

_Placeholder (task 2.3)._ Writing a scenario in `bench/scenarios/`, running it with `make bench`, where results land, and how to promote a number to `docs/BENCHMARKS.md`.

## GPU sessions

_Placeholder (task 2.4)._ `scripts/gpu_session.sh` runs provision, sync, run scenarios, collect results and destroy, with teardown in a trap. Includes prerequisites (AWS credentials, the project's selected Region, spot quota), expected cost per hour and the confirmation step before any apply.

## Deployment

_Placeholder (tasks 6.4, 10.1, 10.2)._ Helm chart, kind smoke test, Terraform for the AWS environment, and the CD pipeline with its eval gate.

## Operations

_Placeholder (tasks 2.2, 9.4)._ Dashboards, alerts, drift response procedure, scaling, and promoting a model to champion.

## Troubleshooting

_Placeholder._ Symptom, cause and fix entries as issues are met. For sudden `AccessDenied` errors on AWS operations that used to work, check the project's spend limit status in AWS Settings > Billing.

## Teardown

_Placeholder (tasks 2.4, 10.1)._ `make down` locally. In the cloud: the scripted destroy, then the Resource Groups Tagging API check that no resource tagged `session=<id>` remains, then recording the session cost in the run's results.

## Production checklist

_Placeholder (task 10.4)._ The project is never called "production-ready" unless every item here is met.
