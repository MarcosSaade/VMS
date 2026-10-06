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
- CI (`.github/workflows/ci.yml`) runs `make setup && make check` on every push and pull request, with the uv dependency and Python caches keyed on `uv.lock`. Any lint, type or test error fails the run. The `integration` job runs `make up`, then `make test-integration`, dumps service state and logs on failure, and always finishes with `make down-volumes`. Actions are pinned by commit SHA; to update one, look up the new release tag's SHA and keep the `# vX.Y.Z` comment in sync. The uv version is pinned in the workflow's `UV_VERSION`. Lint workflow changes locally with `uvx --from actionlint-py actionlint`.
- Configuration: service settings live in `configs/` (`common.yaml`, `<service>.yaml`, optional `<service>.<env>.yaml` chosen by `VMS_ENV`). Override any field without editing files with `VMS_<FIELD>`, using `__` for nesting: `VMS_LOG__LEVEL=DEBUG`, `VMS_TRACKING__BACKENDS='["mlflow"]'`. Priority from highest to lowest: explicit arguments, environment, `.env`, YAML layers, code defaults. `.env` (gitignored) holds secrets and endpoints; every variable is documented in `.env.example`. A `.env` key that matches no field of a service is ignored, because one `.env` is shared by all services.
- Experiment tracking: set `tracking.backends` to any of `noop` (default), `mlflow` and `wandb`. MLflow uses `tracking.mlflow.tracking_uri`, else `MLFLOW_TRACKING_URI`. W&B uses `tracking.wandb.mode`, else `WANDB_MODE`, else `disabled`, so it never needs an account unless you choose `online`. MLflow and W&B are installed by `make setup` (the `tracking` dependency group) and imported only when enabled.

## Data

### Video clips (virtual cameras)

```bash
make data-mini          # 3 short clips (~180 MB, under a minute): CI, laptops, the demo
make data               # all clips that need no license acceptance (~820 MB, about 3 minutes)
make data ACCEPT=virat  # also the VIRAT clips, after you have read and accepted their agreement
```

`python -m vms.datasets` reads `configs/data/sources.yaml`, downloads each clip, verifies it against its pinned checksum, and writes normalized clips to `data/clips/<clip_id>/` (gitignored): JPEG frames scaled to at most 720 px high (never upscaled), at most 30 fps, JPEG quality 90, plus a `clip.json` with the measured properties. MOT sequences are read straight out of the 5 to 6 GB archives with HTTP range requests, so only the frames needed are downloaded. Video files are cached in `data/raw/`. Re-running is idempotent: a clip whose `clip.json` matches its spec hash (source plus processing settings) is skipped, and a failed or interrupted run never leaves a clip that looks complete. The run also refreshes `configs/data/manifest.yaml` (committed), which records each clip's frames, fps, size, mean brightness and content hash. If a rebuilt clip's content hash differs from the committed one (for example, a different OpenCV build), the run warns; `--strict` makes that an error.

| Clips | Source | License | Notes |
|---|---|---|---|
| mot17-02, mot17-04, mot17-08, mot17-09 (+ two 10 s `-mini` trims) | [MOT17](https://motchallenge.net/data/MOT17/) | [CC BY-NC-SA 3.0](https://creativecommons.org/licenses/by-nc-sa/3.0/) | Non-commercial only. Cite Milan et al., MOT16, arXiv:1603.00831. |
| mot20-01, mot20-03 (first 30 s) | [MOT20](https://motchallenge.net/data/MOT20/) | CC BY-NC-SA 3.0 | Non-commercial only. Cite Dendorfer et al., MOT20, arXiv:2003.09003. |
| commons-roncesvalles (first 10 s) | [Wikimedia Commons, by Qviri](https://commons.wikimedia.org/wiki/File:Roncesvalles_Ave_crossing_at_Galley_Ave,_PXO_Type_A_lights_flashing,_July_2026.webm) | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) | Attribute the author; derivatives share alike. |
| virat-000201, virat-040000, virat-050000 | [VIRAT Video Dataset 2.0](https://viratdata.org/), Kitware | [VIRAT Video Data Set Protection Agreement](https://actev.kitware.com/pub/VIRAT-Video-Data-Set-Protection-Agreement-1-4-11.pdf) | Opt-in (`ACCEPT=virat`). Every user must sign the agreement. No redistribution; never try to identify people. A few frames may appear in publications. |

Because of the NC and agreement terms, never commit or publish clips or frames from `data/`, and keep screenshots in public material to a few frames with attribution. Checked in October 2026: the MOTChallenge site's front page returns HTTP 410, but the `data/*.zip` downloads still work. If a source disappears, the fetch fails loudly on the size or checksum check; replace the clip in `sources.yaml` rather than loosening the check.

Adding a clip: look at its frames first, keep only static cameras, and label the scene by eye. For a video file, use the checksum its host publishes (Commons: sha1 from the API; Kitware: sha512). For a zip sequence, set a placeholder checksum, run `uv run python -m vms.datasets pin <clip_id>`, and paste the printed value. Set `source.fps` when the container's frame rate is unreliable (E-006). Bump `version` in `sources.yaml`.

### DVC artifacts

- Large artifacts we produce are tracked with DVC. The default remote is `local`, a gitignored directory `.dvc-storage/` at the repo root (configured in `.dvc/config`). In the cloud, an S3 remote is set in `.dvc/config.local` (not committed).
- `make data-pull` runs `dvc pull` from the default remote. It is a no-op with a message when no remote is configured, the local remote directory is missing or empty, or nothing is DVC-tracked yet, so a fresh clone works offline.
- Push new artifacts with `uv run dvc add <path> && uv run dvc push`, then commit the `.dvc` pointer.

## Running locally

`make demo` (task 1.9) will run the end-to-end CPU demo on `make data-mini`. The local stack exists now:

```bash
make up            # build/pull, start, wait until every health check passes (timeout 300 s), print URLs
make ps            # services and health
make logs          # follow all logs; make logs SERVICE=mlflow for one service
make down          # stop; data volumes are kept
make down-volumes  # stop and delete all stack data (Postgres, object store, Redis, Prometheus, Tempo, Grafana)
make test-integration  # smoke tests against the running stack
```

Compose reads the repo-root `.env` (or `.env.example` if `.env` is missing); every variable also has a default in the compose file. All ports bind to `127.0.0.1`. First start takes about 18 s once images are cached; the stack uses about 1 GiB of RAM at idle (E-002).

| Service | Host URL / port (default) | Port variable | Notes |
|---|---|---|---|
| Grafana | http://localhost:3000 | `GRAFANA_PORT` | Anonymous viewer; admin is `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD`. Datasources Prometheus and Tempo are provisioned. |
| MLflow | http://localhost:5000 | `MLFLOW_PORT` | Backend DB `MLFLOW_DB` in Postgres; artifacts in bucket `MLFLOW_ARTIFACTS_BUCKET`, uploaded through the server. |
| Prometheus | http://localhost:9090 | `PROMETHEUS_PORT` | Scrape config: `deploy/compose/prometheus/prometheus.yml`. Retention `PROMETHEUS_RETENTION` (15d). |
| Tempo | http://localhost:3200 | `TEMPO_PORT` | Query API; traces arrive only through the collector. |
| OTel collector | localhost:4317 (gRPC), 4318 (HTTP) | `OTEL_GRPC_PORT`, `OTEL_HTTP_PORT` | Point `OTEL_EXPORTER_OTLP_ENDPOINT` here. |
| Postgres + pgvector | localhost:5432 | `POSTGRES_PORT` | User/DB from `POSTGRES_*`. `vector` is available; migrations (task 1.4) create it. |
| Redis | localhost:6379 | `REDIS_PORT` | AOF enabled. |
| Object store (SeaweedFS, S3 API) | http://localhost:8333 | `S3_PORT` | Credentials `S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY`; buckets `S3_BUCKET` and `MLFLOW_ARTIFACTS_BUCKET` are created at startup. |

- Port taken on the host: set the `*_PORT` variable in `.env` and update the matching URL (`REDIS_URL`, `S3_ENDPOINT_URL`, `MLFLOW_TRACKING_URI`), then `make up`.
- Profiles: `make up` activates the `cpu` profile; `make up PROFILES=gpu` activates the `gpu` profile (GPU services arrive in Phase 5 and need the NVIDIA container toolkit).
- Credentials and database names are applied when a volume is first created. After changing `POSTGRES_*`, `MLFLOW_DB` or `S3_*` credentials, run `make down-volumes && make up`, which deletes all stack data.
- Images are pinned by tag in the compose file. The MLflow image (`deploy/compose/mlflow/Dockerfile`) adds the Postgres and S3 drivers; keep its tag in step with `mlflow-skinny` in `uv.lock`. Tempo and the collector get a static busybox `wget` for health checks, because their images have no shell.

## Benchmarks

_Placeholder (task 2.3)._ Writing a scenario in `bench/scenarios/`, running it with `make bench`, where results land, and how to promote a number to `docs/BENCHMARKS.md`.

## GPU sessions

_Placeholder (task 2.4)._ `scripts/gpu_session.sh` runs provision, sync, run scenarios, collect results and destroy, with teardown in a trap. Includes prerequisites (AWS credentials, the project's selected Region, spot quota), expected cost per hour and the confirmation step before any apply.

## Deployment

_Placeholder (tasks 6.4, 10.1, 10.2)._ Helm chart, kind smoke test, Terraform for the AWS environment, and the CD pipeline with its eval gate.

## Operations

_Placeholder (tasks 2.2, 9.4)._ Dashboards, alerts, drift response procedure, scaling, and promoting a model to champion.

## Troubleshooting

Symptom, cause and fix entries as issues are met.

- `make up` fails with "port is already allocated" or "address already in use": another process holds a default port. Change the `*_PORT` variable and its URL in `.env` (see Running locally).
- `make up` reports `dependency failed to start: container vms-<service>-1 is unhealthy`: run `make ps`, then `make logs SERVICE=<service>`. `docker inspect vms-<service>-1 --format '{{json .State.Health}}'` shows the health check output.
- MLflow fails with a database authentication error after you change `POSTGRES_PASSWORD`: the password was set when the volume was first created. Run `make down-volumes && make up`.
- `make data` fails with a checksum or size mismatch: the upstream file changed or the download was corrupted. Re-run once (corrupt cached files are re-downloaded). If it persists, the source changed: re-check the clip and re-pin it (see Data).
- `make data` reports an implausible frame rate for a video file: the container's rate is wrong. Set `source.fps` in `configs/data/sources.yaml` (check it with `ffprobe`).
- Sudden `AccessDenied` errors on AWS operations that used to work: check the project's spend limit status in AWS Settings > Billing.

## Teardown

_Placeholder (tasks 2.4, 10.1)._ `make down` locally (`make down-volumes` also deletes stack data). In the cloud: the scripted destroy, then the Resource Groups Tagging API check that no resource tagged `session=<id>` remains, then recording the session cost in the run's results.

## Production checklist

_Placeholder (task 10.4)._ The project is never called "production-ready" unless every item here is met.
