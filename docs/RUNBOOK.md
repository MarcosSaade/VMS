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
- Configuration: service settings live in `configs/` (`common.yaml`, `<service>.yaml`, optional `<service>.<env>.yaml` chosen by `VMS_ENV`). Override any field without editing files with `VMS_<FIELD>`, using `__` for nesting: `VMS_LOG__LEVEL=DEBUG`, `VMS_TELEMETRY__SAMPLE_RATIO=0.1`. Priority from highest to lowest: explicit arguments, environment, `.env`, YAML layers, code defaults. `.env` (gitignored) holds secrets and endpoints; every variable is documented in `.env.example`. A `.env` key that matches no field of a service is ignored, because one `.env` is shared by all services.
- Results: every benchmark and eval run writes a results JSON (the source of truth for reported numbers). There is no experiment tracker; MLflow and W&B were removed in task 0.7 (D16).

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
make logs          # follow all logs; make logs SERVICE=grafana for one service
make down          # stop; data volumes are kept
make down-volumes  # stop and delete all stack data (Postgres, object store, Prometheus, Tempo, Grafana)
make test-integration  # smoke tests against the running stack
```

Compose reads the repo-root `.env` (or `.env.example` if `.env` is missing); every variable also has a default in the compose file. All ports bind to `127.0.0.1`. First start takes about 18 s once images are cached; the stack uses about 365 MiB of RAM at idle (D16).

| Service | Host URL / port (default) | Port variable | Notes |
|---|---|---|---|
| Grafana | http://localhost:3000 | `GRAFANA_PORT` | Anonymous viewer; admin is `GRAFANA_ADMIN_USER` / `GRAFANA_ADMIN_PASSWORD`. Datasources Prometheus and Tempo are provisioned. |
| Prometheus | http://localhost:9090 | `PROMETHEUS_PORT` | Scrape config: `deploy/compose/prometheus/prometheus.yml`. Retention `PROMETHEUS_RETENTION` (15d). |
| Tempo | http://localhost:3200 | `TEMPO_PORT` | Query API; traces arrive only through the collector. |
| OTel collector | localhost:4317 (gRPC), 4318 (HTTP) | `OTEL_GRPC_PORT`, `OTEL_HTTP_PORT` | Point `OTEL_EXPORTER_OTLP_ENDPOINT` here. |
| Postgres + pgvector | localhost:5432 | `POSTGRES_PORT` | User/DB from `POSTGRES_*`. The `vector` extension is available but not created: embeddings are deferred. |
| Object store (SeaweedFS, S3 API) | http://localhost:8333 | `S3_PORT` | Credentials `S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY`; bucket `S3_BUCKET` is created at startup. |

- Port taken on the host: set the `*_PORT` variable in `.env` and update the matching URL (`S3_ENDPOINT_URL` for `S3_PORT`), then `make up`.
- Profiles: `make up` activates the `cpu` profile; `make up PROFILES=gpu` activates the `gpu` profile (GPU services arrive in Phase 5 and need the NVIDIA container toolkit).
- Credentials and database names are applied when a volume is first created. After changing `POSTGRES_*` or `S3_*` credentials, run `make down-volumes && make up`, which deletes all stack data.
- Images are pinned by tag in the compose file. Tempo and the collector get a static busybox `wget` for health checks, because their images have no shell.

### Ingest service, migrations and the camera catalog

```bash
make up            # also runs make migrate: alembic upgrade head + python -m vms.storage bootstrap
make ingest        # ingest on http://127.0.0.1:8080 (configs/ingest.yaml; VMS_HTTP__PORT to change)
make loadgen       # 20 cameras x 5 FPS into it
curl -s localhost:8080/readyz   # {"postgres":"ok","object_store":"ok"}
```

- Connection settings come from `POSTGRES_*` and `S3_*` in `.env` (`vms.storage.config`); tuning (pool sizes, timeouts, limits) from `configs/ingest.yaml`.
- Migrations live in `src/vms/storage/migrations/versions/` as hand-written SQL. `uv run alembic current` shows the applied revision, `uv run alembic downgrade -1` rolls one back, `uv run alembic revision -m "..."` starts a new one.
- `make migrate` is idempotent; re-run it after editing `configs/cameras.yaml` or rotating `POSTGRES_READONLY_PASSWORD`. Bootstrap (connecting as `POSTGRES_USER`) creates or updates the query service's login `POSTGRES_READONLY_USER` as a member of `vms_query_reader` (SELECT on the validator's allowlisted columns only, read-only by default), then upserts cameras and zones from `configs/cameras.yaml`.
- Camera catalog: camera and zone names are what questions use and must stay unique (zone names across all cameras); polygons are normalized 0..1 frame coordinates. Upsert only: a camera or zone removed from the YAML stays in the database and bootstrap logs it as stale; delete it by hand (`DELETE FROM cameras WHERE camera_id = ...`, which also deletes its zones) if it really is gone. Renaming a zone creates a new zone id (`<camera_id>:<name>`).
- Ingest accepts frames from any camera id, catalogued or not (load tests use `cam-001`..`cam-NNN`); queries only see cameras in the catalog because they join `cameras` by name.
- Check what the query login can read: `psql -U vms_query -h localhost vms -c '\dp'`, or try `SELECT status FROM frames` as it (permission denied).
- What ingest stored: `frames` rows (status `pending` until a worker claims them) and objects under `frames/{camera_id}/{yyyy-mm-dd}/` in the `S3_BUCKET` bucket.
- Integration tests create and drop a throwaway database (`vms_it_*`) per test and delete the objects they wrote, so they do not touch dev data.

### Inference worker

```bash
make models        # fetch and verify yolo11n and yolo11s into models/weights/ (else the worker does it on start)
make worker        # one worker on the host: probes on http://127.0.0.1:8090 (configs/worker.yaml)
VMS_DETECTOR__WEIGHTS=yolo11n make worker    # the smaller model; VMS_DETECTOR__DEVICE=cuda on a GPU
make bench-model   # model-only latency (bench/scenarios/worker-model-cpu.yaml); SCENARIO=... for another
```

- Several workers can run at once (each `make worker` in its own shell with a different `VMS_HTTP__PORT`). Each leases up to `queue.max_cameras` cameras; a camera stays with one worker while its lease is renewed.
- Progress: `SELECT status, count(*) FROM frames GROUP BY 1`; `SELECT * FROM camera_leases` shows which worker holds which camera and until when. `/metrics` has `vms_worker_frames_total{outcome}`, `vms_worker_backlog_frames`, `vms_worker_stage_duration_seconds{stage}` and `vms_worker_capture_to_done_seconds` (freshness).
- A frame that keeps failing ends as `status = 'failed'` with the reason in `frames.error`. To retry failed frames after a fix: `UPDATE frames SET status = 'pending', attempts = 0, error = NULL WHERE status = 'failed'`.
- Ultralytics runs offline (`YOLO_OFFLINE=1`, `YOLO_AUTOINSTALL=0`, set by `vms.worker.detector`): no usage analytics, no packages installed at runtime. Weights and the package are AGPL-3.0 (see `configs/models.yaml`).
- PyTorch comes from the CPU wheel index (`pyproject.toml`, `[tool.uv.sources]`). GPU sessions install the CUDA build (task 2.4).

### Load generator

`vms.loadgen` replays clips from `make data-mini` (or `make data`) as virtual cameras. Settings are in `configs/loadgen.yaml`; a scenario in `bench/scenarios/` overrides them, and command-line flags override the scenario.

```bash
make loadgen-stub                       # stub ingest on 127.0.0.1:8080 (STUB_WORKERS=6 for max-rate runs)
make loadgen                            # 20 cameras x 5 FPS for 60 s against target_url
make loadgen SCENARIO=bench/scenarios/loadgen-stub-20x5.yaml
make loadgen-selftest                   # the generator's own max rate, no server
uv run python -m vms.loadgen run --cameras 50 --fps 5 --duration 120 --target-url http://127.0.0.1:8080
```

- Results: one JSON per invocation in `bench/results/` (per repetition: sent/accepted/rejected/failed/dropped frames, send latency and schedule lag p50/p95/p99, CPU per frame, `generator_ok`).
- Live metrics on `http://localhost:9105/metrics` while a run is going (`metrics_port`). Prometheus in the stack does not scrape it yet (task 2.1).
- Trust check before reading any number: `generator_ok` must be true (schedule lag p99 within `max_schedule_lag_p99_ms`) and `cpu.utilization_cores` well below 1.0. Near 1.0 the single event loop is saturated and the offered load is lower than configured. The measured ceiling is in E-010.
- Ctrl-C stops scheduling, waits for in-flight sends and writes partial results.

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

- Frames stay `processing` and nothing advances: a worker died holding leases. Its cameras move to another worker once `expires_at` in `camera_leases` passes (`queue.lease_ttl_s`, default 30 s), and any running worker resets the frames then. With no worker running, nothing moves: start one.
- Worker `/readyz` never turns ready right after start: it loads and warms the model first (a few seconds on CPU; the first run also downloads the weights).
- `make up` fails with "port is already allocated" or "address already in use": another process holds a default port. Change the `*_PORT` variable and its URL in `.env` (see Running locally).
- `make up` reports `dependency failed to start: container vms-<service>-1 is unhealthy`: run `make ps`, then `make logs SERVICE=<service>`. `docker inspect vms-<service>-1 --format '{{json .State.Health}}'` shows the health check output.
- Old `vms-redis-1` or `vms-mlflow-1` containers after pulling task 0.7: `make up` removes them (`--remove-orphans`); `make down-volumes` also drops their old volumes.
- `make data` fails with a checksum or size mismatch: the upstream file changed or the download was corrupted. Re-run once (corrupt cached files are re-downloaded). If it persists, the source changed: re-check the clip and re-pin it (see Data).
- `make data` reports an implausible frame rate for a video file: the container's rate is wrong. Set `source.fps` in `configs/data/sources.yaml` (check it with `ffprobe`).
- `make ingest` starts but `/readyz` returns 503: the named dependency is down or misconfigured. Check `make ps`, then the `POSTGRES_*` / `S3_*` values in `.env` (they must match what the stack was created with).
- Ingest answers 500 `database_unavailable` with `relation "frames" does not exist`: migrations were not applied. Run `make migrate`.
- `vms_ingest_orphaned_objects_total` above 0: an object was written but neither recorded nor removed (database and store both failed for that frame). Harmless for correctness; the objects under `frames/` that have no `frames` row can be deleted.
- `make loadgen` reports every frame failed with reason `connection`: nothing listens on `target_url`. Start the ingest service or `make loadgen-stub`.
- `make loadgen` logs "generator fell behind its schedule": the host or the generator is saturated, so the run offered less load than configured. Reduce cameras, close other load on the machine, or run several loadgen processes with different `camera_prefix` values.
- Sudden `AccessDenied` errors on AWS operations that used to work: check the project's spend limit status in AWS Settings > Billing.

## Teardown

_Placeholder (tasks 2.4, 10.1)._ `make down` locally (`make down-volumes` also deletes stack data). In the cloud: the scripted destroy, then the Resource Groups Tagging API check that no resource tagged `session=<id>` remains, then recording the session cost in the run's results.

## Production checklist

_Placeholder (task 10.4)._ The project is never called "production-ready" unless every item here is met.
