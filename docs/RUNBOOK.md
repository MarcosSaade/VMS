# Runbook

> Location in repo: `docs/RUNBOOK.md`. How to set up, run, operate and tear down the system. Update it whenever running or operations change.

## Setup

Prerequisites: [uv](https://docs.astral.sh/uv/) (it downloads Python 3.12 if needed), GNU make, Docker with Compose v2. Node 22.12 or newer for the host UI targets (`make zones`, `make ui-build`, `make ui-check`); `frontend/.nvmrc` pins the version CI and the app image use (`nvm install` in `frontend/`).

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
- Tracked now: the structured eval snapshot `eval/structured/snapshot.v1/` (16 MB; regenerable from its spec, see Evaluation).

## Running locally

### Demo (end to end on CPU)

```bash
make setup && make demo                 # needs only Docker and uv; ~10 min the first time
make demo DEMO_SECONDS=30               # shorter load (CI runs this)
```

What it does (`scripts/demo.sh`): `make up` (stack, migrations, camera catalog), `make data-mini` (3 clips, ~180 MB the first time), the yolo11n weights, then the application containers from one image (`make app` builds it: ingest on :8080, one worker on :8090, the query service with the UI on :8070), then loadgen with 4 cameras x 5 FPS for `DEMO_SECONDS` (180) against the containerized ingest, waits until the worker has processed every frame (`python -m vms.storage wait-drained`, printing frames, tracks and events per camera), prints the answers to the example questions (`python -m vms.query ask --check`, which fails the demo if any question errors or none returns rows), and finally checks the telemetry (`python -m vms.observe`, see Traces and metrics), printing one frame's and one question's trace as a time breakdown. Everything stays up: open http://localhost:8070, then `make down` (or `make down-volumes` for an empty start next time).

- Knobs: `DEMO_SECONDS`, `DEMO_CAMERAS` (4; the catalog has 4 cameras, more are ingested but invisible to questions), `DEMO_FPS` (5), `DEMO_WEIGHTS` (yolo11n), `DEMO_DRAIN_TIMEOUT_S` (900), `DEMO_TRACE_RATIO` (0.1: share of frames traced).
- With `LLM_MODEL` set, every demo run sends the 10 example questions to that model (about 45k prompt tokens with the planner.v1 prompt, E-017), which a hosted provider bills.
- What it fakes: the planner is the canned one unless `LLM_MODEL` is set (then the same questions go to the real model); clips are 10 s loops, so each loop restarts every person's walk and counts of tracks are much higher than the people in view (D21 already notes tracks overcount); the camera and zone names are labels on public video, not real places.
- On a laptop CPU one yolo11n worker processes fewer than the 20 frames/s offered, so frames queue during loadgen and the wall shows frames tens of seconds old; the drain step waits them out (E-020).
- Re-running adds a new loadgen run to the same database; "last hour" answers then include the previous run.

### Zone editor

```bash
make zones          # builds the UI (Node 22), then http://127.0.0.1:8075
```

- Tabs per catalog camera, each showing frames of the clip that camera plays in the demo (loadgen's round robin over `configs/loadgen.yaml` clips: cam-001 mot17-09-mini, cam-002 mot17-04-mini, cam-003 commons-roncesvalles, cam-004 mot17-09-mini). The slider scrubs through the clip.
- New zone: click "+ New zone", click the corners, then click the first point or press Enter (Backspace removes the last point, Esc cancels). Rename it in the list (lowercase, digits, `_`; unique across all cameras). Selected zone: drag points, double-click an edge to add a point, right-click a point to delete it. Save writes `configs/cameras.yaml` (validated by the same rules as `make migrate`; the header comment is kept). Revert drops unsaved edits.
- Afterwards: `make migrate` loads the zones into Postgres (running workers reload them within `zones.refresh_s`, 60 s), and check them on video: `uv run python -m vms.worker render mot17-09-mini:entrance mot17-04-mini:checkout commons-roncesvalles:parking_lot mot17-09-mini:loading_dock --out ~/Desktop/zones.mp4`.
- Zone ids are `<camera_id>:<name>`: renaming a zone starts a new zone; events of the old name stay under the old id.
- Show a different clip for a camera: `uv run python -m vms.zones edit --clip cam-004=mot17-04-mini`.

### Stack

The local stack on its own:

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
- Ingest accepts frames from any camera id, catalogued or not (load tests use `cam-001`..`cam-NNN`); queries only see cameras in the catalog: the compiler restricts every query to catalog camera ids.
- Check what the query login can read: `psql -U vms_query -h localhost vms -c '\dp'`, or try `SELECT status FROM frames` as it (permission denied).
- What ingest stored: `frames` rows (status `pending` until a worker claims them) and objects under `frames/{camera_id}/{yyyy-mm-dd}/` in the `S3_BUCKET` bucket.
- Integration tests create and drop a throwaway database (`vms_it_*`) per test and delete the objects they wrote, so they do not touch dev data.

### Inference worker

```bash
make models        # fetch and verify yolo11n and yolo11s into models/weights/ (else the worker does it on start)
make worker        # one worker on the host: probes on http://127.0.0.1:8090 (configs/worker.yaml)
VMS_DETECTOR__WEIGHTS=yolo11n make worker    # the smaller model; VMS_DETECTOR__DEVICE=cuda on a GPU
make bench-model   # model-only latency (bench/scenarios/worker-model-cpu.yaml); SCENARIO=... for another
make onnx          # export the pinned weights to ONNX into models/onnx/ (+ a sidecar with the weights' sha256)
make onnx-parity   # ONNX Runtime vs PyTorch on the detection eval subset and data-mini (WEIGHTS=yolo11s); ~25 min
VMS_DETECTOR__BACKEND=onnxruntime make worker   # the ONNX backend (needs make onnx first)
```

- Backends (`detector.backend`; `configs/worker.yaml` keeps `ultralytics` for host tools, the compose worker runs `onnxruntime` per D5): `ultralytics` (PyTorch through Ultralytics' predictor) or `onnxruntime` (`vms.worker.onnx_detector`: the ONNX export with numpy letterbox and NMS that copy Ultralytics'). `model_version` ends in `+onnx` for the latter. The worker refuses an export made from other weights or edited after export ("run `make onnx` again"). `detector.threads` sets intra-op threads for either backend.
- Containerized worker (compose, `make demo`): ONNX Runtime FP32, `infer_batch` 4, spinning off by default (D5); `make demo` exports the ONNX model if `models/onnx/<weights>.onnx` is missing. Override with `WORKER_BACKEND=ultralytics`, `WORKER_INFER_BATCH`, `WORKER_SPIN=true` in `.env` or the shell (then `compose --profile app up -d worker`). After `make models` updates weights, run `make onnx` again: the worker refuses a stale export. The GPU override (`docker-compose.gpu.yml`) pins PyTorch CUDA at batch 1. `detector.spin` (`VMS_DETECTOR__SPIN`) applies to onnxruntime only: true is ORT's busy-wait, which keeps ~9 cores busy at any load.
- Cross-camera batching: `queue.infer_batch` (`VMS_QUEUE__INFER_BATCH`, at most `queue.claim_batch`) frames per forward pass; the worker uses what is already claimed and never waits to fill a batch. `vms_worker_infer_batch_frames` shows the batch sizes it actually runs, and `vms_worker_model_info` shows backend, precision and `infer_batch` (`curl -s localhost:8090/metrics | grep model_info`).
- `make onnx-parity` exits 1 if parity fails: a letterbox that differs from Ultralytics by one byte, raw outputs beyond the tolerances in `vms.worker.parity.Tolerances`, or a detection on one side only that is not explained by a score at the threshold or an IoU at the NMS threshold. Results in `bench/results/*-onnx-parity-*.json`.

- Several workers can run at once (each `make worker` in its own shell with a different `VMS_HTTP__PORT`). Each leases up to `queue.max_cameras` cameras; a camera stays with one worker while its lease is renewed.
- Progress: `SELECT status, count(*) FROM frames GROUP BY 1`; `SELECT * FROM camera_leases` shows which worker holds which camera and until when. `/metrics` has `vms_worker_frames_total{outcome}`, `vms_worker_backlog_frames`, `vms_worker_stage_duration_seconds{stage}` and `vms_worker_capture_to_done_seconds` (freshness).
- Tracking and zones: `configs/worker.yaml` `tracker` and `zones`. Zones come from `configs/cameras.yaml` (`make migrate` syncs them); workers reload a camera's zones every `zones.refresh_s`. Check results with `SELECT event_type, count(*) FROM events GROUP BY 1` and `SELECT camera_id, count(*), avg(n_frames) FROM tracks GROUP BY 1`. Tracks lasting far longer than a visit point at a tracking bug (see E-015). `vms_worker_tracker_resets_total` counts new tracker sessions after pauses.
- Review video of what the worker does (detections, track ids, anchors, zones, events), no database needed; zones come from `configs/cameras.yaml` by camera name, frames are sampled like loadgen sends them and the video plays in real time:
  `uv run python -m vms.worker render mot17-09:entrance mot17-04:checkout commons-roncesvalles:parking_lot --out ~/Desktop/vms-tracking-review.mp4` (`--seconds 10` per clip, `--weights yolo11n`). Grey boxes are stored detections without a track; header "total" counts confirmed tracks (the `#n` numbers skip ids ByteTrack spent on tentative tracks it never confirmed).
- A frame that keeps failing ends as `status = 'failed'` with the reason in `frames.error`. To retry failed frames after a fix: `UPDATE frames SET status = 'pending', attempts = 0, error = NULL WHERE status = 'failed'`.
- Ultralytics runs offline (`YOLO_OFFLINE=1`, `YOLO_AUTOINSTALL=0`, set by `vms.worker.detector`): no usage analytics, no packages installed at runtime. Weights and the package are AGPL-3.0 (see `configs/models.yaml`).
- PyTorch comes from the CPU wheel index (`pyproject.toml`, `[tool.uv.sources]`). GPU sessions install the CUDA build (task 2.4).

### Drift monitor

```bash
make drift                                       # on the host: probes and /metrics on http://127.0.0.1:8095 (configs/drift.yaml)
uv run python -m vms.drift references            # each camera's reference range
uv run python -m vms.drift recalibrate cam-001   # new calibration period from now (reference_s); --all for every camera
uv run python -m vms.drift recalibrate cam-001 --start 2026-10-16T09:00:00+00:00 --end 2026-10-16T09:10:00+00:00
```

- The compose service `drift` (profile app) starts with `make demo`; Prometheus scrapes it as job `drift`. A camera's statistics appear `reference_s` (60 s, `DRIFT_REFERENCE_S`) plus a few seconds after its first frame; until then `vms_drift_reference_frames` is 0. Check with `curl -s localhost:8095/metrics | grep vms_drift_shift` or in Prometheus: `vms_drift_shift_value{signal="brightness"}`.
- The reference is stored as a capture-time range per camera (`SELECT * FROM drift_references`); the monitor rereads the range's rows after a restart. A reference recorded while the camera was already drifted makes the drift the new normal: recalibrate onto a known-clean period. Re-running `make demo` keeps the first run's references (same camera ids and clips).
- NaN means undefined, not "no drift": fewer than `signals.min_frames` frames in the window (camera stopped, backlog), or fewer than `signals.min_detections` detections (empty scene) for the confidence and class-mix statistics.
- The data-mini clips loop every 10 s, so a 30 s window of a 300-frame clip holds the same frames as the reference and its input shifts are exactly 0 (E-050); expect clean values to vary more on real cameras.

### Drift alerts

The monitor runs the detector chosen in task 9.3 (D34): per camera and channel, z = (statistic - clean mean) / clean std from `calibration` in `configs/drift.yaml`, alarm while |z| > `detector.h` (10; one-sided channels: z > h). Gauges `vms_drift_z_value{camera_id,channel}`, `vms_drift_detector_statistic_value` (what is compared with h: |z|, or S for CUSUM), `vms_drift_alarm_value` (1 while alarming), `vms_drift_alarms_total`, `vms_drift_detector_h_value`, and for the threshold detector `vms_drift_band_value{camera_id,signal,bound}` (the window means at which a signal alarms, drawn as the band on the Drift dashboard); rules in `deploy/compose/prometheus/rules/drift.yml` (Prometheus, Alerts tab; there is no Alertmanager in the stack, so nothing is sent anywhere).

| Alert | Means | Confirm |
|---|---|---|
| `VmsCameraDrift` (warning) | A channel of the camera is more than h clean standard deviations from its reference. `channel` says which statistic moved (`shift:brightness`, `shift:log_blur`, `class_psi`, ...), not why: darkening also moves noise, a camera shift reads as blur (BENCHMARKS, "Detection delay and false alarms"). | The Drift dashboard: the camera's row (signal against its alarm band) and "Detector statistic by channel"; then the camera in the UI (camera wall) for fog, a dirty lens, a moved mount, night, a new scene. |
| `VmsDriftNoData` (info) | The camera's windows have fewer than 20 processed frames for 2 minutes: it stopped sending or the worker is behind. Drift is undetectable meanwhile. | `vms_worker_backlog_frames`, the Pipeline dashboard; loadgen or the camera itself. |
| `VmsDriftMonitorDown` (warning) | Prometheus cannot scrape the drift service. | `docker compose ps drift`, its logs. |

Procedure when `VmsCameraDrift` fires (for example on one camera at 3 a.m.), in order:
1. **Is the monitor itself sound?** The alert is real only if the statistics are: on the Drift dashboard, "Fewest frames per window" must be well above 20 and `VmsDriftNoData` silent for the camera. A camera that just resumed after a gap can show odd windows (the detector skips windows covering < 90% of 30 s, but check the "Frames in window" panel of its row).
2. **One camera or many?** "Alarm by camera". Several cameras alarming together points at something shared: the worker (model swapped, a different backend or weights, preprocessing change; check the Inference dashboard and recent deploys) or a site-wide change (power, a lighting schedule, weather). One camera points at that camera.
3. **Which channel, and since when?** "Detector statistic by channel", then the camera's row: which signal left its band, how far (tens of h means a step change, just over h a marginal one), and whether it jumped (fault, re-aim, light switched) or ramped (dusk, fog, a lens slowly dirtying). Remember the channel names a symptom: darkening also lowers noise, a camera shift reads as blur, a new scene moves class PSI and detections together.
4. **Look at the frames.** The UI camera wall (or the object store keys of the camera's recent frames) shows night, glare, fog, a dirty or blocked lens, a moved mount, a new scene. Compare with a frame from the reference range (`uv run python -m vms.drift references`).
5. **Does it hurt detection?** Input drift that leaves the outputs (detections per frame, confidence, class mix) inside their bands may be harmless; the mAP table under drift (BENCHMARKS, "Detection quality under drift") says how much a given darkening, blur or noise costs. Output drift without input drift means the scene (traffic, class mix) or the model changed.
6. **Choose a response** (below), and record what it was: an alarm that was a false alarm counts against the calibration.

Responses:
- An intended change (camera re-aimed, new lighting): `uv run python -m vms.drift recalibrate CAMERA` (new reference from now; or `--start/--end` on a known-clean period). The alarm clears at the next update.
- A fault (dirty or fogged lens, failing sensor): fix the camera; the alarm clears by itself once the window is back near the reference.
- Flag the camera while it is degraded: its detections and the answers built on them are less reliable until fixed. There is no flag in the schema yet; for now note it, and leave the alarm firing rather than recalibrating onto the degraded state (that would make it the new normal, D33).
- A real domain change that hurts detection (night, weather): route its frames for labeling (keep the object store keys of the alarm period: `SELECT object_key FROM frames WHERE camera_id = '...' AND ts BETWEEN ...`), measure its mAP cost on them or on the 9.2 transforms (`make eval-drift`), and retrain on a clean plus drifted mix (task 9.5) if the cost is material. Until a model handles it, a per-time-of-day reference (not built, D34) is the honest fix for nightly alarms.
- Too many alarms on a healthy site: the shipped calibration comes from four looping MOT17 clips on one afternoon (E-054), so it understates normal variation. Recalibrate on that site's clean video (a day, to include its lighting and traffic): record it, list its runs in a `drift-score` scenario, `make drift-score SCENARIO=...`, then copy `calibration_all` from the results into `configs/drift.yaml` and restart the monitor. Raising `detector.h` trades delay for fewer alarms (the sweep table shows the cost). Removing `calibration` turns alarms off.

### Query service

```bash
make query         # http://127.0.0.1:8070 (configs/query.yaml; VMS_HTTP__PORT to change)
curl -s localhost:8070/v1/query -H 'Content-Type: application/json' \
  -d '{"question": "How many people did each camera see in the last hour?"}' | jq
uv run python -m vms.query plan "How many cars were in the parking lot today?"   # plan + SQL, no database
uv run python -m vms.query plan --compare configs/demo/canned_plans.v1.jsonl    # plan the demo questions, report matches
```

- Which LLM plans: `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` in `.env` (`vms.query.llm`). With `LLM_MODEL` empty the service uses the canned planner, which answers only the questions in `configs/demo/canned_plans.v1.jsonl` and abstains on anything else with no reason and the answer "No LLM is configured..." (the UI shows "Not answered"); the startup log says so. `VMS_LLM__PROVIDER=canned|openai` forces one.
  - Ollama: `ollama pull qwen2.5:7b`, then `LLM_BASE_URL=http://localhost:11434/v1`, `LLM_MODEL=qwen2.5:7b`, no key. Ollama supports JSON-schema structured output (the default, `llm.response_format: json_schema`).
  - Hosted OpenAI-compatible APIs: set `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`. DeepSeek (`https://api.deepseek.com`) rejects `json_schema`; set `VMS_LLM__RESPONSE_FORMAT=json_object`.
- The response is one JSON object for every outcome: `status` (`ok`, `abstained`, `error`), `answer`, `result_kind`, `columns`/`rows`, `plan`, `compiled_sql` (the validated SQL that ran, with `%(pN)s` placeholders) and `params`, `error` (`type`, `detail`, plan `errors`), `timings_ms` per stage, `tokens`, `llm`. HTTP 200 for answers and abstentions; 422 `invalid_plan`; 503 `planner_unavailable`; 504 `timeout`; 500 `compile_error`, `sql_rejected`, `execution_error`.
- Compiled SQL runs as `POSTGRES_READONLY_USER` with `reader.statement_timeout_ms` (5 s) and the row cap `sql.max_limit` (1000). Frame images: `GET /v1/frames/{frame_id}/image` (the service reads the object store; browsers need no S3 access). Cameras and zones: `GET /v1/catalog`, re-read from Postgres every `catalog_ttl_s`.
- Every question is a row in `query_log` (written as `POSTGRES_USER`; the read-only login cannot read it): `SELECT ts, question, success, error_type, latency_ms, timings, llm FROM query_log ORDER BY ts DESC LIMIT 20`.
- Metrics on `/metrics`: `vms_query_requests_total{status,error_type}`, `vms_query_stage_duration_seconds{stage}`, `vms_query_llm_tokens_total`, `vms_query_plan_repairs_total`, `vms_query_query_log_failures_total`.

### UI

```bash
make app           # build the app image and start the query service + UI in compose: http://localhost:8070
make query         # or on the host; serves the UI too once it is built:
make ui-build      #   npm ci && npm run build into frontend/dist (needs Node 22)
make ui-dev        # dev server with hot reload on http://localhost:5173, proxying /v1 to :8070
make ui-check      # frontend typecheck + vitest (CI runs the same in the frontend job)
```

- The UI is served by the query service at `/`; there is no separate web server. Without `frontend/dist` (host mode, not built), `/` answers 404 and the API still works.
- Camera wall empty ("waiting for frames"): nothing has been processed for that camera yet; it needs ingest, a worker and frames from loadgen (`make demo`). Tiles show how long ago their frame was captured: a growing age means the worker is behind.
- Example chips are the canned planner's questions (`configs/demo/canned_plans.v1.jsonl`); with a real LLM any question works.
- The containerized query service reaches an LLM on the host at `http://host.docker.internal:11434/v1` (the compose default for `LLM_BASE_URL`). On Linux, Ollama listens on 127.0.0.1 by default and is unreachable from containers: start it with `OLLAMA_HOST=0.0.0.0`, or run `make query` on the host instead.
- Image builds behind a TLS-intercepting proxy: copy the proxy's CA into the repo (`.certs/proxy-ca.pem`, gitignored) and set `EXTRA_CA_CERT=../../.certs/proxy-ca.pem` in `.env` (relative to `deploy/compose/`); compose passes it to the build as a secret, never stored in the image. A file outside the repo fails with "additional privileges requested" on newer Compose (E-023).
- Screenshots for the README: `cd frontend && CHROMIUM_PATH=/path/to/chrome npm run screenshots -- http://localhost:8070 ../docs/images` while the demo runs.

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
- Drift injection (task 9.2): a scenario's `config.drift.events` changes what some cameras send from a known onset on (`vms.loadgen.drift`: darken, blur, noise, shift, occlusion with a `strength`, or `switch_clip` to another clip; `ramp_s` 0 is abrupt, otherwise linear over `ramp_s`; `onset_jitter_s` adds a seeded per-camera delay). One scenario per kind plus a clean control, all on full-length clips (`make data` first): `make loadgen SCENARIO=bench/scenarios/drift-darken.yaml` (also `drift-blur`, `drift-noise`, `drift-shift`, `drift-occlusion`, `drift-classmix`, `drift-clean`; 6 min per repetition, 3 repetitions) against the demo stack with the drift monitor. Each repetition's results list the resolved onsets (`repetitions[].drift.events`: camera, kind, `onset_s`, `onset_at`, ramp, strength) and the render time of drifted frames (`render_ms`). Each scenario uses its own camera prefix, so its cameras get their own drift references; the first repetition's first 60 s is the reference for all three. Each repetition also posts its onsets to Grafana as annotations for the Drift dashboard (`annotations` in `configs/loadgen.yaml`; best effort, counted under `repetitions[].drift.annotations`).

### Traces and metrics

Every service exports spans to the collector (`OTEL_EXPORTER_OTLP_ENDPOINT`) and serves `/metrics`. What is traced and measured: ARCHITECTURE 4.11 and D25.

- **Find a trace.** Grafana > Explore > Tempo, query type TraceQL (or `curl -G localhost:3200/api/search --data-urlencode 'q=...'`):
  - a whole frame, loadgen to DB write: `{resource.service.name="loadgen"} >> {name="worker.frame"}`
  - slow frames: `{name="worker.frame" && span.freshness_s > 2.0}` (a float literal: `> 2` matches nothing, the attribute is a double)
  - one camera: `{name="worker.frame" && span.camera_id="cam-001"}`
  - questions that needed a repair: `{name="query.llm" && span.attempt = 2}`
  - Traces become searchable 1 to 2 minutes after they end (E-024); fetching one by ID works at once.
- **Sampling.** Loadgen decides per frame (`telemetry.sample_ratio` in `configs/loadgen.yaml`, 1%; `VMS_TELEMETRY__SAMPLE_RATIO=0.1` for more); ingest and the worker follow that decision. Every question is traced (`configs/query.yaml`).
- **Metrics.** Prometheus at http://localhost:9090 scrapes `ingest`, `worker`, `drift` and `query` containers every 5 s, found by Docker DNS (`deploy/compose/prometheus/prometheus.yml`). Status > Targets lists them; a job with no target means that container is not running. Containers stopped after they ran (`compose stop`, `rm`) stay listed as down targets, because Prometheus keeps a DNS name's last answer when the lookup fails; `make test-integration` then fails `test_prometheus_scrapes_every_target`. Restart Prometheus (`docker compose -f deploy/compose/docker-compose.yml restart prometheus`) or use `make down && make up`. Services started on the host (`make ingest`, `make worker`, `make query`) are not scraped: use their `/metrics` directly, or the containers. Useful queries:
  - freshness p95: `histogram_quantile(0.95, sum by (le) (rate(vms_worker_capture_to_done_seconds_bucket[1m])))`
  - share of frames within the 2 s SLO: `sum(rate(vms_worker_capture_to_done_seconds_bucket{le="2.0"}[5m])) / sum(rate(vms_worker_capture_to_done_seconds_count[5m]))`
  - ingest rate by outcome: `sum by (outcome) (rate(vms_ingest_frames_total[1m]))`
  - backlog and its age: `max(vms_worker_backlog_frames)`, `max(vms_worker_backlog_age_seconds)` (every worker reports the same global value: use `max`, never `sum`)
  - worker time per stage: `sum by (stage) (rate(vms_worker_stage_duration_seconds_sum[1m]))`
- **Check.** `uv run python -m vms.observe [--since <unix time>]` passes when every required metric has a series and Tempo has a complete frame trace and a complete query trace (since the given time, else the last hour). It waits up to `--timeout-s` (240) for Tempo.

## Benchmarks

Every reported number comes from a scenario in `bench/scenarios/` run through `make bench` (methodology: `docs/BENCHMARKS.md`).

```bash
make demo DEMO_SECONDS=30                                  # stack + ingest, worker, query service
make bench                                                 # bench/scenarios/bench-smoke.yaml (~5 min)
make bench SCENARIO=bench/scenarios/<name>.yaml            # any kind: e2e, run, selftest, model
make bench-report RESULTS="bench/results/<file>.json ..."  # Markdown tables for BENCHMARKS.md
```

- Writing a scenario: write the YAML before running it. `kind: e2e` takes `repeats` (at least 3 for a reported number), `warmup_s`, `hold_s` (at least `min_window_s`, 15 s), `ingest: {cameras: [...], fps, loadgen: {...overrides of configs/loadgen.yaml}}` and/or `query: {concurrency: [...], questions, think_time_s}`; the docstring of `vms/bench/scenario.py` has an example. Change one variable at a time against a named baseline scenario.
- What it needs: the e2e runner talks to the endpoints in `configs/bench.yaml` (ingest 8080, query 8070, Prometheus 9090; `VMS_INGEST_URL`, `VMS_QUERY_URL`, `VMS_PROMETHEUS_URL` override). The services must run in containers (`make demo` or `make app` plus `compose --profile app up ingest worker`): Prometheus does not scrape host processes, so a step against `make worker` has no freshness and fails. The canned planner keeps query load free; with `LLM_*` set the questions cost tokens.
- While it runs: one JSON log line per step with its freshness p95, lost frames and verdict. Ctrl-C finishes the current step and writes partial results (`interrupted: true`). Nothing else should send frames or questions meanwhile; the database is shared, so a demo running alongside changes the numbers.
- Results: `bench/results/<date>-<name>-<run_id>.json`, with the Markdown table printed at the end. Commit the files behind reported numbers, log the run in `EXPERIMENTS.md`, and put the curated number in `BENCHMARKS.md` with links to both.
- Before a long run: check free disk (`df -h /`). A full disk makes SeaweedFS refuse writes, every frame then fails at ingest, and the object store needs `docker restart vms-objectstore-1` after space is freed (E-038). Each 5 FPS camera writes about 3 GB per hour of frames.
- Worker ramps (5.5): start the worker the scenario expects, then run it, e.g. `LLM_MODEL= WORKER_BACKEND=ultralytics WORKER_INFER_BATCH=1 docker compose --env-file .env -f deploy/compose/docker-compose.yml --profile app up -d --wait ingest worker query` and `make bench SCENARIO=bench/scenarios/ramp-cpu-pt.yaml`. The others: `ramp-cpu-ort` (`WORKER_BACKEND=onnxruntime WORKER_INFER_BATCH=1 WORKER_SPIN=true`), `ramp-cpu-ort-b4` and `-ext` (`... WORKER_INFER_BATCH=4 WORKER_SPIN=true`), `ramp-cpu-ort-b4-nospin` (the compose defaults). The E-039 `ramp-cpu.yaml` predates `expect_worker`, so start it with the PyTorch settings by hand. About 35 to 45 min each.
- Pinning the worker: a scenario's `expect_worker: {backend, precision, infer_batch}` is checked against `vms_worker_model_info` before the first step, and the run stops with an error if the running worker differs (for example, compose started with the wrong `WORKER_BACKEND`). Every e2e results file records the worker's labels under `metrics.workers`.
- Before trusting a step: `harness_ok` true (loadgen kept its schedule, the query clients used under 0.8 cores), `drain_s` small (else the previous step's backlog spilled over) and the hardware recorded in the file matches the table you add it to.

### Profiling

```bash
make bench SCENARIO=bench/scenarios/profile-worker-cpu.yaml   # worker at saturation (~2 min)
make bench SCENARIO=bench/scenarios/profile-query-cpu.yaml    # query service, canned planner (~2 min)
gunzip -k bench/profiles/<file>.speedscope.json.gz            # then open it at https://www.speedscope.app
```

- `kind: profile` (`vms.bench.profile`) stops the service's container (`docker compose stop worker|query`), runs the same code on the host as a child of `py-spy record` (no ptrace privileges needed for a child), drives loadgen or query clients at it, then sends SIGTERM and starts the container again, also on failure. Needs the stack and the other app containers up, as for e2e scenarios.
- Results: `bench/results/<date>-<name>-<run_id>.json` with seconds and shares per category and per thread, and the profile in `bench/profiles/` (gzipped speedscope, tens of KB). Categories are regexes in the scenario, matched against our own frames innermost first (`any_frame: true` also matches library frames, to split a stage).
- Caveats: the host process is not scraped by Prometheus (no freshness during a profile); the profile includes the warmup; samples are wall time of running Python threads, so native threads (PyTorch's own pool) do not appear.
- If a profile run is interrupted before its `finally`, start the container by hand: `docker compose --env-file .env -f deploy/compose/docker-compose.yml --profile app start worker`.

## Evaluation

### Structured-query eval set (task 3.1)

The questions, gold answers and splits live in `eval/structured/`; the database they are answered on is a synthetic snapshot (D29).

```bash
make up
make eval-db         # (re)create the vms_eval database from snapshot v1 (~15 s)
make eval-build      # run every gold SQL and gold plan on it; write dev/test files and the manifest
make eval-check      # offline: items valid, no few-shot leakage, splits and hashes in step (also a unit test)
make eval-review                 # print the test split for review: question, plan, SQL, expected rows
make eval-review REVIEW=dev
```

- `make eval-db` drops and recreates `vms_eval` (configs/eval.yaml `db`) on the stack's Postgres: migrations, the read-only login, the eval catalog (`configs/eval/cameras.v1.yaml`), then COPY of `eval/structured/snapshot.v1/`. The snapshot comes from `make data-pull` when the DVC remote has it; otherwise it is generated from `configs/eval/snapshot.v1.yaml` (~5 s). Either way its files must match the sha256s in `eval/structured/snapshot.v1.json`. The dev database is never touched.
- Anything that answers eval questions must use the eval clock, `now` in `snapshot.v1.json` (2026-03-31 14:30 UTC): "today" and "last week" are relative to it, and gold SQL binds it as `%(now)s`.
- Adding or editing questions: edit `eval/structured/items.v1.jsonl` (one item per line: `id`, `category`, optional `group` for paraphrases, `question`, gold `plan`, `expressible_by_plan`, gold `sql` unless the question has no answer, `ordered`, `tags`, `notes`), then `make eval-build`. It fails on any disagreement between the gold SQL and the gold plan, a gold SQL the validator rejects, a result at the validator's injected limit (possibly truncated), or a near duplicate of a few-shot question. Items keep the side recorded in `eval/structured/splits.v1.json` (the reviewed split); a new item is assigned around those pins and added to the file by the build. Editing a test item still changes the test hash: do it before the first reported test number, then update BENCHMARKS.md, and never after (make `items.v2` instead).
- The snapshot is frozen. Changing `configs/eval/snapshot.v1.yaml` or `cameras.v1.yaml` fails a unit test; a new snapshot is a new version (spec, catalog, items, splits), generated with `make eval-snapshot` and pushed with `uv run dvc add eval/structured/snapshot.v<N> && uv run dvc push`.
- Review before the test split is used: read every test item and a sample of dev with `make eval-review`; check that the question reads as the plan and SQL interpret it, and that the expected rows are plausible. Fix disagreements in the items file and rebuild.

### Running evals (task 3.2)

```bash
make eval-db                                    # once per snapshot version
make eval PLANNER=gold                          # harness self-test: exits 1 unless 100% with no breach
make eval SPLIT=dev                             # the configured LLM (LLM_* in .env); costs tokens
make eval-report RESULTS="eval/results/a.json eval/results/b.json"   # side by side, for BENCHMARKS.md
make eval SPLIT=dev LIMIT=5                     # smoke run; marked partial, not reportable
```

- The engine is the query service's typed-plan engine, built in-process against `vms_eval` as the read-only login, with the clock pinned to the eval clock. Nothing is written to `query_log`.
- Another model: set `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY` for the command (and `VMS_LLM__RESPONSE_FORMAT=json_object` for providers without `json_schema`, such as DeepSeek). Add its price per million tokens under `prices` in `configs/eval.yaml` to get a cost.
- The run refuses a split file whose hash differs from `dataset.v1.json`, and an eval database that does not hold the snapshot. Run the test split only for numbers that will be reported, once per configuration; tune on dev.
- Red-team questions run one at a time, each between two database fingerprints; the run reports any change as a breach.
- Commit the results files behind reported numbers; log each run in EXPERIMENTS.md.
- CI runs the gold self-test on dev after `make eval-db` in the integration job, with results written to the runner's temp dir (`VMS_RUN__RESULTS_DIR`). If it fails there, the failing item ids and error classes are printed above the error line; reproduce with `make eval PLANNER=gold` locally.

### Detection eval subset (task 3.4)

```bash
make eval-detection-data                    # fetch and verify the 573 listed images (~95 MB, about a minute)
make eval SUITE=detection                   # yolo11s (configs/eval-detection.yaml), ~1.5 min on the laptop
make eval SUITE=detection WEIGHTS=yolo11n   # any key of configs/models.yaml (make models first)
make eval SUITE=detection BACKEND=onnxruntime PRECISION=int8   # another backend (precision ladder below)
uv run python -m vms.eval.detection check   # offline: spec and list match the manifest (also a unit test)
```

- No stack, database or LLM needed. Images land in `data/eval/detection/v1/` (gitignored), each checked against the sha256 in `eval/detection/subset.v1.jsonl`; a missing or corrupt image is fetched again, and `make eval SUITE=detection` refuses to run on an incomplete set. COCO comes from the path-style S3 URL because `images.cocodataset.org` serves a certificate for another name.
- Each run writes `eval/results/<date>-detection-<weights>-<run_id>.json` (mAP@50 and mAP@50-95 per source with interval, bootstrap std and mean, per-class AP) and the raw detections to `data/eval/detection/predictions/<run_id>.jsonl.gz`. A smoke run on a few images (`VMS_MAX_IMAGES=20`) is marked `partial` and is never a reported number.
- Licenses: COCO annotations are CC BY 4.0 and its images keep their Flickr licenses; MOT17 is CC BY-NC-SA 3.0. Neither is redistributed; never commit or publish the images.
- The subset is frozen. Changing `configs/eval/detection.v1.yaml` or the list fails a unit test; a new selection is v2 (`detection.v2.yaml`, then `uv run python -m vms.eval.detection build` with the v2 paths, which writes the list and manifest once and on later runs only checks that a rebuild is identical). Images used to calibrate INT8 (5.2) or fine-tune (9.5) must not be in the list.

### Detection quality under drift (task 9.2)

```bash
make eval-drift          # bench/scenarios/drift-map-cpu.yaml: yolo11n ORT FP32, clean + 6 conditions (~20 min on the laptop)
make eval-drift SCENARIO=bench/scenarios/<other>.yaml
```

- Needs `make models`, `make onnx`, `make eval-detection-data`. Writes one `eval.detection` results JSON per condition (`metrics.drift` names the transform) and a summary `eval/results/*-drift-map-cpu-*.json` (`kind: eval.detection.drift`) with each condition's mAP and its paired change against the clean run.
- Only photometric drift (darken, blur, noise) is scored: it keeps every box valid. Noise is seeded per image (`drift.seed`), so reruns give the same pixels.

### Drift detector scoring (task 9.3)

```bash
make drift-score                                  # bench/scenarios/drift-detect-cpu.yaml (~15 s once series are cached)
make drift-score SCENARIO=bench/scenarios/<other>.yaml
```

- Needs the listed loadgen runs' frames and detections still in Postgres (the stack that recorded them): each camera is replayed from the database by its recorded camera ids and start time. Series are cached per run and repetition in `data/drift/series/` (delete to recompute). A camera with no processed frames stops the run with an error (frames still pending, or the worker never leased the camera: check `SELECT status, count(*) FROM frames GROUP BY 1`).
- Writes `bench/results/*-drift-detect-cpu-*.json` (`kind: drift.score`: calibration, the full sweep with per-repetition summaries, the leave-one-repetition-out folds, delays and first channels by drift) and the plot `docs/images/drift-delay-vs-false-alarms.png`.
- Recording the 7 runs takes about 2 h on the laptop (`uv run python -m vms.loadgen run --scenario bench/scenarios/drift-<kind>.yaml`, 18 min each). Camera ids repeat across reruns of a scenario; the monitor keeps their first reference, the scorer does not use it.

### Precision ladder (task 5.2)

```bash
make onnx                                   # FP32 export (once per weights)
make calibration-data                       # 128 calibration images into data/calibration/v1/ (~19 MB)
make quantize WEIGHTS=yolo11s               # models/onnx/yolo11s.int8.onnx + sidecar (~20 s)
make eval SUITE=detection BACKEND=onnxruntime PRECISION=int8 NAME=detection-yolo11s-int8
make eval-detection-compare BASE=eval/results/<fp32>.json OTHER=eval/results/<int8>.json
make bench SCENARIO=bench/scenarios/ladder-cpu.yaml   # speed per backend and batch size (~25 min)
VMS_DETECTOR__BACKEND=onnxruntime VMS_DETECTOR__PRECISION=int8 make worker
```

- Calibration set: `configs/calibration.v1.yaml` (64 COCO train2017 images, 16 frames each of MOT17-05/10/11/13), frozen into `configs/calibration.v1.lock.jsonl`; a unit test fails if the list changes or overlaps the detection eval subset. A new selection is v2 (`uv run python -m vms.worker calibration build` with the v2 paths; on an existing list it only checks the rebuild is identical).
- Quantization settings are in `configs/quantize.yaml` (`VMS_INT8__*` overrides, e.g. `VMS_INT8__EXCLUDE_NODES='[]'` quantizes the whole graph). The sidecar `yolo11s.int8.onnx.json` records the settings, the excluded nodes, the op types that run in INT8, and the hashes of the weights, the FP32 export and the calibration list; the worker refuses an INT8 model whose inputs changed ("run `make quantize`"). `model_version` ends in `+onnx-int8@<sha256 prefix>`, so eval results name the exact model.
- `compare` re-scores two eval runs from their saved detections with the same bootstrap resamples (no inference) and writes `eval/results/<date>-<name>-<run_id>.json` with the paired difference per source and the D32 verdict (`d32_pass`). It refuses runs scored on different terms (subset, sources, bootstrap, max detections) and partial runs.
- The ladder runs each step in its own process and rotates the step order per repetition. For numbers worth reporting: AC power, nothing else heavy running, and the compose stack stopped (`docker stop` the `vms-*` containers; `docker start` them after). The laptop's CPU is a hybrid (2 performance + 8 efficiency cores), so expect a few percent between repetitions.

**GPU rungs (D38).** Run on a Runpod session, not locally (the laptop has no GPU): `make runpod-session` runs `ladder-gpu-n` (PyTorch CUDA, ONNX Runtime CUDA, TensorRT FP16, TensorRT INT8; then each step's mAP and the paired comparisons, D32 for INT8 against TensorRT FP16) and the two TensorRT batch sweeps. `scripts/pod_remote.sh setup` installs the CUDA 13 builds (torch cu130, `onnxruntime-gpu[cuda,cudnn]==1.31.0`, `tensorrt-cu13==11.3.0.99`), fetches the calibration and eval images and runs `make trt-prepare WEIGHTS=yolo11n`, which makes whatever is missing: the FP16 ONNX export (`yolo11n.fp16.onnx`, exported on the GPU), the symmetric INT8 model for TensorRT (`yolo11n.int8-trt.onnx`; `make quantize` settings under `int8_tensorrt` in `configs/quantize.yaml`, same calibration list and excluded nodes as the CPU model), and one engine per precision in `models/trt/` (profile in `configs/tensorrt.yaml`). Each engine's sidecar names its source model, TensorRT version and GPU; the worker and the eval refuse an engine that does not match ("run `make trt-prepare`"). On a GPU host by hand: `VMS_DETECTOR__BACKEND=tensorrt VMS_DETECTOR__PRECISION=fp16 VMS_DETECTOR__DEVICE=cuda make eval SUITE=detection`.

## GPU sessions

AWS denied the GPU quota (EC2 and SageMaker, D35), so **GPU sessions run on Runpod pods** (D37): one pod per session, created, used and deleted by `scripts/runpod_session.sh` (`python -m vms.cloud.pod_cli`). A pod is a container, so only in-process scenarios run there (`kind: model` and `kind: ladder`: the precision ladder, batch sweeps, model-only latency); end-to-end scenarios are refused and the GPU end-to-end ramp is pending. CPU sessions stay on AWS ([CPU sessions](#cpu-sessions-d35)); the EC2 GPU path below them is built and waits for a quota.

### Runpod GPU sessions (D37)

Settings: `configs/runpod.yaml` (GPU type, cloud tier, CUDA floor, image digest, disk, time limit). Override per run with `VMS_GPU_TYPE="NVIDIA GeForce RTX 4090"`, `VMS_CLOUD=COMMUNITY` or `VMS_MAX_SESSION_MINUTES=90`; a different GPU is a new baseline, never mixed with A40 numbers.

**Cost.** 1x A40 (48 GB) on the secure cloud: $0.59 per hour from the catalog on 2026-10-19, plus 40 GB of container disk (about $0.005/h), billed per second while the pod exists. Preflight prints the live price. Setup (uv, the environment, the ~3 GB of CUDA wheels, weights) takes roughly 10 minutes before the first scenario. The pod deletes itself after 150 minutes whatever happens, so one session costs at most about $1.50. Each results file records its cost; the session's own file (`bench/results/<date>-gpu-session-<id>.json`, kind `cloud.pod_session`) has the total.

Prerequisites, once:
1. Prepaid credit with **auto-pay off**, and a low-balance email alert (Runpod console > Billing). The balance is the hard spend cap: at $0 every pod stops. Credits are not refundable, so top up small amounts ($20 on 2026-10-19).
2. An API key: Runpod console > Settings > API Keys, with read and write access to pods. Put it in `.env` as `RUNPOD_API_KEY=` (gitignored). It stays on this machine; pods get their own pod-scoped key.
3. `rsync` and `ssh` on this machine, and the mini clips (`make data-mini`; they are copied to the pod).

Running a session:
```bash
make runpod-preflight                 # free: API key, GPU price and stock, pods left from earlier sessions
git status                            # must be clean: results carry the commit's SHA (--allow-dirty for a test)
make runpod-session                   # RUNPOD_SCENARIOS="bench/scenarios/a.yaml ..." to choose; asks first
scripts/runpod_session.sh run --yes bench/scenarios/worker-model-gpu-n.yaml   # no prompt (scripted use)
```
`make runpod-session` prints the GPU, cloud tier, stock, CUDA floor, image, hourly cost and cap, and creates the pod only after you type `launch`. Then: a per-session SSH key, the pod (`vms-gpu-<id>`, port 22 published, no volume), direct SSH ready, the time-limit check (the session stops before running anything if the pod's self-delete timer is not armed with a pod-scoped key), code and clips synced, `scripts/pod_remote.sh setup` (output streams here), one `scripts/pod_remote.sh run <scenario>` per scenario, results copied into `bench/results/`, pod deleted, verification. A failing scenario is recorded and the next one still runs. Ctrl-C at any point tears down. Per-session files (private key, known_hosts, state) are in `.gpu-sessions/<id>/` (gitignored).

Exit codes: 0 all good; 1 setup or a scenario failed; 2 **a pod remains after teardown**; 3 not confirmed (nothing created); 4 preflight failed (nothing created); 130 interrupted.

If a session reports exit 2, or this machine died mid-session: `make runpod-teardown SESSION=<id>` (from the saved state and the pod's name), then `make runpod-list` must print "no project pods exist"; check the Runpod console's Pods page too. To look at a running pod: `ssh -i .gpu-sessions/<id>/id_ed25519 -o UserKnownHostsFile=.gpu-sessions/<id>/known_hosts -p <ssh_port> root@<ssh_host>` (both in `.gpu-sessions/<id>/state.json`).

After a session: check the session file's `metrics.teardown_verified` is true and `metrics.time_limit` starts with `armed ... key=yes`, compare the cost with Runpod's billing page, log the run in `EXPERIMENTS.md`, and commit the results files behind any reported number.

### EC2 GPU sessions (D28; waiting for a quota)

One on-demand GPU instance per session, launched, used and destroyed by `scripts/gpu_session.sh` (`python -m vms.cloud`; design in ARCHITECTURE 9, D28). Settings: `configs/cloud.yaml`. Not usable while the account's G and VT quota is 0; kept because the CPU sessions share the driver.

**Cost.** g6.xlarge (1x L4, 4 vCPUs) costs about **$0.82 per hour** in us-east-2 (instance $0.8048 + 100 GB gp3 + public IPv4), billed per second with a 60 s minimum. A session needs roughly 15 to 25 minutes of setup (tools, `make setup`, the stack, the CUDA image) before the first scenario. The instance terminates itself after 150 minutes whatever happens, so one session costs at most about $2.05. Each results file records its cost; the session's own file (`bench/results/<date>-gpu-session-<id>.json`) has the total.

Prerequisites, once:
1. AWS credentials for the project: `aws login` (or `AWS_PROFILE` in the shell). The Region is the project's selected Region (AWS Settings > View all projects > Overview > Additional Info > Region; us-east-2 here, also in `~/.aws/config`).
2. On-demand G and VT vCPU quota (`L-DB2E81BA`, Service Quotas > Amazon EC2 > "Running On-Demand G and VT instances") of at least 4 in that Region. Spot is not an option on this account: the managed SCP denies it (D28).
3. The budget alarm, already created on 2026-10-13: `scripts/gpu_session.sh budget --email <address>` (`vms-monthly`, $20/month, usage before credits; a second run changes nothing).
4. `rsync` and `ssh` on this machine, and the mini clips (`make data-mini`; they are copied to the instance).

Running a session:
```bash
make gpu-preflight                    # free: credentials, Region, quota, AMI, dry-run launch, leftovers
git status                            # must be clean: results carry the commit's SHA (--allow-dirty for a test)
make gpu-session                      # GPU_SCENARIOS="bench/scenarios/a.yaml ..." to choose; asks first
scripts/gpu_session.sh run --yes bench/scenarios/gpu-smoke.yaml   # no prompt (scripted use)
```
`make gpu-session` prints the account, Region, instance type, AMI, hourly cost and cap, and launches only after you type `launch`. Then: key pair and security group (SSH from your public IP only), instance, SSH ready, code and clips synced, `scripts/gpu_remote.sh setup` (output streams here), one `scripts/gpu_remote.sh run <scenario>` per scenario (e2e scenarios run `make bench` on the instance; `kind: model` scenarios run inside the CUDA worker image), results copied into `bench/results/`, teardown, verification. A failing scenario is recorded and the next one still runs. Ctrl-C at any point tears down (a second Ctrl-C during teardown is ignored). Per-session files (private key, known_hosts, state) are in `.gpu-sessions/<id>/` (gitignored).

Exit codes: 0 all good; 1 setup or a scenario failed; 2 **resources remain after teardown**; 3 not confirmed (nothing created); 4 preflight failed (nothing created); 130 interrupted.

While it runs, to look at the instance yourself: `ssh -i .gpu-sessions/<id>/id_ed25519 -o UserKnownHostsFile=.gpu-sessions/<id>/known_hosts ubuntu@<public_ip>` (the IP is in `.gpu-sessions/<id>/state.json`); Grafana and Prometheus are on the instance's localhost (`ssh -L 3000:localhost:3000 ...`).

After a session: check the session file's `metrics.teardown_verified` is true, log the run in `EXPERIMENTS.md` with its cost, and commit the results files behind any reported number.

### CPU sessions (D35)

The same driver on a CPU instance: `machines.cpu` in `configs/cloud.yaml` (c7i.2xlarge, Ubuntu 24.04 from Canonical's SSM parameter, 60 GB gp3; Docker, Compose and buildx from the Ubuntu archive, installed by `host_setup` before anything else). It needs the on-demand **standard** vCPU quota (`L-1216C47A`, "Running On-Demand Standard (A, C, D, H, I, M, R, T, Z) instances"), not the G and VT one. Cost: c7i.2xlarge about **$0.37 per hour** with volume and IPv4, c7i.4xlarge (`INSTANCE=c7i.4xlarge`) about $0.72; the 150-minute cap holds (raise it for one session with `VMS_MAX_SESSION_MINUTES=240`).
```bash
make cpu-preflight                              # INSTANCE=c7i.4xlarge for another priced type
make cpu-session                                # CPU_SCENARIOS: ramp, query load, ladder, batch sweep
make cpu-session INSTANCE=c7i.4xlarge CPU_SCENARIOS="bench/scenarios/finetune-blur-cpu.yaml"
```
On a CPU machine `scripts/gpu_remote.sh` (with `VMS_ACCELERATOR=cpu`) brings up the same images and services as a laptop, runs `make onnx`, `make calibration-data` and `make quantize WEIGHTS=yolo11n` (the worker's ONNX Runtime backend and the ladder's INT8 step), runs every scenario on the host through `make bench`, and fetches the training and eval images before a `kind: finetune` scenario. Besides `bench/results/`, the session copies back `eval/results/`, the eval's raw detections and `models/finetune/` (`remote.collect`); after it, `dvc add models/finetune/*.pt` and commit the `.dvc` files. Each scenario has a one-hour limit (`timeouts.scenario_s`); the fine-tune takes about 1 h 15 min on c7i.4xlarge (two 16-minute trainings, data, twelve evals), so raise it for that session: `VMS_TIMEOUTS__SCENARIO_S=10800`. If a fine-tune session stops after training, `python -m vms.finetune evaluate --scenario bench/scenarios/finetune-blur-cpu.yaml` scores the copied-back weights locally (checked against their training records) and writes the summary.

## Deployment

_Placeholder (tasks 6.4, 10.1, 10.2)._ Helm chart, kind smoke test, Terraform for the AWS environment, and the CD pipeline with its eval gate.

## Operations

### Dashboards

Grafana (http://localhost:3000, anonymous viewer) opens on **VMS / Pipeline**; the "VMS dashboards" menu at the top switches between them. Every panel's (i) tooltip says what it shows and what to do about it.

| Dashboard | Answers | Look first at |
|---|---|---|
| Pipeline | Is the system keeping up, and how fresh are results? | "Within SLO" and "Oldest waiting"; then "Where freshness goes" (waiting vs working) |
| Inference | Where does a worker's time go per frame? | "Where worker time goes" (stage shares), then "Stage latency p95" |
| Query | Are questions fast and answered? | "Question latency" against the 1.5 s line, then "Time per stage" (LLM vs database) |
| Resources | Which process is short of CPU or memory? | "CPU per service" against the machine's cores |
| Drift | Has a camera's input or output changed from its reference, and since when? | "Cameras alarming" and "Alarm by camera"; then that camera's row (pick it in the Camera menu) |

- Dashboards are code: edit `src/vms/observe/dashboards.py`, run `make dashboards`, and Grafana picks up the regenerated JSON in `deploy/grafana/dashboards/` within 10 s (no restart). Edits made in the Grafana UI cannot be saved (provisioned, read-only); a unit test fails if the JSON and the code disagree.
- Rates use `$__rate_interval` (20 s at the 5 s scrape interval). Quantile panels show gaps when there was no traffic: no frames or questions means no latency, not the last value.
- GPU panels (Inference, collapsed row) fill only on the GPU instance, where the DCGM exporter runs (2.4).
- `make demo`'s telemetry check also fails if any panel (other than GPU) has no data; the Drift dashboard's `$camera` is checked as "all cameras".
- Drift dashboard markers: red regions are drift alarms (from the alarm gauge); blue markers are injected drift onsets, posted by the load generator to Grafana when a drift scenario's repetition starts (a region over the ramp for gradual drift). Toggle either at the top of the dashboard. Onsets need `GRAFANA_ADMIN_USER`/`GRAFANA_ADMIN_PASSWORD` (`.env`) to match the stack; otherwise the run logs a warning and its results show `drift.annotations.failed`. Onset markers stay in Grafana's database (volume `grafana-data`); delete them under Dashboards > Annotations or with `DELETE /api/annotations/<id>`.
- Screenshots for the README and write-up, while there is live data (for example during or right after `make demo`): `cd frontend && CHROMIUM_PATH=/path/to/chrome npm run screenshots:grafana -- http://localhost:3000 ../docs/images [from] [to]`, with `from` and `to` as Grafana time values (Unix ms of the run, or `now-15m`). Kiosk mode, light theme, 1600 px wide. `DASHBOARDS=vms-drift` takes only the listed dashboards; `DRIFT_CAMERAS=drift-darken-001,drift-darken-003` picks the Drift dashboard's cameras (one repeated section each, all of them in the screenshot), as for `docs/images/grafana-drift.png` (E-056).

Alerts and the drift procedure are under Running locally, "Drift alerts". _Placeholder (deferred 6.5, 10.2):_ scaling, and promoting a model to champion.

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
- `make zones` or `make ui-build` fails with `npm error notsup ... Required: {"node":">=22.12.0"}`: the host Node is too old. Before the engine check existed this showed up as `Cannot find module '@rolldown/binding-linux-x64-gnu'`, because npm silently skips optional native bindings whose `engines` the running Node does not meet. Install the pinned Node (`cd frontend && nvm install && nvm alias default 22.22.0`), open a new shell and re-run.
- The UI answers every question that is not an example chip with "No LLM is configured, so only the example questions can be answered": `LLM_MODEL` is empty, so the canned planner is in use (the plan panel shows `canned`). Set `LLM_BASE_URL` and `LLM_MODEL` in `.env` (for example a local Ollama model) and restart the query service.
- Query service answers 503 `planner_unavailable`: the LLM at `LLM_BASE_URL` is unreachable or timing out (`llm.timeout_s`, after `llm.max_retries`). Check `curl $LLM_BASE_URL/models`, or unset `LLM_MODEL` to use the canned planner.
- Query service answers 422 `invalid_plan` for most questions with a hosted model: the provider may ignore or reject `json_schema`; look at `error.errors` in the response and try `VMS_LLM__RESPONSE_FORMAT=json_object`.
- Query service `/readyz` returns 503 with `postgres: unavailable`: the read-only login is missing or its password differs from `.env`. Run `make migrate`.
- `make loadgen` reports every frame failed with reason `connection`: nothing listens on `target_url`. Start the ingest service or `make loadgen-stub`.
- `make loadgen` logs "generator fell behind its schedule": the host or the generator is saturated, so the run offered less load than configured. Reduce cameras, close other load on the machine, or run several loadgen processes with different `camera_prefix` values.
- `make bench` fails with "backlog still N frames after 900 s": the worker cannot drain what an earlier step or run left. Wait, or start from an empty database (`make down-volumes`). A step with freshness `null` and `ingest_ok: false` means no frame reached the worker in the window: is the worker container running, and scraped (`make ps`, Prometheus targets)?
- `python -m vms.observe` (the demo's last step) reports "no healthy target": the named container is not running or not ready (`make ps`). "No series for" a metric: the service runs but has not done that work yet (for example no question asked), or runs on the host where Prometheus does not scrape it. "No complete frame trace": check that loadgen, ingest and the worker all export (`docker logs vms-worker-1` for OTLP export errors; `VMS_TELEMETRY__ENABLED` must not be false) and that the frames were sampled (`DEMO_TRACE_RATIO`); then open the newest loadgen trace in Grafana to see where the chain stops.
- `make eval` stops with "eval database ... not snapshot v1" or "unavailable": run `make eval-db`. "does not match dataset.v1.json": the split files changed after the build; run `make eval-build` (and update the test hash in BENCHMARKS.md if the test file changed).
- `make eval` with a local Ollama model is very slow or scores near 0: Ollama's default context (4,096 tokens here) is shorter than the planner prompt (~4,600), so the prompt is cut; raise the server's context (`OLLAMA_CONTEXT_LENGTH=8192 ollama serve`) and expect minutes per question on a laptop CPU (E-031).
- `make eval-db` fails with "snapshot does not match its manifest": the files under `eval/structured/snapshot.v1/` are stale or were regenerated by a changed generator. Delete the directory and re-run (it regenerates); if the regenerated files still differ, the generator's output changed (a code or Python change): restore it, since every gold answer depends on those bytes.
- `make eval-build` reports "gold plan and gold SQL disagree": one of the two golds misreads the question, or the compiler changed semantics. Run both on `vms_eval` (`psql`, binding `now` from the manifest) and compare; fix the wrong gold, or record a compiler change as a decision.
- Sudden `AccessDenied` errors on AWS operations that used to work: check the project's spend limit status in AWS Settings > Billing.
- `make gpu-preflight` fails with "quota ... is 0": request the on-demand G and VT vCPU quota (`L-DB2E81BA`) in Service Quotas; a spot quota does not help (spot is denied by the SCP). "A dry-run launch was refused ... service control policy": the instance type or a launch option is denied by the managed SCP. "Missing Dependency ... botocore[crt]": run `make setup` (the `aws login` credential provider needs it).
- A GPU session fails in setup at the CUDA check ("no CUDA"): the instance's NVIDIA driver is older than the image's CUDA build needs (cu126: driver 560 or newer); check `nvidia-smi` in the setup output, and change `TORCH_CUDA` in `deploy/compose/docker-compose.gpu.yml` or the AMI.
- A GPU session hangs at "SSH ready" until timeout: your public IP changed (the security group allows only the IP seen at launch) or a firewall blocks outbound port 22. Set `VMS_SSH_CIDR` to the right range.

## Teardown

Locally: `make down` (`make down-volumes` also deletes stack data).

Cloud sessions tear themselves down and verify it (above). How you know nothing is still billing on **Runpod** (GPU sessions):
- The session ends with "teardown verified": no pod named `vms-gpu-<id>` is listed and the pod id it recorded reads as gone. `make runpod-list` lists every pod named `vms-*` from any session and should print "no project pods exist"; the Runpod console's Pods page should be empty too. Sessions create no network volumes (they bill even with no pod), so the console's Storage page should stay empty.
- If a session exits 2, or this machine died mid-session: `make runpod-teardown SESSION=<id>`. Even without it, the pod deletes itself after 150 minutes, and the balance with auto-pay off caps the rest.

On **AWS** (CPU sessions, and EC2 GPU sessions if the quota arrives):
- The session ends with "teardown verified": it looked up every resource tagged `session=<id>` (Resource Groups Tagging API) plus the instance, security group and key pair it recorded, and none still exists. A terminated instance can stay listed by the Tagging API for a while; it counts as gone once EC2 reports it `terminated` (terminated instances are not billed).
- `make gpu-list` lists every resource tagged `project=vms` that still exists, from any session. It should print "no project resources exist" (the budget is not an EC2 resource and is not listed).
- If a session exits 2, or this machine died mid-session: `make gpu-teardown SESSION=<id>` (the id is printed at the start and names `.gpu-sessions/<id>/`). It finds the resources from the session's tags and saved state, deletes them and verifies. Even without it, the instance terminated itself after 150 minutes; the security group and key pair left behind cost nothing but should be removed.
- Billing data lags about a day: compare the session cost in the results file with Billing and Cost Management (filter by tag `project=vms` once the tag is activated as a cost allocation tag) the next day.

## Production checklist

_Placeholder (task 10.4)._ The project is never called "production-ready" unless every item here is met.
