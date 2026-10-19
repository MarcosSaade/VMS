SHELL := /bin/bash
.DEFAULT_GOAL := help

UV ?= uv
RUN := $(UV) run
COMPOSE_FILE ?= deploy/compose/docker-compose.yml
# Compose reads variables from the repo-root .env (falls back to .env.example on a fresh clone).
ENV_FILE ?= $(if $(wildcard .env),.env,.env.example)
# Compose profiles to activate: cpu (default) or gpu, for services that exist in both variants.
PROFILES ?= cpu
STACK_WAIT_TIMEOUT ?= 300
COMPOSE := COMPOSE_PROFILES=$(PROFILES) docker compose --env-file $(ENV_FILE) -f $(COMPOSE_FILE)

# Prints a clear error for targets whose implementation lands in a later task.
define not_yet
	@echo "make $@: not implemented yet (see task $(1) in docs/PLAN.md)." >&2; exit 1
endef

.PHONY: help
help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-18s %s\n", $$1, $$2}'

# --- Development ---------------------------------------------------------------

.PHONY: setup
setup: ## Install Python 3.12 env from uv.lock, git hooks and .env
	$(UV) sync --locked
	@if [ -d .git ]; then $(RUN) pre-commit install; fi
	@if [ ! -f .env ]; then cp .env.example .env && echo "Created .env from .env.example"; fi

.PHONY: fmt
fmt: ## Format and autofix with ruff
	$(RUN) ruff format .
	$(RUN) ruff check --fix .

.PHONY: lint
lint: ## Lint and check formatting
	$(RUN) ruff check .
	$(RUN) ruff format --check .

.PHONY: typecheck
typecheck: ## mypy --strict on src/vms
	$(RUN) mypy

.PHONY: test
test: ## Unit tests (integration, gpu and slow are skipped)
	$(RUN) pytest

.PHONY: test-integration
test-integration: ## Integration tests against the compose CPU stack
	@# pytest exits 5 when nothing is collected; that is fine until integration tests exist.
	@$(RUN) pytest -m integration; rc=$$?; \
	if [ $$rc -eq 5 ]; then echo "No integration tests collected yet."; exit 0; fi; exit $$rc

.PHONY: check
check: lint typecheck test ## Lint + typecheck + unit tests

# --- Local stack ---------------------------------------------------------------

.PHONY: up
up: ## Start the compose stack, wait until healthy, migrate and bootstrap the DB (PROFILES=cpu|gpu)
	$(COMPOSE) up -d --build --remove-orphans --wait --wait-timeout $(STACK_WAIT_TIMEOUT)
	@$(MAKE) --no-print-directory migrate
	@scripts/stack_urls.sh $(ENV_FILE)

.PHONY: app
app: ## Build and start the containerized query service + UI on http://localhost:8070 (after make up)
	$(COMPOSE) --profile app build query
	$(COMPOSE) --profile app up -d --wait --wait-timeout $(STACK_WAIT_TIMEOUT) query
	@echo "UI and query API: http://localhost:$${QUERY_PORT:-8070}"

.PHONY: migrate
migrate: ## Apply migrations, then the read-only login and camera catalog (POSTGRES_* from .env)
	$(RUN) alembic upgrade head
	$(RUN) python -m vms.storage bootstrap

.PHONY: down
down: ## Stop the compose stack and the app containers (keeps data volumes)
	$(COMPOSE) --profile app down

.PHONY: down-volumes
down-volumes: ## Stop the compose stack and the app containers, delete data volumes
	$(COMPOSE) --profile app down --volumes

.PHONY: ps
ps: ## Show compose services and health
	$(COMPOSE) --profile app ps

.PHONY: logs
logs: ## Follow compose logs (SERVICE=<name> for one service)
	$(COMPOSE) logs -f --tail=200 $(SERVICE)

# make demo DEMO_SECONDS=30 for a short run (CI). DEMO_CAMERAS, DEMO_FPS, DEMO_WEIGHTS too.
DEMO_SECONDS ?= 180

.PHONY: demo
demo: ## End-to-end CPU demo: stack, clips, ingest + worker + query/UI, loadgen, example questions
	DEMO_SECONDS=$(DEMO_SECONDS) scripts/demo.sh $(ENV_FILE)

# --- Data ----------------------------------------------------------------------

# Clips whose license must be accepted first are skipped unless named: make data ACCEPT=virat
ACCEPT ?=
DATA_ACCEPT := $(foreach id,$(ACCEPT),--accept-license $(id))

.PHONY: data
data: ## Fetch, verify and normalize all video clips into data/ (ACCEPT=virat for opt-in clips)
	$(RUN) python -m vms.datasets fetch --subset full $(DATA_ACCEPT)

.PHONY: data-mini
data-mini: ## Fetch the small clip subset for CI and laptops (~3 clips)
	$(RUN) python -m vms.datasets fetch --subset mini

.PHONY: data-pull
data-pull: ## dvc pull from the default remote (no-op if missing or empty)
	@scripts/data_pull.sh

# --- Services -------------------------------------------------------------------

.PHONY: ingest
ingest: ## Run the ingest service on the host (needs make up)
	$(RUN) python -m vms.ingest

.PHONY: worker
worker: ## Run the inference worker on the host (needs make up; fetches weights on first run)
	$(RUN) python -m vms.worker run

.PHONY: drift
drift: ## Run the drift monitor on the host (needs make up; per-camera drift gauges on :8095)
	$(RUN) python -m vms.drift run

.PHONY: drift-score
drift-score: ## Score drift detectors (threshold vs CUSUM) on stored drift runs (SCENARIO=bench/scenarios/drift-detect-cpu.yaml)
	$(RUN) python -m vms.drift score --scenario $(or $(SCENARIO),bench/scenarios/drift-detect-cpu.yaml)

.PHONY: query
query: ## Run the query service on the host (needs make up; LLM from LLM_* in .env, else canned)
	$(RUN) python -m vms.query serve

.PHONY: ui-build
ui-build: ## Build the UI into frontend/dist (served by make query); needs Node 22
	cd frontend && npm ci --no-audit --no-fund && npm run build

.PHONY: ui-dev
ui-dev: ## UI dev server on http://localhost:5173, proxying /v1 to make query
	cd frontend && npm run dev

.PHONY: ui-check
ui-check: ## Frontend typecheck and unit tests (vitest); needs Node 22
	cd frontend && npm ci --no-audit --no-fund && npm run typecheck && npm test

.PHONY: zones
zones: ui-build ## Zone editor on http://127.0.0.1:8075: draw zones on clip frames, saves configs/cameras.yaml
	$(RUN) python -m vms.zones edit

.PHONY: models
models: ## Fetch and verify the pinned detector weights into models/weights/
	$(RUN) python -m vms.worker weights

.PHONY: onnx
onnx: ## Export the pinned weights to ONNX into models/onnx/ (needed by detector.backend=onnxruntime)
	$(RUN) python -m vms.worker export-onnx

.PHONY: calibration-data
calibration-data: ## Fetch and verify the INT8 calibration images (64 COCO train2017 + 64 MOT17 frames, ~19 MB)
	$(RUN) python -m vms.worker calibration fetch

.PHONY: quantize
quantize: ## Static INT8 quantization of the ONNX exports (needs make onnx and make calibration-data; WEIGHTS=yolo11s)
	$(RUN) python -m vms.worker quantize $(WEIGHTS)

.PHONY: trt-prepare
trt-prepare: ## On a GPU: FP16 export, TensorRT INT8 model and TensorRT engines (WEIGHTS=yolo11n; D38)
	$(RUN) python -m vms.worker trt-prepare $(or $(WEIGHTS),yolo11n)

.PHONY: finetune-data
finetune-data: ## Fetch and verify the 9.5 training images (800 COCO train2017 + ~790 MOT17 frames)
	$(RUN) python -m vms.finetune fetch

.PHONY: finetune
finetune: ## Fine-tune YOLO11n against blur and score it before/after (SCENARIO=bench/scenarios/finetune-blur-cpu.yaml; needs finetune-data, eval-detection-data)
	$(RUN) python -m vms.finetune run --scenario $(or $(SCENARIO),bench/scenarios/finetune-blur-cpu.yaml)

.PHONY: onnx-parity
onnx-parity: ## ONNX Runtime vs PyTorch parity on the eval subset and data-mini (WEIGHTS=yolo11n|yolo11s)
	$(RUN) python -m vms.worker onnx-parity --weights $(or $(WEIGHTS),yolo11n)

# --- Load generation ------------------------------------------------------------

SCENARIO ?=
STUB_WORKERS ?= 1

.PHONY: loadgen
loadgen: ## Virtual cameras against ingest (SCENARIO=bench/scenarios/<name>.yaml)
	$(RUN) python -m vms.loadgen run $(if $(SCENARIO),--scenario $(SCENARIO))

.PHONY: loadgen-selftest
loadgen-selftest: ## Measure the load generator's own max rate (no server)
	$(RUN) python -m vms.loadgen selftest --scenario bench/scenarios/loadgen-selftest.yaml

.PHONY: loadgen-stub
loadgen-stub: ## Stub ingest server for loadgen tests (STUB_WORKERS=n processes)
	$(RUN) python -m vms.loadgen stub --workers $(STUB_WORKERS)

# --- Cloud sessions on AWS (task 2.4, D35; RUNBOOK: GPU sessions) -----------------

GPU_SCENARIOS ?= bench/scenarios/gpu-smoke.yaml bench/scenarios/worker-model-gpu-n.yaml
SESSION ?=

.PHONY: gpu-preflight
gpu-preflight: ## Free checks before a GPU session: credentials, Region, quota, AMI, dry-run launch
	scripts/gpu_session.sh preflight

.PHONY: gpu-session
gpu-session: ## Launch, run GPU_SCENARIOS, collect results, tear down and verify (asks first; costs money)
	scripts/gpu_session.sh run $(GPU_SCENARIOS)

# CPU sessions (D35): the same driver on a CPU instance (configs/cloud.yaml machines.cpu);
# INSTANCE=c7i.4xlarge picks another priced type.
CPU_SCENARIOS ?= bench/scenarios/ramp-cpu-ort-b4-nospin.yaml bench/scenarios/query-load-cpu.yaml \
	bench/scenarios/ladder-cpu-n.yaml bench/scenarios/sweep-cpu-ort-n.yaml
INSTANCE ?=
CPU_ENV = VMS_MACHINE=cpu $(if $(INSTANCE),VMS_INSTANCE_TYPE=$(INSTANCE))

.PHONY: cpu-preflight
cpu-preflight: ## Free checks before a CPU session (INSTANCE=c7i.2xlarge by default)
	$(CPU_ENV) scripts/gpu_session.sh preflight

.PHONY: cpu-session
cpu-session: ## Launch a CPU instance, run CPU_SCENARIOS, collect, tear down and verify (asks first; costs money)
	$(CPU_ENV) scripts/gpu_session.sh run $(CPU_SCENARIOS)

.PHONY: gpu-teardown
gpu-teardown: ## Delete what a GPU session left behind and verify (SESSION=<id>)
	@test -n "$(SESSION)" || { echo "make gpu-teardown: set SESSION=<id>" >&2; exit 1; }
	scripts/gpu_session.sh teardown --session $(SESSION)

.PHONY: gpu-list
gpu-list: ## AWS resources tagged project=vms that still exist
	scripts/gpu_session.sh list

# --- GPU sessions on Runpod (D37; RUNBOOK: GPU sessions) ---------------------------
# AWS denied the GPU quota (D35), so GPU work runs on Runpod pods: model scenarios only (a pod
# has no Docker). Needs RUNPOD_API_KEY in .env. VMS_GPU_TYPE picks another catalog GPU.

RUNPOD_SCENARIOS ?= bench/scenarios/ladder-gpu-n.yaml bench/scenarios/sweep-gpu-trt-fp16-n.yaml \
	bench/scenarios/sweep-gpu-trt-int8-n.yaml

.PHONY: runpod-preflight
runpod-preflight: ## Free checks before a Runpod GPU session: API key, GPU price and stock, old pods
	scripts/runpod_session.sh preflight

.PHONY: runpod-session
runpod-session: ## Create a pod, run RUNPOD_SCENARIOS, collect, delete and verify (asks first; costs money)
	scripts/runpod_session.sh run $(RUNPOD_SCENARIOS)

.PHONY: runpod-teardown
runpod-teardown: ## Delete a Runpod session's pod and verify (SESSION=<id>)
	@test -n "$(SESSION)" || { echo "make runpod-teardown: set SESSION=<id>" >&2; exit 1; }
	scripts/runpod_session.sh teardown --session $(SESSION)

.PHONY: runpod-list
runpod-list: ## Runpod pods named vms-* that still exist
	scripts/runpod_session.sh list

# --- Benchmarks and evaluation -------------------------------------------------

.PHONY: dashboards
dashboards: ## Regenerate the Grafana dashboards JSON from vms.observe.dashboards
	$(RUN) python -m vms.observe.dashboards

.PHONY: bench-model
bench-model: ## Model-only worker latency (SCENARIO=bench/scenarios/worker-model-cpu.yaml)
	$(RUN) python -m vms.worker bench --scenario $(or $(SCENARIO),bench/scenarios/worker-model-cpu.yaml)

.PHONY: bench
bench: ## Run a benchmark scenario of any kind (SCENARIO=bench/scenarios/<name>.yaml; e2e needs make demo's services)
	$(RUN) python -m vms.bench run --scenario $(or $(SCENARIO),bench/scenarios/bench-smoke.yaml)

RESULTS ?=
.PHONY: bench-report
bench-report: ## Markdown tables of results files for BENCHMARKS.md (RESULTS="bench/results/a.json ...")
	@test -n "$(RESULTS)" || { echo "make bench-report: set RESULTS=<results JSON files>" >&2; exit 1; }
	@$(RUN) python -m vms.bench report $(RESULTS)

# Structured-query eval set (task 3.1, RUNBOOK: Evaluation). The snapshot is frozen: regenerate
# it only as a new version, since that changes every gold answer.
.PHONY: eval-snapshot
eval-snapshot: ## Regenerate the synthetic eval snapshot and its manifest (new versions only)
	$(RUN) python -m vms.eval snapshot

.PHONY: eval-db
eval-db: ## (Re)create the vms_eval database from the snapshot (pulled or regenerated; hash-checked)
	$(RUN) python -m vms.eval load

.PHONY: eval-build
eval-build: ## Run every gold SQL and plan on vms_eval; write the dev/test files and manifest
	$(RUN) python -m vms.eval build

.PHONY: eval-check
eval-check: ## Offline checks of the eval set: few-shot leakage, splits, hashes
	$(RUN) python -m vms.eval check

REVIEW ?= test
.PHONY: eval-review
eval-review: ## Print an eval split for human review (REVIEW=test|dev)
	$(RUN) python -m vms.eval review --split $(REVIEW)

# make eval SUITE=structured SPLIT=dev (needs make eval-db). PLANNER=gold is the harness
# self-test (exits non-zero unless it scores 100%; CI runs it). The LLM comes from LLM_* in
# .env and may cost money. LIMIT=N asks only the first N items (smoke runs; not reportable).
# make eval SUITE=detection [WEIGHTS=yolo11n] (needs make models and make eval-detection-data).
SUITE ?= structured
SPLIT ?= dev
PLANNER ?= config
LIMIT ?=
.PHONY: eval
eval: ## Run an eval suite, write eval/results/*.json (SUITE=structured SPLIT=dev PLANNER=config|gold, or SUITE=detection)
	@if [ "$(SUITE)" = "detection" ]; then $(MAKE) --no-print-directory eval-detection; \
	else $(RUN) python -m vms.eval run --suite $(SUITE) --split $(SPLIT) --planner $(PLANNER) $(if $(LIMIT),--limit $(LIMIT)); fi

.PHONY: eval-detection-data
eval-detection-data: ## Fetch and verify the detection eval images (500 COCO val2017 + 73 MOT17 frames, ~95 MB)
	$(RUN) python -m vms.eval.detection fetch

.PHONY: eval-detection
eval-detection: ## Detection mAP@50 and mAP@50-95 with bootstrap CIs (WEIGHTS=yolo11s|yolo11n BACKEND=ultralytics|onnxruntime PRECISION=fp32|int8 NAME=...)
	$(if $(WEIGHTS),VMS_DETECTOR__WEIGHTS=$(WEIGHTS)) $(if $(BACKEND),VMS_DETECTOR__BACKEND=$(BACKEND)) \
	$(if $(PRECISION),VMS_DETECTOR__PRECISION=$(PRECISION)) \
	$(RUN) python -m vms.eval.detection run $(if $(NAME),--name $(NAME))

.PHONY: eval-detection-compare
eval-detection-compare: ## Paired mAP difference between two detection runs, with the D32 verdict (BASE=a.json OTHER=b.json)
	@test -n "$(BASE)" -a -n "$(OTHER)" || { echo "set BASE=<results JSON> OTHER=<results JSON>" >&2; exit 1; }
	$(RUN) python -m vms.eval.detection compare $(BASE) $(OTHER) $(if $(NAME),--name $(NAME))

.PHONY: eval-drift
eval-drift: ## Detection mAP clean and under each photometric drift, paired deltas (SCENARIO=bench/scenarios/drift-map-cpu.yaml)
	$(RUN) python -m vms.eval.detection drift --scenario $(or $(SCENARIO),bench/scenarios/drift-map-cpu.yaml)

.PHONY: eval-report
eval-report: ## Markdown tables of eval results files (RESULTS="eval/results/a.json ...")
	@test -n "$(RESULTS)" || { echo "make eval-report: set RESULTS=<results JSON files>" >&2; exit 1; }
	@$(RUN) python -m vms.eval report $(RESULTS)
