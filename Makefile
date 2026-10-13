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

# --- Benchmarks and evaluation -------------------------------------------------

.PHONY: dashboards
dashboards: ## Regenerate the Grafana dashboards JSON from vms.observe.dashboards
	$(RUN) python -m vms.observe.dashboards

.PHONY: bench-model
bench-model: ## Model-only worker latency (SCENARIO=bench/scenarios/worker-model-cpu.yaml)
	$(RUN) python -m vms.worker bench --scenario $(or $(SCENARIO),bench/scenarios/worker-model-cpu.yaml)

.PHONY: bench
bench: ## Run a benchmark scenario
	$(call not_yet,2.3)

.PHONY: eval
eval: ## Run eval suites
	$(call not_yet,3.5)
