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

.PHONY: migrate
migrate: ## Apply migrations, then the read-only login and camera catalog (POSTGRES_* from .env)
	$(RUN) alembic upgrade head
	$(RUN) python -m vms.storage bootstrap

.PHONY: down
down: ## Stop the compose stack (keeps data volumes)
	$(COMPOSE) down

.PHONY: down-volumes
down-volumes: ## Stop the compose stack and delete its data volumes
	$(COMPOSE) down --volumes

.PHONY: ps
ps: ## Show compose services and health
	$(COMPOSE) ps

.PHONY: logs
logs: ## Follow compose logs (SERVICE=<name> for one service)
	$(COMPOSE) logs -f --tail=200 $(SERVICE)

.PHONY: demo
demo: ## End-to-end CPU demo on the mini dataset
	$(call not_yet,1.9)

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

.PHONY: bench
bench: ## Run a benchmark scenario
	$(call not_yet,2.3)

.PHONY: eval
eval: ## Run eval suites
	$(call not_yet,3.5)
