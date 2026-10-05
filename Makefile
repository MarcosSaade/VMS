SHELL := /bin/bash
.DEFAULT_GOAL := help

UV ?= uv
RUN := $(UV) run
COMPOSE_FILE ?= deploy/compose/docker-compose.yml
COMPOSE := docker compose -f $(COMPOSE_FILE)

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
up: ## Start the compose CPU stack
	@test -f $(COMPOSE_FILE) || { echo "make up: $(COMPOSE_FILE) not found yet (task 0.5)." >&2; exit 1; }
	$(COMPOSE) up -d

.PHONY: down
down: ## Stop the compose stack
	@test -f $(COMPOSE_FILE) || { echo "make down: $(COMPOSE_FILE) not found yet (task 0.5)." >&2; exit 1; }
	$(COMPOSE) down

.PHONY: logs
logs: ## Follow compose logs
	@test -f $(COMPOSE_FILE) || { echo "make logs: $(COMPOSE_FILE) not found yet (task 0.5)." >&2; exit 1; }
	$(COMPOSE) logs -f --tail=200

.PHONY: demo
demo: ## End-to-end CPU demo on the mini dataset
	$(call not_yet,1.9)

# --- Data ----------------------------------------------------------------------

.PHONY: data
data: ## Download public datasets and weights (checksummed)
	$(call not_yet,1.1)

.PHONY: data-mini
data-mini: ## Download the small demo dataset
	$(call not_yet,1.1)

.PHONY: data-pull
data-pull: ## dvc pull from the default remote (no-op if missing or empty)
	@scripts/data_pull.sh

# --- Benchmarks and evaluation -------------------------------------------------

.PHONY: bench
bench: ## Run a benchmark scenario
	$(call not_yet,2.3)

.PHONY: eval
eval: ## Run eval suites
	$(call not_yet,3.5)
