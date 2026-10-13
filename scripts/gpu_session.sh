#!/usr/bin/env bash
# GPU sessions on AWS (task 2.4): preflight | run SCENARIO... | teardown --session ID | list |
# budget --email ADDRESS. The logic lives in vms.cloud (python -m vms.cloud --help); teardown
# runs in a finally block there, so it happens on failure and Ctrl-C too. RUNBOOK: GPU sessions.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run python -m vms.cloud "$@"
