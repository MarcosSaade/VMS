#!/usr/bin/env bash
# GPU sessions on Runpod pods (D37): preflight | run SCENARIO... | teardown --session ID | list.
# The logic lives in vms.cloud.pod_cli (python -m vms.cloud.pod_cli --help); teardown runs in a
# finally block there, so it happens on failure and Ctrl-C too. RUNBOOK: GPU sessions.
set -euo pipefail
cd "$(dirname "$0")/.."
exec uv run python -m vms.cloud.pod_cli "$@"
