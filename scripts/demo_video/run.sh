#!/usr/bin/env bash
# Records the demo video (task 11.3) from the running stack.
#   1. needs the stack and app containers up (make demo), Node 22, ffmpeg and Chrome (CHROME_PATH)
#   2. streams 4 cameras with loadgen in the background so the camera wall and dashboards are live
#   3. runs the Playwright recorder (record.js), then stops loadgen
# Usage: scripts/demo_video/run.sh [OUT.mp4]   (npm ci in this folder first)
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
repo="$(cd "$here/../.." && pwd)"
out="$(realpath -m "${1:-vms-demo.mp4}")"

ingest_port="$(sed -n -E 's/^INGEST_PORT=([0-9]+).*/\1/p' "$repo/.env" | tail -n 1)"
query_port="$(sed -n -E 's/^QUERY_PORT=([0-9]+).*/\1/p' "$repo/.env" | tail -n 1)"
grafana_port="$(sed -n -E 's/^GRAFANA_PORT=([0-9]+).*/\1/p' "$repo/.env" | tail -n 1)"

# Start from an empty queue: a backlog left by an earlier run makes the dashboards red.
(cd "$repo" && uv run python -m vms.storage wait-drained --timeout-s 900)

results="$(mktemp -d)"
(cd "$repo" && VMS_RESULTS_DIR="$results" uv run python -m vms.loadgen run \
  --cameras 4 --fps 5 --duration 600 --warmup 0 \
  --target-url "http://127.0.0.1:${ingest_port:-8080}" >"$results/loadgen.log" 2>&1) &
# Stop uv and the Python process under it (killing the background subshell alone leaves them).
trap 'pkill -f "vms.loadgen run --cameras 4 --fps 5 --duration 600" || true; rm -rf "$results"' EXIT

echo "loadgen streaming; filling the last minutes of data before recording..."
sleep 300  # the 5-minute Grafana window then shows only this steady run
node "$here/record.js" "$out" "http://localhost:${query_port:-8070}" "http://localhost:${grafana_port:-3000}"
