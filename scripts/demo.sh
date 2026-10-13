#!/usr/bin/env bash
# make demo: the end-to-end CPU demo (task 1.9). Needs only Docker and uv.
#
#   1. the stack (make up: Postgres, object store, observability; migrations; camera catalog)
#   2. the mini clips (make data-mini) and the detector weights (yolo11n by default)
#   3. the application containers: ingest, one worker, the query service with the UI
#   4. loadgen: DEMO_CAMERAS virtual cameras at DEMO_FPS for DEMO_SECONDS, then waits until the
#      worker has processed every frame
#   5. the example questions, answered by the canned planner unless LLM_MODEL is set
#   6. the telemetry check (vms.observe): metrics scraped, one frame traced from loadgen to the
#      worker's DB write, one question traced through every query stage
#
# Everything stays up afterwards: open the UI, ask more questions, then `make down`.
# Re-running adds to the same database; `make down-volumes` starts from empty.
set -euo pipefail
cd "$(dirname "$0")/.."

env_file="${1:-.env}"
[[ -f "$env_file" ]] || env_file=.env.example
setting() {  # value of NAME in the env file, else the default
  local value=""
  value="$(sed -n -E "s/^$1=([^ #]*).*/\1/p" "$env_file" | tail -n 1)"
  echo "${value:-$2}"
}

seconds="${DEMO_SECONDS:-180}"
cameras="${DEMO_CAMERAS:-4}"
fps="${DEMO_FPS:-5}"
weights="${DEMO_WEIGHTS:-yolo11n}"
drain_timeout="${DEMO_DRAIN_TIMEOUT_S:-900}"
ingest_port="$(setting INGEST_PORT 8080)"
query_port="$(setting QUERY_PORT 8070)"
grafana_port="$(setting GRAFANA_PORT 3000)"
compose=(docker compose --env-file "$env_file" -f deploy/compose/docker-compose.yml)

step() { printf '\n==> %s\n' "$*"; }

step "1/6 stack: make up"
make --no-print-directory up ENV_FILE="$env_file"

step "2/6 clips and weights"
make --no-print-directory data-mini
uv run python -m vms.worker weights "$weights"

step "3/6 application containers: ingest, worker ($weights), query + UI"
# Build the one image first: ingest and worker reuse it and must not try to pull it.
"${compose[@]}" --profile app build query
DEMO_WEIGHTS="$weights" "${compose[@]}" --profile app up -d --wait --wait-timeout 600 \
  query ingest worker

started="$(date +%s)"  # the telemetry check looks only at traces from this run on
step "4/6 loadgen: $cameras cameras x $fps FPS for $seconds s, then wait for the worker"
results_dir="$(mktemp -d)"  # demo runs are not benchmarks: keep bench/results clean
# Trace 10% of frames (loadgen's default is 1%) so the demo leaves plenty of frame traces to
# open in Grafana; ingest and the worker follow loadgen's decision per frame.
VMS_RESULTS_DIR="$results_dir" VMS_TELEMETRY__SAMPLE_RATIO="${DEMO_TRACE_RATIO:-0.1}" \
  uv run python -m vms.loadgen run \
    --cameras "$cameras" --fps "$fps" --duration "$seconds" --warmup 0 \
    --target-url "http://127.0.0.1:${ingest_port}"
rm -rf "$results_dir"
uv run python -m vms.storage wait-drained --timeout-s "$drain_timeout"

step "5/6 questions"
uv run python -m vms.query ask --url "http://127.0.0.1:${query_port}" \
  --from-file configs/demo/canned_plans.v1.jsonl --check

step "6/6 telemetry: metrics in Prometheus, a frame trace and a query trace in Tempo"
uv run python -m vms.observe --since "$started"

cat <<URLS

Demo is up:
  UI (camera wall, questions)   http://localhost:${query_port}
  Grafana                       http://localhost:${grafana_port}
  Ask from the shell            uv run python -m vms.query ask "Show the last 10 zone events."
  Stop                          make down        (make down-volumes also deletes the data)
URLS
