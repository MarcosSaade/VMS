#!/usr/bin/env bash
# Runs ON the session instance; vms.cloud calls it over SSH (scripts/gpu_session.sh, task 2.4)
# with VMS_ACCELERATOR=gpu or cpu (the machine in configs/cloud.yaml, D35).
#
#   setup            uv, the locked environment, the stack, the mini clips (copied from the
#                    operator's machine), the detector weights, the worker image, then ingest,
#                    the worker and the query service (canned planner). gpu: the CUDA worker
#                    image and DCGM exporter; cpu: the same images and services as a laptop
#   run SCENARIO     one scenario through make bench on the host, as on a laptop; on gpu,
#                    model scenarios run inside the CUDA worker image, the only place with CUDA
#                    PyTorch
#
# No secrets reach the instance: .env comes from .env.example, so no LLM key and no AWS
# credentials. Results land in bench/results/, which the session copies back.
set -euo pipefail
cd "$(dirname "$0")/.."

UV_VERSION=0.12.20  # as in CI and deploy/docker/Dockerfile
export PATH="$HOME/.local/bin:$PATH"
ACCEL="${VMS_ACCELERATOR:-gpu}"
compose=(docker compose --env-file .env -f deploy/compose/docker-compose.yml)
services=(query ingest worker)
if [[ "$ACCEL" == gpu ]]; then
  compose+=(-f deploy/compose/docker-compose.gpu.yml)
  services+=(dcgm-exporter)
fi

step() { printf '\n==> %s\n' "$*"; }

setup() {
  step "$ACCEL host and tools"
  if [[ "$ACCEL" == gpu ]]; then nvidia-smi; else lscpu | head -20; fi
  command -v uv >/dev/null || curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
  if ! docker compose version >/dev/null 2>&1; then
    sudo apt-get update -q && sudo apt-get install -y -q docker-compose-plugin
  fi
  [[ -f .env ]] || cp .env.example .env

  step "environment and stack"
  make setup
  make up

  step "clips and weights"
  make data-mini   # the synced clips are verified and kept, not downloaded again
  uv run python -m vms.worker weights
  make onnx        # the CPU worker's backend (D5); the ladder's ONNX Runtime steps
  if [[ "$ACCEL" == cpu ]]; then
    make calibration-data           # ~19 MB, for the ladder's INT8 step
    make quantize WEIGHTS=yolo11n
  fi

  step "images: the CPU application image (gpu: then the CUDA layer on top)"
  "${compose[@]}" --profile app build query
  "${compose[@]}" --profile app build worker

  step "services: ${services[*]}"
  "${compose[@]}" --profile app up -d --wait --wait-timeout 600 "${services[@]}"
  if [[ "$ACCEL" == gpu ]]; then
    "${compose[@]}" exec -T worker python -c \
      "import torch; assert torch.cuda.is_available(), 'no CUDA'; \
print('torch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.get_device_name(0))"
  fi
}

run() {
  local scenario="$1" kind
  kind="$(uv run python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["kind"])' \
    "$scenario")"
  step "scenario $scenario ($kind)"
  if [[ "$kind" == finetune ]]; then
    make finetune-data eval-detection-data   # training list and eval subset, verified
  fi
  case "$ACCEL:$kind" in
    gpu:model)
      mkdir -p bench/results && chmod a+rwx bench/results  # the image runs as uid 10001
      "${compose[@]}" --profile app run --rm --no-deps -T \
        -v "$PWD/bench:/app/bench" -v "$PWD/data:/app/data:ro" \
        worker python -m vms.bench run --scenario "$scenario"
      ;;
    *)
      make bench SCENARIO="$scenario"
      ;;
  esac
}

case "${1:-}" in
  setup) setup ;;
  run) run "${2:?scenario file}" ;;
  *) echo "usage: $0 setup | run SCENARIO" >&2; exit 2 ;;
esac
