#!/usr/bin/env bash
# Runs ON a Runpod pod; vms.cloud.pod_session calls it over SSH (scripts/runpod_session.sh, D37).
# A pod is a container: no Docker or Compose, so no stack and no services. Only in-process
# scenarios (`kind: model` and `ladder`) run here, in the project's environment with CUDA PyTorch.
#
#   setup            uv, the locked environment with torch and torchvision swapped for their
#                    CUDA builds (the versions uv.lock pins, as in deploy/docker/Dockerfile.gpu),
#                    the mini clips (copied from the operator's machine), the detector weights
#                    and their ONNX export
#   run SCENARIO     one model or ladder scenario through python -m vms.bench
#
# No secrets reach the pod: .env comes from .env.example, so no LLM key and no Runpod or AWS
# key of the operator's. Results land in bench/results/, which the session copies back.
set -euo pipefail
cd "$(dirname "$0")/.."

UV_VERSION=0.12.20  # as in CI and deploy/docker/Dockerfile
TORCH_CUDA=cu126    # as in deploy/docker/Dockerfile.gpu (drivers from 560, D28)
export PATH="$HOME/.local/bin:$PATH"
# After the CUDA swap, `uv run` must not re-sync the environment back to the CPU wheels.
export UV_NO_SYNC=1

step() { printf '\n==> %s\n' "$*"; }

setup() {
  step "GPU and tools"
  nvidia-smi
  if [[ "$(uv --version 2>/dev/null | cut -d' ' -f2)" != "$UV_VERSION" ]]; then
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
  fi
  uv --version
  [[ -f .env ]] || cp .env.example .env

  step "environment: uv.lock, then CUDA torch and torchvision at the locked versions"
  env -u UV_NO_SYNC uv sync --locked
  torch="$(uv run python -c 'from importlib.metadata import version; print(version("torch").split("+")[0])')"
  vision="$(uv run python -c 'from importlib.metadata import version; print(version("torchvision").split("+")[0])')"
  uv pip install --python .venv/bin/python \
    --index "https://download.pytorch.org/whl/${TORCH_CUDA}" \
    --default-index https://pypi.org/simple \
    "torch==${torch}+${TORCH_CUDA}" "torchvision==${vision}+${TORCH_CUDA}"
  uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; \
print('torch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.get_device_name(0))"

  step "clips and weights"
  make data-mini   # the synced clips are verified and kept, not downloaded again
  uv run python -m vms.worker weights
  make onnx
}

run() {
  local scenario="$1" kind
  kind="$(uv run python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["kind"])' \
    "$scenario")"
  step "scenario $scenario ($kind)"
  if [[ "$kind" != model && "$kind" != ladder ]]; then
    echo "a pod runs only in-process scenarios, kind model or ladder (no Docker in a pod, D37)" >&2
    exit 2
  fi
  uv run python -m vms.bench run --scenario "$scenario"
}

case "${1:-}" in
  setup) setup ;;
  run) run "${2:?scenario file}" ;;
  *) echo "usage: $0 setup | run SCENARIO" >&2; exit 2 ;;
esac
