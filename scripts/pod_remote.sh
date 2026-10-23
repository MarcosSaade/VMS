#!/usr/bin/env bash
# Runs ON a Runpod pod; vms.cloud.pod_session calls it over SSH (scripts/runpod_session.sh, D37).
# A pod is a container: no Docker or Compose, so no stack and no services. Only in-process
# scenarios run here: `kind: model` and `ladder` in the project's environment with CUDA PyTorch,
# and `kind: llm`, whose engines (vLLM, Ollama) run as child processes of the scenario (D40).
#
#   setup vision     uv, the locked environment with torch and torchvision swapped for their
#                    CUDA 13 builds (the versions uv.lock pins), onnxruntime swapped for
#                    onnxruntime-gpu and TensorRT added (pinned below; D38), the mini clips
#                    (copied from the operator's machine), the calibration and detection eval
#                    images, the detector weights, their ONNX export, and the TensorRT engines
#   setup llm        uv and the locked environment (the load client and the planner, CPU only),
#                    vLLM in its own virtualenv (it pins its own CUDA torch) and the Ollama
#                    binary, both pinned below; models are downloaded by the scenario
#   run SCENARIO     one model, ladder or llm scenario through python -m vms.bench
#
# No secrets reach the pod: .env comes from .env.example, so no LLM key and no Runpod or AWS
# key of the operator's. Results land in bench/results/, which the session copies back.
set -euo pipefail
cd "$(dirname "$0")/.."

UV_VERSION=0.12.20  # as in CI and deploy/docker/Dockerfile
# One CUDA major for everything on the pod (D38): onnxruntime-gpu 1.31 is built for CUDA 13, so
# torch and TensorRT use their CUDA 13 builds too. Needs a host driver for CUDA >= 13.0
# (configs/runpod.yaml min_cuda_version). Versions are pinned here, outside uv.lock, because
# the laptop's lock is CPU-only; each results file records the installed versions.
TORCH_CUDA=cu130
ORT_GPU_VERSION=1.31.0        # = onnxruntime in uv.lock
TENSORRT_VERSION=11.3.0.99    # tensorrt-cu13, released 2026-09-09
# LLM serving (D40): the latest releases older than two weeks on 2026-10-23.
VLLM_VERSION=0.31.0           # PyPI, 2026-10-05; brings torch 2.13 for the host's CUDA
OLLAMA_VERSION=0.40.2         # GitHub release, 2026-10-08
VLLM_ENV=/root/vllm-env       # = configs/serving.yaml vllm.bin's virtualenv
export PATH="$HOME/.local/bin:$PATH"
# After the CUDA swap, `uv run` must not re-sync the environment back to the CPU wheels.
export UV_NO_SYNC=1

step() { printf '\n==> %s\n' "$*"; }

tools() {
  step "GPU and tools"
  nvidia-smi
  if [[ "$(uv --version 2>/dev/null | cut -d' ' -f2)" != "$UV_VERSION" ]]; then
    curl -LsSf "https://astral.sh/uv/${UV_VERSION}/install.sh" | sh
  fi
  uv --version
  [[ -f .env ]] || cp .env.example .env
}

setup_llm() {
  tools
  step "environment: uv.lock (load client and planner; no GPU packages)"
  env -u UV_NO_SYNC uv sync --locked

  step "vLLM ${VLLM_VERSION} in ${VLLM_ENV}"
  uv venv --python 3.12 "$VLLM_ENV"
  uv pip install --python "$VLLM_ENV/bin/python" --torch-backend=auto "vllm==${VLLM_VERSION}"
  "$VLLM_ENV/bin/python" -c "import torch, vllm; assert torch.cuda.is_available(), 'no CUDA'; \
print('vllm', vllm.__version__, 'torch', torch.__version__, 'CUDA', torch.version.cuda)"

  step "Ollama ${OLLAMA_VERSION}"
  command -v zstd >/dev/null || { apt-get -q update && apt-get -q install -y zstd; }
  curl -fsSL "https://github.com/ollama/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst" \
    | zstd -d | tar -x -C /usr/local
  /usr/local/bin/ollama --version || true  # prints a warning when no server runs yet
}

setup_vision() {
  tools

  step "environment: uv.lock, then CUDA torch and torchvision at the locked versions"
  env -u UV_NO_SYNC uv sync --locked
  torch="$(uv run python -c 'from importlib.metadata import version; print(version("torch").split("+")[0])')"
  vision="$(uv run python -c 'from importlib.metadata import version; print(version("torchvision").split("+")[0])')"
  uv pip install --python .venv/bin/python \
    --index "https://download.pytorch.org/whl/${TORCH_CUDA}" \
    --default-index https://pypi.org/simple \
    "torch==${torch}+${TORCH_CUDA}" "torchvision==${vision}+${TORCH_CUDA}"
  uv pip uninstall --python .venv/bin/python onnxruntime
  uv pip install --python .venv/bin/python \
    "onnxruntime-gpu[cuda,cudnn]==${ORT_GPU_VERSION}" "tensorrt-cu13==${TENSORRT_VERSION}"
  uv run python -c "import torch; assert torch.cuda.is_available(), 'no CUDA'; \
print('torch', torch.__version__, 'CUDA', torch.version.cuda, torch.cuda.get_device_name(0))"
  uv run python -c "import onnxruntime as o; o.preload_dlls(); \
assert 'CUDAExecutionProvider' in o.get_available_providers(), o.get_available_providers(); \
print('onnxruntime', o.__version__, o.get_available_providers())"
  uv run python -c "import tensorrt; print('tensorrt', tensorrt.__version__)"

  step "data: clips, calibration and eval images"
  make data-mini   # the synced clips are verified and kept, not downloaded again
  make calibration-data eval-detection-data

  step "weights, ONNX export, TensorRT engines"
  uv run python -m vms.worker weights
  make onnx
  # One engine at a time: a precision that fails to build is logged and its ladder step fails
  # on its own, so the other steps still run (D38).
  make trt-prepare WEIGHTS=yolo11n TRT_PRECISION=fp16 || echo "!! TensorRT FP16 engine failed" >&2
  make trt-prepare WEIGHTS=yolo11n TRT_PRECISION=int8 || echo "!! TensorRT INT8 engine failed" >&2
}

run() {
  local scenario="$1" kind
  kind="$(uv run python -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))["kind"])' \
    "$scenario")"
  step "scenario $scenario ($kind)"
  if [[ "$kind" != model && "$kind" != ladder && "$kind" != llm ]]; then
    echo "a pod runs only in-process scenarios, kind model, ladder or llm (no Docker, D37)" >&2
    exit 2
  fi
  uv run python -m vms.bench run --scenario "$scenario"
}

case "${1:-}" in
  setup)
    case "${2:-vision}" in
      vision) setup_vision ;;
      llm) setup_llm ;;
      *) echo "usage: $0 setup vision|llm" >&2; exit 2 ;;
    esac ;;
  run) run "${2:?scenario file}" ;;
  *) echo "usage: $0 setup vision|llm | run SCENARIO" >&2; exit 2 ;;
esac
