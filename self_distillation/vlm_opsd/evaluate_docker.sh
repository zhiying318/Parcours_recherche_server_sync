#!/usr/bin/env bash

# 启动方式：
# bash self_distillation/vlm_opsd/evaluate_docker.sh 4 \
#   --adapter self_distillation/output/qwen3_5_9b_opsd_continue/checkpoint-50 \
#   --output-jsonl self_distillation/output/test_checkpoint-50.jsonl
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 GPU_ID [--adapter ADAPTER] [evaluation arguments...]" >&2
  echo "Example: $0 4 --adapter self_distillation/output/qwen3_5_9b_opsd_continue/checkpoint-50" >&2
  exit 2
fi

GPU_IDS="$1"
shift
IMAGE="${COMFORT_ADD_PROMPT_IMAGE:-${ORIANY_IMAGE:-docker.v2.aispeech.com/sjtu/sjtu_chenlu-zzy-cuda_12.4-ubuntu_22.04-torch_2.6-orient-anything:v0.1}}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST_HF_CACHE="${OPSD_VLM_HF_CACHE:-${HOME}/.cache/huggingface}"
HOST_PYTHON_PACKAGES="${OPSD_VLM_PYTHON_PACKAGES:-${HOME}/.cache/opsd-vlm/python-torch28}"
HOST_CUDA_HOME="${OPSD_CUDA_HOME:-${CUDA_HOME:-/usr/local/cuda}}"
HF_ENDPOINT_VALUE="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ ! -x "${HOST_CUDA_HOME}/bin/nvcc" ]]; then
  echo "CUDA Toolkit not found at ${HOST_CUDA_HOME}/bin/nvcc" >&2
  echo "Set OPSD_CUDA_HOME to a host CUDA Toolkit directory." >&2
  exit 2
fi

cd "$REPO_ROOT"
mkdir -p "$HOST_HF_CACHE" "$HOST_PYTHON_PACKAGES"
docker image inspect "$IMAGE" >/dev/null

exec docker run --rm --init \
  --name "opsd-vlm-eval-${USER:-user}-$(date +%Y%m%d-%H%M%S)" \
  --network host \
  --gpus "\"device=${GPU_IDS}\"" \
  --shm-size "${OPSD_VLM_SHM_SIZE:-64g}" \
  --user "$(id -u):$(id -g)" \
  --env HTTP_PROXY \
  --env HTTPS_PROXY \
  --env http_proxy \
  --env https_proxy \
  --env ALL_PROXY \
  --env all_proxy \
  --env NO_PROXY \
  --env no_proxy \
  --env CUDA_HOME=/usr/local/cuda \
  --env CUDA_PATH=/usr/local/cuda \
  --env HOME=/tmp/opsd-vlm-home \
  --env HF_HOME=/cache/huggingface \
  --env "HF_ENDPOINT=${HF_ENDPOINT_VALUE}" \
  --env HF_HUB_DISABLE_TELEMETRY=1 \
  --env HF_HUB_DISABLE_XET=1 \
  --env HF_TOKEN \
  --env PYTHONUSERBASE=/cache/python \
  --env PYTHONPATH=/cache/python/lib/python3.11/site-packages \
  --env PATH=/cache/python/bin:/opt/conda/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
  --volume "${REPO_ROOT}:/workspace/project" \
  --volume "${HOST_HF_CACHE}:/cache/huggingface" \
  --volume "${HOST_PYTHON_PACKAGES}:/cache/python:ro" \
  --volume "${HOST_CUDA_HOME}:/usr/local/cuda:ro" \
  --workdir /workspace/project \
  "$IMAGE" bash -lc '
    set -euo pipefail
    python -c "import peft, PIL, torch, torchvision, transformers; print(\"torch=\", torch.__version__)"
    python -m self_distillation.vlm_opsd.evaluate "$@"
  ' bash "$@"
