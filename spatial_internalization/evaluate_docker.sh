#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 GPU_ID [evaluation arguments...]" >&2
  echo "Example: $0 0 --variant geometry_reasoning --adapter spatial_internalization/checkpoints/geometry_reasoning/final --output-jsonl spatial_internalization/evaluation_results/geometry_reasoning_test.jsonl" >&2
  exit 2
fi

GPU_ID="$1"
shift
IMAGE="${SPATIAL_INTERNALIZATION_IMAGE:-${COMFORT_ADD_PROMPT_IMAGE:-docker.v2.aispeech.com/sjtu/sjtu_chenlu-zzy-cuda_12.4-ubuntu_22.04-torch_2.6-orient-anything:v0.1}}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST_HF_CACHE="${SPATIAL_HF_CACHE:-${HOME}/.cache/huggingface}"
HOST_PYTHON_PACKAGES="${SPATIAL_PYTHON_PACKAGES:-${HOME}/.cache/spatial-internalization/python}"
HOST_CUDA_HOME="${SPATIAL_CUDA_HOME:-${CUDA_HOME:-/usr/local/cuda}}"
HF_ENDPOINT_VALUE="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p "$HOST_HF_CACHE" "$HOST_PYTHON_PACKAGES"
docker image inspect "$IMAGE" >/dev/null

exec docker run --rm --init \
  --name "spatial-internalization-eval-${USER:-user}-$(date +%Y%m%d-%H%M%S)" \
  --network host \
  --gpus "device=${GPU_ID}" \
  --shm-size "${SPATIAL_SHM_SIZE:-64g}" \
  --user "$(id -u):$(id -g)" \
  --env HTTP_PROXY --env HTTPS_PROXY --env http_proxy --env https_proxy \
  --env ALL_PROXY --env all_proxy --env NO_PROXY --env no_proxy \
  --env CUDA_HOME=/usr/local/cuda --env CUDA_PATH=/usr/local/cuda \
  --env HOME=/tmp/spatial-internalization-home \
  --env HF_HOME=/cache/huggingface --env "HF_ENDPOINT=${HF_ENDPOINT_VALUE}" \
  --env HF_HUB_DISABLE_TELEMETRY=1 --env HF_HUB_DISABLE_XET=1 \
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
    python -c "import accelerate, peft, PIL, torch, transformers; assert torch.cuda.is_available(), torch.cuda.device_count(); print(torch.__version__, torch.cuda.get_device_name(0))"
    exec python -m spatial_internalization.evaluate "$@"
  ' bash "$@"
