#!/usr/bin/env bash
# Same Docker image, dataset mount and dependency cache as test07.
set -euo pipefail

IMAGE="${COMFORT_ADD_PROMPT_IMAGE:-${ORIANY_IMAGE:-docker.v2.aispeech.com/sjtu/sjtu_chenlu-zzy-cuda_12.4-ubuntu_22.04-torch_2.6-orient-anything:v0.1}}"
GPU_SPEC="${COMFORT_ADD_PROMPT_GPU:-all}"
PROJECT_HOST="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
PROJECT_CONTAINER="/workspace/project"
DATASET_HOST="${PROJECT_HOST}/COMFORT/data"
if [[ ! -d "$DATASET_HOST/comfort_human_car_geometry_gt" ]]; then
  DATASET_HOST="${PROJECT_HOST}/.worktrees/geometry-teacher/COMFORT/data"
fi
if [[ ! -d "$DATASET_HOST/comfort_human_car_geometry_gt" ]]; then
  echo "Geometry-GT dataset not found in the main checkout or geometry-teacher worktree." >&2
  exit 1
fi
HOST_HF_CACHE="${COMFORT_HF_CACHE:-${HOME}/.cache/huggingface}"
HOST_PYTHON_PACKAGES="${COMFORT_PYTHON_PACKAGES:-${HOME}/.cache/comfort-additionalprompt/python}"
HF_ENDPOINT_VALUE="${HF_ENDPOINT:-https://huggingface.co}"
HF_HUB_DOWNLOAD_TIMEOUT_VALUE="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"
HF_HUB_ETAG_TIMEOUT_VALUE="${HF_HUB_ETAG_TIMEOUT:-60}"
HF_HUB_DISABLE_XET_VALUE="${HF_HUB_DISABLE_XET:-0}"
FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1%2Bcu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl}"

mkdir -p "$HOST_HF_CACHE"
mkdir -p "$HOST_PYTHON_PACKAGES"

nvidia-smi >/dev/null 2>&1 || { echo "No working NVIDIA GPU on $(hostname)." >&2; exit 1; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "Docker image not found: $IMAGE" >&2; exit 1; }

GPU_ARGS=()
if [[ "$GPU_SPEC" == "all" ]]; then
  GPU_ARGS+=(--gpus all)
else
  GPU_ARGS+=(--gpus "device=${GPU_SPEC}")
fi

exec docker run --rm --init \
  --name "comfort-additionalprompt-qwen35-test08-thinking-budget-${USER:-user}-$(date +%Y%m%d-%H%M%S)" \
  --network host \
  --env HTTP_PROXY --env HTTPS_PROXY --env http_proxy --env https_proxy --env NO_PROXY --env no_proxy \
  "${GPU_ARGS[@]}" --shm-size "${COMFORT_ADD_PROMPT_SHM_SIZE:-16g}" \
  --user "$(id -u):$(id -g)" --env HOME=/tmp/comfort-additionalprompt-home \
  --env "HF_HOME=/cache/huggingface" \
  --env "HF_ENDPOINT=${HF_ENDPOINT_VALUE}" \
  --env "HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT_VALUE}" \
  --env "HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT_VALUE}" \
  --env "HF_HUB_DISABLE_TELEMETRY=1" \
  --env "HF_HUB_DISABLE_XET=${HF_HUB_DISABLE_XET_VALUE}" \
  --env HF_TOKEN \
  --env "FLASH_ATTN_WHEEL_URL=${FLASH_ATTN_WHEEL_URL}" \
  --env "PYTHONUSERBASE=/cache/python" \
  --env "PYTHONPATH=/cache/python/lib/python3.11/site-packages" \
  --env "PATH=/cache/python/bin:/opt/conda/bin:/usr/local/bin:/usr/bin:/bin" \
  --volume "${PROJECT_HOST}:${PROJECT_CONTAINER}" \
  --volume "${DATASET_HOST}:${PROJECT_CONTAINER}/COMFORT/data:ro" \
  --volume "${HOST_HF_CACHE}:/cache/huggingface" \
  --volume "${HOST_PYTHON_PACKAGES}:/cache/python" \
  --workdir "${PROJECT_CONTAINER}" \
  "$IMAGE" bash -lc '
    set -euo pipefail
    python -c "import qwen_vl_utils" 2>/dev/null || python -m pip install --user --no-cache-dir qwen-vl-utils
    python -c "import flash_attn; assert flash_attn.__version__ == \"2.7.4.post1\"" 2>/dev/null || python -m pip install --user --no-cache-dir "$FLASH_ATTN_WHEEL_URL"
    exec bash comfort_addionalprompt_tests/test08_thinking_budget/run_qwen35.sh "$@"
  ' comfort-thinking-budget "$@"
