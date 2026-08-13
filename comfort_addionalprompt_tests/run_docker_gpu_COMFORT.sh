#!/usr/bin/env bash
set -euo pipefail

# Run one prompt-ablation experiment inside the project Docker image.
# Usage:
#   bash comfort_addionalprompt_tests/run_docker_gpu_COMFORT.sh qwen35 test01_camera_side
#   bash comfort_addionalprompt_tests/run_docker_gpu_COMFORT.sh gemma  test01_camera_side

if [[ "$#" -eq 1 ]]; then
  MODEL_FAMILY="qwen35"
  EXPERIMENT="$1"
elif [[ "$#" -eq 2 ]]; then
  MODEL_FAMILY="$1"
  EXPERIMENT="$2"
else
  echo "Usage: $0 [qwen35|gemma] TEST_DIRECTORY_NAME" >&2
  echo "Example: $0 gemma test01_camera_side" >&2
  exit 2
fi

case "$MODEL_FAMILY" in
  qwen35|gemma) ;;
  *) echo "Unknown model family: $MODEL_FAMILY" >&2; exit 2 ;;
esac

case "$EXPERIMENT" in
  test00_baseline|test01_camera_side|test02_camera_coordinate_system|\
  test03_object_camera_xyz|test04_person_object_camera_xyz|test05_person_forward_axis|\
  test06_camera_coordinate_system_person_forward_axis|test07_camera_geometry_before_question) ;;
  *) echo "Unknown experiment: $EXPERIMENT" >&2; exit 2 ;;
esac

IMAGE="${COMFORT_ADD_PROMPT_IMAGE:-${ORIANY_IMAGE:-docker.v2.aispeech.com/sjtu/sjtu_chenlu-zzy-cuda_12.4-ubuntu_22.04-torch_2.6-orient-anything:v0.1}}"
GPU_SPEC="${COMFORT_ADD_PROMPT_GPU:-all}"
PROJECT_HOST="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
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
HF_ENDPOINT_VALUE="${HF_ENDPOINT:-https://hf-mirror.com}"
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
  --name "comfort-additionalprompt-${MODEL_FAMILY}-${EXPERIMENT}-${USER:-user}-$(date +%Y%m%d-%H%M%S)" \
  --network host \
  --env HTTP_PROXY --env HTTPS_PROXY --env http_proxy --env https_proxy --env NO_PROXY --env no_proxy \
  "${GPU_ARGS[@]}" --shm-size "${COMFORT_ADD_PROMPT_SHM_SIZE:-16g}" \
  --user "$(id -u):$(id -g)" --env HOME=/tmp/comfort-additionalprompt-home \
  --env "HF_HOME=/cache/huggingface" \
  --env "HF_ENDPOINT=${HF_ENDPOINT_VALUE}" \
  --env "HF_HUB_DISABLE_TELEMETRY=1" \
  --env "HF_HUB_DISABLE_XET=1" \
  --env HF_TOKEN \
  --env "PYTHONUSERBASE=/cache/python" \
  --env "PYTHONPATH=/cache/python/lib/python3.11/site-packages" \
  --env "PATH=/cache/python/bin:/opt/conda/bin:/usr/local/bin:/usr/bin:/bin" \
  --volume "${PROJECT_HOST}:${PROJECT_CONTAINER}" \
  --volume "${DATASET_HOST}:${PROJECT_CONTAINER}/COMFORT/data:ro" \
  --volume "${HOST_HF_CACHE}:/cache/huggingface" \
  --volume "${HOST_PYTHON_PACKAGES}:/cache/python" \
  --workdir "${PROJECT_CONTAINER}" \
  "$IMAGE" bash -lc \
  "python comfort_addionalprompt_tests/generate_data.py && \
   if [[ \"${MODEL_FAMILY}\" == \"qwen35\" ]]; then \
     (python -c 'import qwen_vl_utils; print(\"qwen-vl-utils: using cached installation\")' 2>/dev/null || \
      python -m pip install --user --no-cache-dir qwen-vl-utils) && \
     (python -c 'import flash_attn; assert flash_attn.__version__ == \"2.7.4.post1\", flash_attn.__version__; print(\"flash-attn 2.7.4.post1: using cached installation\")' 2>/dev/null || \
      python -m pip install --user --no-cache-dir '${FLASH_ATTN_WHEEL_URL}') && \
     if [[ \"${FLASH_ATTN_SMOKE:-0}\" == \"1\" ]]; then \
       CUDA_VISIBLE_DEVICES=\"${QWEN_GPU:-0}\" python -u comfort_addionalprompt_tests/smoke_test_flash_attn.py; \
     else \
       QWEN_GPU=\"${QWEN_GPU:-0}\" bash comfort_addionalprompt_tests/${EXPERIMENT}/run_qwen35.sh; \
     fi; \
   else \
     GEMMA_GPU=\"${GEMMA_GPU:-0}\" bash comfort_addionalprompt_tests/${EXPERIMENT}/run_gemma.sh; \
   fi"
