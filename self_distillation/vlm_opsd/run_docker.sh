#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 GPU_IDS [training arguments...]" >&2
  echo "Example: $0 0,1,3,5 --max-steps 100" >&2
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
IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
NUM_PROCESSES="${#GPU_ARRAY[@]}"
# Account for both micro-batch size and explicitly requested accumulation.
PER_DEVICE_BATCH_SIZE=1
REQUESTED_ACCUMULATION=""
TRAIN_ARGS=("$@")
for ((i=0; i<${#TRAIN_ARGS[@]}; i++)); do
  case "${TRAIN_ARGS[i]}" in
    --per-device-batch-size) PER_DEVICE_BATCH_SIZE="${TRAIN_ARGS[i+1]:-}" ;;
    --per-device-batch-size=*) PER_DEVICE_BATCH_SIZE="${TRAIN_ARGS[i]#*=}" ;;
    --gradient-accumulation-steps) REQUESTED_ACCUMULATION="${TRAIN_ARGS[i+1]:-}" ;;
    --gradient-accumulation-steps=*) REQUESTED_ACCUMULATION="${TRAIN_ARGS[i]#*=}" ;;
  esac
done
for VALUE in "$PER_DEVICE_BATCH_SIZE" "${REQUESTED_ACCUMULATION:-1}"; do
  if [[ ! "$VALUE" =~ ^[1-9][0-9]*$ ]]; then
    echo "Batch size and gradient accumulation must be positive integers" >&2
    exit 2
  fi
done
MICRO_GLOBAL_BATCH_SIZE=$((NUM_PROCESSES * PER_DEVICE_BATCH_SIZE))
GRADIENT_ACCUMULATION_STEPS="${REQUESTED_ACCUMULATION:-1}"
TARGET_GLOBAL_BATCH_SIZE="${OPSD_GLOBAL_BATCH_SIZE:-$((MICRO_GLOBAL_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))}"
if [[ ! "$TARGET_GLOBAL_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] ||
   (( TARGET_GLOBAL_BATCH_SIZE % MICRO_GLOBAL_BATCH_SIZE != 0 )); then
  echo "OPSD_GLOBAL_BATCH_SIZE must be a positive multiple of GPU count * per-device batch ($MICRO_GLOBAL_BATCH_SIZE)" >&2
  exit 2
fi
GRADIENT_ACCUMULATION_STEPS=$((TARGET_GLOBAL_BATCH_SIZE / MICRO_GLOBAL_BATCH_SIZE))
if [[ -n "$REQUESTED_ACCUMULATION" && "$REQUESTED_ACCUMULATION" != "$GRADIENT_ACCUMULATION_STEPS" ]]; then
  echo "Explicit gradient accumulation conflicts with OPSD_GLOBAL_BATCH_SIZE" >&2
  exit 2
fi
echo "GPUs=$NUM_PROCESSES per_device_batch=$PER_DEVICE_BATCH_SIZE accumulation=$GRADIENT_ACCUMULATION_STEPS global_batch=$TARGET_GLOBAL_BATCH_SIZE"

mkdir -p "$HOST_HF_CACHE" "$HOST_PYTHON_PACKAGES"
docker image inspect "$IMAGE" >/dev/null

exec docker run --rm --init \
  --name "opsd-vlm-${USER:-user}-$(date +%Y%m%d-%H%M%S)" \
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
  --env WANDB_API_KEY \
  --env WANDB_ENTITY \
  --env WANDB_MODE \
  --env "OPSD_NUM_PROCESSES=${NUM_PROCESSES}" \
  --env "OPSD_GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}" \
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
    python -c "import accelerate, causal_conv1d, datasets, fla, flash_attn, peft, torch, transformers, wandb; assert torch.__version__.startswith(\"2.8.\"), torch.__version__"
    if [[ "$OPSD_NUM_PROCESSES" == 1 ]]; then
      exec python -m self_distillation.vlm_opsd.train \
        --gradient-accumulation-steps "$OPSD_GRADIENT_ACCUMULATION_STEPS" "$@"
    fi
    accelerate launch \
      --config_file self_distillation/vlm_opsd/accelerate_multi_gpu.yaml \
      --num_processes "$OPSD_NUM_PROCESSES" \
      -m self_distillation.vlm_opsd.train \
      --gradient-accumulation-steps "$OPSD_GRADIENT_ACCUMULATION_STEPS" "$@"
  ' bash "$@"
