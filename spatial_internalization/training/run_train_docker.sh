#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 GPU_ID [training arguments...]" >&2
  echo "Example: $0 0 --variant fixed_template_reasoning" >&2
  exit 2
fi

GPU_ID="$1"
shift
IMAGE="${SPATIAL_INTERNALIZATION_IMAGE:-${COMFORT_ADD_PROMPT_IMAGE:-docker.v2.aispeech.com/sjtu/sjtu_chenlu-zzy-cuda_12.4-ubuntu_22.04-torch_2.6-orient-anything:v0.1}}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST_HF_CACHE="${SPATIAL_HF_CACHE:-${HOME}/.cache/huggingface}"
HOST_PYTHON_PACKAGES="${SPATIAL_PYTHON_PACKAGES:-${HOME}/.cache/spatial-internalization/python}"
HOST_CUDA_HOME="${SPATIAL_CUDA_HOME:-${CUDA_HOME:-/usr/local/cuda}}"
HF_ENDPOINT_VALUE="${HF_ENDPOINT:-https://hf-mirror.com}"

PER_DEVICE_BATCH_SIZE=3
GRADIENT_ACCUMULATION_STEPS=1
VARIANT=""
TRAIN_ARGS=("$@")
for ((i=0; i<${#TRAIN_ARGS[@]}; i++)); do
  case "${TRAIN_ARGS[i]}" in
    --per-device-batch-size) j=$((i + 1)); PER_DEVICE_BATCH_SIZE="${TRAIN_ARGS[j]:-}" ;;
    --per-device-batch-size=*) PER_DEVICE_BATCH_SIZE="${TRAIN_ARGS[i]#*=}" ;;
    --gradient-accumulation-steps) j=$((i + 1)); GRADIENT_ACCUMULATION_STEPS="${TRAIN_ARGS[j]:-}" ;;
    --gradient-accumulation-steps=*) GRADIENT_ACCUMULATION_STEPS="${TRAIN_ARGS[i]#*=}" ;;
    --variant) j=$((i + 1)); VARIANT="${TRAIN_ARGS[j]:-}" ;;
    --variant=*) VARIANT="${TRAIN_ARGS[i]#*=}" ;;
  esac
done
if [[ ! "$PER_DEVICE_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]] ||
   [[ ! "$GRADIENT_ACCUMULATION_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "Batch size and gradient accumulation must be positive integers" >&2
  exit 2
fi
if [[ -z "$VARIANT" ]]; then
  echo "run_docker.sh requires --variant fixed_template_reasoning|verbalized_reasoning" >&2
  exit 2
fi

mkdir -p "$HOST_HF_CACHE" "$HOST_PYTHON_PACKAGES"
docker image inspect "$IMAGE" >/dev/null

echo "GPU=$GPU_ID variant=$VARIANT per_device_batch=$PER_DEVICE_BATCH_SIZE accumulation=$GRADIENT_ACCUMULATION_STEPS effective_batch=$((PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))"
echo "W&B project=${WANDB_PROJECT:-spatial-internalization} mode=${WANDB_MODE:-online}"

exec docker run --rm --init \
  --name "spatial-internalization-${USER:-user}-$(date +%Y%m%d-%H%M%S)" \
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
  --env WANDB_API_KEY --env WANDB_ENTITY --env WANDB_PROJECT --env WANDB_MODE --env WANDB_NAME \
  --env PYTHONUSERBASE=/cache/python \
  --env PYTHONPATH=/cache/python/lib/python3.11/site-packages \
  --env PATH=/cache/python/bin:/opt/conda/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
  --env "SPATIAL_PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}" \
  --env "SPATIAL_GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}" \
  --volume "${REPO_ROOT}:/workspace/project" \
  --volume "${HOST_HF_CACHE}:/cache/huggingface" \
  --volume "${HOST_PYTHON_PACKAGES}:/cache/python:ro" \
  --volume "${HOST_CUDA_HOME}:/usr/local/cuda:ro" \
  --workdir /workspace/project \
  "$IMAGE" bash -lc '
    set -euo pipefail
    python -c "import accelerate, peft, PIL, torch, transformers, wandb; print(torch.__version__)"
    exec python -m spatial_internalization.training.train_sft \
      --per-device-batch-size "${SPATIAL_PER_DEVICE_BATCH_SIZE:-1}" \
      --gradient-accumulation-steps "${SPATIAL_GRADIENT_ACCUMULATION_STEPS:-4}" \
      --report-to wandb \
      --wandb-project "${WANDB_PROJECT:-spatial-internalization}" \
      "$@"
  ' bash \
  --per-device-batch-size "$PER_DEVICE_BATCH_SIZE" \
  --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
  "$@"
