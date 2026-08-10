#!/usr/bin/env bash
set -euo pipefail

IMAGE="${ORIANY_IMAGE:-docker.v2.aispeech.com/sjtu/sjtu_chenlu-zzy-cuda_12.4-ubuntu_22.04-torch_2.6-orient-anything:v0.1}"
GPU_DEVICE="${ORIANY_GPU:-0}"
PROJECT_HOST="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
PROJECT_CONTAINER="/workspace/project"
FULL_IMAGE_ARGS=()
OUTPUT_VARIANT="human_crop"
if [[ "${ORIANY_FULL_IMAGE:-0}" == "1" ]]; then
  FULL_IMAGE_ARGS+=(--full-image)
  OUTPUT_VARIANT="full_image"
fi

nvidia-smi >/dev/null 2>&1 || { echo "No working NVIDIA GPU on $(hostname)." >&2; exit 1; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "Docker image not found: $IMAGE" >&2; exit 1; }

exec docker run --rm --init \
  --name "orianyv2-comfort-${USER:-user}-$(date +%Y%m%d-%H%M%S)" \
  --gpus "device=${GPU_DEVICE}" --shm-size 16g \
  --user "$(id -u):$(id -g)" --env HOME=/tmp/orianyv2-home \
  --env "ORIANY_OUTPUT_VARIANT=${OUTPUT_VARIANT}" \
  --volume "${PROJECT_HOST}:${PROJECT_CONTAINER}" \
  --workdir "${PROJECT_CONTAINER}/Orient-Anything-V2" \
  "$IMAGE" bash run_COMFORT_batch.sh \
    "${FULL_IMAGE_ARGS[@]}"
