#!/usr/bin/env bash
set -euo pipefail

# Run this script on the bare-metal A800 host, not on a login/debug node.
IMAGE="${ORIANY_IMAGE:-docker.v2.aispeech.com/sjtu/sjtu_chenlu-zzy-cuda_12.4-ubuntu_22.04-torch_2.6-orient-anything:v0.1}"
GPU_DEVICE="${ORIANY_GPU:-0}"
PROJECT_HOST="$(realpath "$(dirname "${BASH_SOURCE[0]}")/..")"
PROJECT_CONTAINER="/workspace/project"

if ! nvidia-smi >/dev/null 2>&1; then
  echo "No working NVIDIA GPU on $(hostname). Log in to the A800 host first." >&2
  exit 1
fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Docker image not found: $IMAGE" >&2
  echo "Build it from the project root:" >&2
  echo "  docker build -t $IMAGE $PROJECT_HOST" >&2
  exit 1
fi

exec docker run --rm --init \
  --name "orianyv2-${USER:-user}-$(date +%Y%m%d-%H%M%S)" \
  --gpus "device=${GPU_DEVICE}" \
  --shm-size 16g \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp/orianyv2-home \
  --volume "${PROJECT_HOST}:${PROJECT_CONTAINER}" \
  --workdir "${PROJECT_CONTAINER}/Orient-Anything-V2" \
  "$IMAGE" \
  bash run_coco_batch.sh
