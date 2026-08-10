#!/usr/bin/env bash
set -euo pipefail

IMAGE="${GEOMETRY_IMAGE:-sjtu-geometry-teacher:v0.2}"
GPU_DEVICE="${GEOMETRY_GPU:?Set GEOMETRY_GPU explicitly after checking that the GPU is free}"
PROJECT_HOST="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
CACHE_HOST="${GEOMETRY_CACHE:-${PROJECT_HOST}/.cache/geometry_teacher}"

if [[ ! "${GPU_DEVICE}" =~ ^[0-9]+$ ]]; then
  echo "GEOMETRY_GPU must be one numeric GPU index, got: ${GPU_DEVICE}" >&2
  exit 2
fi
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Docker image not found: ${IMAGE}" >&2
  exit 2
fi

mkdir -p "${CACHE_HOST}"
exec docker run --rm --init \
  --name "geometry-teacher-batch-${USER:-user}-gpu${GPU_DEVICE}-$(date +%Y%m%d-%H%M%S)" \
  --network host \
  --gpus "device=${GPU_DEVICE}" \
  --shm-size 16g \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp/geometry-teacher-home \
  --env HF_HOME=/cache/huggingface \
  --env HF_HUB_OFFLINE=1 \
  --env TRANSFORMERS_OFFLINE=1 \
  --env TORCH_HOME=/cache/torch \
  --env PYTHONPATH=/workspace/project \
  --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --env GEOMETRY_CUDA_MEMORY_FRACTION="${GEOMETRY_CUDA_MEMORY_FRACTION:-}" \
  --volume "${PROJECT_HOST}:/workspace/project" \
  --volume "${CACHE_HOST}:/cache" \
  --workdir /workspace/project \
  "${IMAGE}" python -m geometry_teacher.batch "$@"
