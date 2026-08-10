#!/usr/bin/env bash
set -euo pipefail

IMAGE="${GEOMETRY_IMAGE:-sjtu-geometry-teacher:v0.1}"
GPU_DEVICE="${GEOMETRY_GPU:?Set GEOMETRY_GPU explicitly after checking that the GPU is free}"
PROJECT_HOST="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
PROJECT_CONTAINER="/workspace/project"
CACHE_HOST="${GEOMETRY_CACHE:-${PROJECT_HOST}/.cache/geometry_teacher}"

if [[ ! "${GPU_DEVICE}" =~ ^[0-9]+$ ]]; then
  echo "GEOMETRY_GPU must be one numeric GPU index, got: ${GPU_DEVICE}" >&2
  exit 2
fi
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Docker image not found: ${IMAGE}" >&2
  echo "Build it with: geometry_teacher/deploy/build_image.sh" >&2
  exit 2
fi

mkdir -p "${CACHE_HOST}"

exec docker run --rm --init \
  --name "geometry-teacher-${USER:-user}-$(date +%Y%m%d-%H%M%S)" \
  --network host \
  --gpus "device=${GPU_DEVICE}" \
  --shm-size 16g \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp/geometry-teacher-home \
  --env HF_HOME=/cache/huggingface \
  --env TORCH_HOME=/cache/torch \
  --env GEOMETRY_CUDA_MEMORY_FRACTION="${GEOMETRY_CUDA_MEMORY_FRACTION:-}" \
  --env HTTP_PROXY="${HTTP_PROXY:-}" \
  --env HTTPS_PROXY="${HTTPS_PROXY:-}" \
  --env NO_PROXY="${NO_PROXY:-}" \
  --env http_proxy="${http_proxy:-}" \
  --env https_proxy="${https_proxy:-}" \
  --env no_proxy="${no_proxy:-}" \
  --volume "${PROJECT_HOST}:${PROJECT_CONTAINER}" \
  --volume "${CACHE_HOST}:/cache" \
  --workdir "${PROJECT_CONTAINER}" \
  "${IMAGE}" \
  python -m geometry_teacher.cli "$@"
