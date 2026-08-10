#!/usr/bin/env bash
set -euo pipefail

IMAGE="${GEOMETRY_IMAGE:-sjtu-geometry-teacher:v0.2}"
PROJECT_HOST="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
PROJECT_CONTAINER="/workspace/project"
CACHE_HOST="${GEOMETRY_CACHE:-${PROJECT_HOST}/.cache/geometry_teacher}"

if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "Docker image not found: ${IMAGE}" >&2
  echo "Build it with GEOMETRY_IMAGE=${IMAGE} geometry_teacher/deploy/build_image.sh" >&2
  exit 2
fi

mapfile -t ACTIVE_PIPELINES < <(
  docker ps \
    --filter "name=geometry-teacher-${USER:-user}-" \
    --format '{{.Names}}'
)
if (( ${#ACTIVE_PIPELINES[@]} > 0 )); then
  echo "Refusing to prefetch while a Geometry Teacher pipeline is using the shared cache:" >&2
  printf '  %s\n' "${ACTIVE_PIPELINES[@]}" >&2
  echo "Stop that pipeline from its original terminal, then rerun this script." >&2
  exit 2
fi

mkdir -p "${CACHE_HOST}"

exec docker run --rm --init \
  --name "geometry-teacher-prefetch-${USER:-user}-$(date +%Y%m%d-%H%M%S)" \
  --network host \
  --user "$(id -u):$(id -g)" \
  --env HOME=/tmp/geometry-teacher-home \
  --env HF_HOME=/cache/huggingface \
  --env HF_HUB_DISABLE_XET=1 \
  --env HF_HUB_DISABLE_TELEMETRY=1 \
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
  python geometry_teacher/deploy/prefetch_models.py --cache-dir /cache/huggingface
