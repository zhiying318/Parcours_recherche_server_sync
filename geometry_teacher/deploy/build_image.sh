#!/usr/bin/env bash
set -euo pipefail

BASE_IMAGE="${ORIANY_IMAGE:-docker.v2.aispeech.com/sjtu/sjtu_chenlu-zzy-cuda_12.4-ubuntu_22.04-torch_2.6-orient-anything:v0.1}"
TARGET_IMAGE="${GEOMETRY_IMAGE:-sjtu-geometry-teacher:v0.1}"
PROJECT_HOST="$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")"
VGGT_COMMIT="${VGGT_COMMIT:-a288dd0f14786c93483e45524328726ab7b1b4ce}"
DOWNLOAD_DIR="${PROJECT_HOST}/geometry_teacher/deploy/.downloads"
VGGT_ARCHIVE="${DOWNLOAD_DIR}/vggt-${VGGT_COMMIT}.tar.gz"

mkdir -p "${DOWNLOAD_DIR}"
if [[ ! -f "${VGGT_ARCHIVE}" ]]; then
  TEMP_ARCHIVE="${VGGT_ARCHIVE}.part"
  curl --fail --location --retry 8 --retry-all-errors \
    --connect-timeout 30 --max-time 600 \
    --output "${TEMP_ARCHIVE}" \
    "https://codeload.github.com/facebookresearch/vggt/tar.gz/${VGGT_COMMIT}"
  mv "${TEMP_ARCHIVE}" "${VGGT_ARCHIVE}"
fi

if ! tar --list --gzip --file "${VGGT_ARCHIVE}" \
  "vggt-${VGGT_COMMIT}/vggt/models/vggt.py" >/dev/null; then
  echo "Invalid VGGT source archive: ${VGGT_ARCHIVE}" >&2
  exit 2
fi

exec docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "VGGT_COMMIT=${VGGT_COMMIT}" \
  --tag "${TARGET_IMAGE}" \
  --file "${PROJECT_HOST}/geometry_teacher/deploy/Dockerfile" \
  "${PROJECT_HOST}"
