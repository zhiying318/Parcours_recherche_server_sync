#!/usr/bin/env bash
set -euo pipefail

IMAGE="${COMFORT_ADD_PROMPT_IMAGE:-${ORIANY_IMAGE:-docker.v2.aispeech.com/sjtu/sjtu_chenlu-zzy-cuda_12.4-ubuntu_22.04-torch_2.6-orient-anything:v0.1}}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST_PYTHON_PACKAGES="${OPSD_VLM_PYTHON_PACKAGES:-${HOME}/.cache/opsd-vlm/python-torch28}"
HOST_CUDA_HOME="${OPSD_CUDA_HOME:-${CUDA_HOME:-/usr/local/cuda}}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"
TORCH_VERSION="${TORCH_VERSION:-2.8.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.23.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.8.0}"
TRITON_VERSION="${TRITON_VERSION:-3.4.0}"
EINOPS_VERSION="${EINOPS_VERSION:-0.8.1}"
FLA_VERSION="${FLA_VERSION:-0.5.2}"
FLASH_ATTN_WHEEL_URL="${FLASH_ATTN_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl}"
CAUSAL_CONV1D_WHEEL_URL="${CAUSAL_CONV1D_WHEEL_URL:-https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/causal_conv1d-1.6.2.post1%2Bcu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl}"

if [[ ! -x "${HOST_CUDA_HOME}/bin/nvcc" ]]; then
  echo "CUDA Toolkit not found at ${HOST_CUDA_HOME}/bin/nvcc" >&2
  echo "Set OPSD_CUDA_HOME to a host CUDA Toolkit directory." >&2
  exit 2
fi

mkdir -p "$HOST_PYTHON_PACKAGES"
docker image inspect "$IMAGE" >/dev/null

exec docker run --rm --init \
  --network host \
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
  --env PATH=/opt/conda/bin:/usr/local/cuda/bin:/usr/local/bin:/usr/bin:/bin \
  --env PYTHONUSERBASE=/cache/python \
  --volume "${HOST_PYTHON_PACKAGES}:/cache/python" \
  --volume "${HOST_CUDA_HOME}:/usr/local/cuda:ro" \
  --volume "${REPO_ROOT}:/workspace/project:ro" \
  --workdir /workspace/project \
  "$IMAGE" bash -lc \
  "set -euo pipefail && \
   python -m pip install --user --upgrade --index-url '${TORCH_INDEX_URL}' \
     'torch==${TORCH_VERSION}' 'torchvision==${TORCHVISION_VERSION}' \
     'torchaudio==${TORCHAUDIO_VERSION}' && \
   python -m pip install --user --upgrade \
     'triton==${TRITON_VERSION}' 'einops==${EINOPS_VERSION}' && \
   python -m pip install --user --upgrade -r self_distillation/vlm_opsd/requirements-docker.txt && \
   python -m pip install --user --upgrade \
     'flash-linear-attention[cuda]==${FLA_VERSION}' && \
   python -m pip install --user --retries 10 --timeout 120 --no-deps \
     '${CAUSAL_CONV1D_WHEEL_URL}' '${FLASH_ATTN_WHEEL_URL}'"
