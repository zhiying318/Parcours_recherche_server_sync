#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${MINIMAX_H3_VENV:-$ROOT_DIR/.venv-minimax-h3}"
MODEL_DIR="${MINIMAX_H3_MODEL_DIR:-$ROOT_DIR/checkpoints/MiniMax-H3}"

# Select one physical A100: MINIMAX_H3_GPU=5 ./minimax_h3/serve_one_a100.sh
export CUDA_VISIBLE_DEVICES="${MINIMAX_H3_GPU:-0}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/checkpoints/minimax_h3_hf_cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT_DIR/.cache/minimax-h3}"
export SGLANG_CACHE_DIR="${SGLANG_CACHE_DIR:-$XDG_CACHE_HOME/sglang}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$SGLANG_CACHE_DIR/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SGLANG_CACHE_DIR/triton}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$SGLANG_CACHE_DIR}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [[ ! -x "$VENV_DIR/bin/sglang" ]]; then
  echo "Missing $VENV_DIR. Run minimax_h3/setup.sh first." >&2
  exit 1
fi
if [[ ! -f "$MODEL_DIR/.fl2va-download-complete" ]]; then
  echo "FL2VA weights are incomplete. Resume with minimax_h3/download_weights.sh." >&2
  exit 1
fi

mkdir -p "$HF_HOME" "$SGLANG_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
exec "$VENV_DIR/bin/sglang" serve \
  --model-path "$MODEL_DIR" \
  --model-variant fl2va \
  --num-gpus 1 \
  --tp-size 1 \
  --ulysses-degree 1 \
  --performance-mode memory \
  --dit-layerwise-offload true \
  --layerwise-offload-components dit text_encoder vae \
  --dit-offload-prefetch-size 1 \
  --dit-layerwise-resident-layers 4 \
  --enable-torch-compile false \
  --host 127.0.0.1 \
  --port "${MINIMAX_H3_PORT:-30010}" \
  "$@"
