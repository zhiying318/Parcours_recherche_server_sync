#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${MINIMAX_H3_VENV:-$ROOT_DIR/.venv-minimax-h3}"

# Edit this default or launch with: MINIMAX_H3_GPUS=4,5,6,7 ./minimax_h3/serve_a100.sh
export CUDA_VISIBLE_DEVICES="${MINIMAX_H3_GPUS:-0,1,2,3}"
export HF_HOME="${HF_HOME:-$ROOT_DIR/checkpoints/minimax_h3_hf_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ROOT_DIR/.cache/minimax-h3}"
export SGLANG_CACHE_DIR="${SGLANG_CACHE_DIR:-$XDG_CACHE_HOME/sglang}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$SGLANG_CACHE_DIR/inductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$SGLANG_CACHE_DIR/triton}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$SGLANG_CACHE_DIR}"

PORT="${MINIMAX_H3_PORT:-30010}"
LOCAL_MODEL_DIR="${MINIMAX_H3_MODEL_DIR:-$ROOT_DIR/checkpoints/MiniMax-H3}"
if [[ -f "$LOCAL_MODEL_DIR/.fl2va-download-complete" ]]; then
  MODEL_PATH="$LOCAL_MODEL_DIR"
elif [[ -d "$LOCAL_MODEL_DIR" && -n "$(find "$LOCAL_MODEL_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Local FL2VA download is incomplete. Resume it with minimax_h3/download_weights.sh." >&2
  exit 1
else
  MODEL_PATH="MiniMaxAI/MiniMax-H3"
fi
IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
num_gpus="${#gpu_ids[@]}"

if [[ ! -x "$VENV_DIR/bin/sglang" ]]; then
  echo "Missing $VENV_DIR. Run minimax_h3/setup.sh first." >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
  echo "No working NVIDIA driver/GPU is visible in this shell." >&2
  exit 1
fi
if [[ "$num_gpus" -ne 4 ]]; then
  echo "The resident A100-80GB preset requires exactly 4 GPUs; got $CUDA_VISIBLE_DEVICES." >&2
  echo "For fewer GPUs use the offload recipe documented in minimax_h3/README.md." >&2
  exit 1
fi

mkdir -p "$HF_HOME" "$SGLANG_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
echo "Using physical GPUs $CUDA_VISIBLE_DEVICES (exposed to SGLang as 0..3)"
echo "Hugging Face cache: $HF_HOME"
echo "Model path: $MODEL_PATH"

exec "$VENV_DIR/bin/sglang" serve \
  --model-path "$MODEL_PATH" \
  --model-variant fl2va \
  --num-gpus 4 \
  --tp-size 2 \
  --ulysses-degree 2 \
  --encoder-parallel auto \
  --performance-mode speed \
  --enable-torch-compile false \
  --host 127.0.0.1 \
  --port "$PORT" \
  "$@"
