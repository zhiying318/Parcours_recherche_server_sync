#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${MINIMAX_H3_VENV:-$ROOT_DIR/.venv-minimax-h3}"
MODEL_DIR="${MINIMAX_H3_MODEL_DIR:-$ROOT_DIR/checkpoints/MiniMax-H3}"
# Keep the token location created by `hf auth login` when HF_HOME is redirected
# to the project-local cache below. This path contains the token but is never
# printed, copied, or committed by this script.
if [[ -z "${HF_TOKEN_PATH:-}" && -x "$VENV_DIR/bin/python" ]]; then
  export HF_TOKEN_PATH="$(env -u HF_HOME -u HF_TOKEN_PATH "$VENV_DIR/bin/python" -c 'from huggingface_hub.constants import HF_TOKEN_PATH; print(HF_TOKEN_PATH)')"
fi
export HF_HOME="${HF_HOME:-$ROOT_DIR/checkpoints/minimax_h3_hf_cache}"
# Xet's high-performance transfer path is safe for this immutable checkpoint
# and substantially improves resumable multi-file downloads on fast links.
export HF_XET_HIGH_PERFORMANCE="${HF_XET_HIGH_PERFORMANCE:-1}"

if [[ ! -x "$VENV_DIR/bin/hf" ]]; then
  echo "Missing environment. Run minimax_h3/setup.sh first." >&2
  exit 1
fi

mkdir -p "$MODEL_DIR" "$HF_HOME"
echo "Downloading the official FL2VA partition (~144 GB) to $MODEL_DIR"
"$VENV_DIR/bin/hf" download MiniMaxAI/MiniMax-H3 \
  --include model_index.json \
  --include 'FL2VA/*' \
  --max-workers "${MINIMAX_H3_DOWNLOAD_WORKERS:-16}" \
  --local-dir "$MODEL_DIR"

touch "$MODEL_DIR/.fl2va-download-complete"
echo "FL2VA download complete: $MODEL_DIR"
