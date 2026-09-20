#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/../.."
export PYTHONHASHSEED=0
export CUDA_VISIBLE_DEVICES="${QWEN_GPU:-0}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
exec python -u "$SCRIPT_DIR/run.py" "$@"
