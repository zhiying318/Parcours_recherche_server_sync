#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${MINIMAX_H3_VENV:-$ROOT_DIR/.venv-minimax-h3}/bin/python"

"$PYTHON" "$ROOT_DIR/minimax_h3/smoke_i2v.py" \
  "$ROOT_DIR/Version0_dataset/QwenImage_v0/banana_left_of_chair_FACE-CAMERA.png" \
  "$ROOT_DIR/Version0_dataset/QwenImage_v0/ball-of-yarn_right_of_chair_FACE-CAMERA.png" \
  "$ROOT_DIR/Version0_dataset/QwenImage_v0/book_right_of_chair_FACE-CAMERA.png" \
  --motion orbit-left \
  --duration 5 \
  --steps 50 \
  "$@"
