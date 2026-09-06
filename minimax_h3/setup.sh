#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${MINIMAX_H3_VENV:-$ROOT_DIR/.venv-minimax-h3}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT_DIR/.cache/uv-minimax-h3}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="$PYTHON_BIN"
elif command -v conda >/dev/null 2>&1 && conda env list | grep -q '^eval-notebook '; then
  python_bin="$(conda run -n eval-notebook which python)"
else
  python_bin="python3"
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required (https://docs.astral.sh/uv/)." >&2
  exit 1
fi

echo "Creating environment at $VENV_DIR with $python_bin"
mkdir -p "$UV_CACHE_DIR"
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  uv venv --python "$python_bin" "$VENV_DIR"
else
  echo "Reusing existing environment at $VENV_DIR"
fi
uv pip install --python "$VENV_DIR/bin/python" --prerelease=allow "sglang[diffusion]"

"$VENV_DIR/bin/python" - <<'PY'
import importlib.metadata
print("sglang", importlib.metadata.version("sglang"))
PY

echo "Environment is ready. Start the service with minimax_h3/serve_a100.sh"
