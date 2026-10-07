#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
if [[ -x "$REPO_ROOT/.venv-api/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv-api/bin/python"
else
  PYTHON_BIN="${SPATIAL_API_PYTHON:-python}"
fi

cd "$REPO_ROOT"
exec "$PYTHON_BIN" -m spatial_internalization.stage2_trajectory_guided.api_smoke_test "$@"
