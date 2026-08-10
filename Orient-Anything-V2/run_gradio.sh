#!/usr/bin/env bash
set -euo pipefail

if [[ "${CONDA_DEFAULT_ENV:-}" != "orianyv2" ]]; then
  echo "Activate the environment first: conda activate orianyv2" >&2
  exit 1
fi

export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/orianyv2_numba"

cd "$(dirname "${BASH_SOURCE[0]}")"
exec python app.py
