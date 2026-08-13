#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROMPT_INFO_BEFORE_QUESTION=1
exec bash "$SCRIPT_DIR/../run_experiment_qwen35.sh" "$SCRIPT_DIR"
