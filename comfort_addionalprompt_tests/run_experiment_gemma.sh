#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_experiment_common.sh" "$@"
GEMMA_GPU="${GEMMA_GPU:-0}"

output_csv="$RESULTS_DIR/mcq_long_gemma4.csv"
if needs_run "Gemma4" "$output_csv"; then
  echo "[Gemma4] GPU=${GEMMA_GPU}: starting non-thinking evaluation" >&2
  CUDA_VISIBLE_DEVICES="$GEMMA_GPU" python -u -m spatial_eval.cli \
    --backend gemma4 \
    --model_id google/gemma-4-E4B-it \
    --out_csv "$output_csv" \
    "${COMMON_ARGS[@]}" "${PROMPT_ARGS[@]}"
fi

output_csv="$RESULTS_DIR/mcq_long_gemma4_thinking.csv"
if needs_run "Gemma4 thinking" "$output_csv"; then
  echo "[Gemma4 thinking] GPU=${GEMMA_GPU}: starting evaluation" >&2
  CUDA_VISIBLE_DEVICES="$GEMMA_GPU" python -u -m spatial_eval.cli \
    --backend gemma4 \
    --model_id google/gemma-4-E4B-it \
    --enable_thinking \
    --out_csv "$output_csv" \
    --max_new_tokens_mcq 20480 \
    "${COMMON_ARGS[@]}" "${PROMPT_ARGS[@]}"
fi

