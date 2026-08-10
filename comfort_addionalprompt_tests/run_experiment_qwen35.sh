#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/run_experiment_common.sh" "$@"
QWEN_GPU="${QWEN_GPU:-0}"

output_csv="$RESULTS_DIR/mcq_long_qwen3_5vl.csv"
if needs_run "Qwen3.5-VL" "$output_csv"; then
  echo "[Qwen3.5-VL] GPU=${QWEN_GPU}: starting non-thinking evaluation" >&2
  CUDA_VISIBLE_DEVICES="$QWEN_GPU" python -u -m spatial_eval.cli \
    --backend qwen3.5vl \
    --model_id Qwen/Qwen3.5-9B \
    --out_csv "$output_csv" \
    "${COMMON_ARGS[@]}" "${PROMPT_ARGS[@]}"
fi

output_csv="$RESULTS_DIR/mcq_long_qwen3_5vl_thinking.csv"
if needs_run "Qwen3.5-VL thinking" "$output_csv"; then
  echo "[Qwen3.5-VL thinking] GPU=${QWEN_GPU}: starting evaluation" >&2
  CUDA_VISIBLE_DEVICES="$QWEN_GPU" python -u -m spatial_eval.cli \
    --backend qwen3.5vl-thinking \
    --model_id Qwen/Qwen3.5-9B \
    --out_csv "$output_csv" \
    --max_new_tokens_mcq 20480 \
    "${COMMON_ARGS[@]}" "${PROMPT_ARGS[@]}"
fi

