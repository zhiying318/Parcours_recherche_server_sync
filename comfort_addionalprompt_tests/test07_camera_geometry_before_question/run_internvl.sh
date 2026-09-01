#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results_preciseprompt"
GPU="${INTERNVL_GPU:-0}"

mkdir -p "$RESULTS_DIR"
cd "$PROJECT_ROOT"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

COMMON_ARGS=(
  --backend internvl
  --model_id OpenGVLab/InternVL3_5-8B-HF
  --image_json "$SCRIPT_DIR/data/image_paths.json"
  --mcq_prompt_info_json "$SCRIPT_DIR/data/prompt_info.json"
  --mcq_prompt_info_before_question
  --ask_mode mcq
  --answer_length long
  --max_new_tokens_mcq 20480
  --mcq_seed 123
  --device_map cuda:0
  --resume
)

echo "InternVL3.5: non-thinking"
CUDA_VISIBLE_DEVICES="$GPU" python -u -m spatial_eval.cli \
  "${COMMON_ARGS[@]}" \
  --out_csv "$RESULTS_DIR/mcq_long_internvl.csv"

echo "InternVL3.5: thinking"
CUDA_VISIBLE_DEVICES="$GPU" python -u -m spatial_eval.cli \
  "${COMMON_ARGS[@]}" \
  --enable_thinking \
  --out_csv "$RESULTS_DIR/mcq_long_internvl_thinking.csv"
