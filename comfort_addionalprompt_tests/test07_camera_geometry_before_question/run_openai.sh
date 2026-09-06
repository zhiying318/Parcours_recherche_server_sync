#!/usr/bin/env bash
set -euo pipefail

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script.}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL before running this script.}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results_preciseprompt"
MODEL_ID="${OPENAI_MODEL_ID:-gpt-5}"
MAX_OUTPUT_TOKENS="${OPENAI_MAX_OUTPUT_TOKENS:-81920}"

mkdir -p "$RESULTS_DIR"
cd "$PROJECT_ROOT"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

COMMON_ARGS=(
  --backend openai
  --model_id "$MODEL_ID"
  --image_json "$SCRIPT_DIR/data/image_paths.json"
  --mcq_prompt_info_json "$SCRIPT_DIR/data/prompt_info.json"
  --mcq_prompt_info_before_question
  --ask_mode mcq
  --answer_length long
  --max_new_tokens_mcq "$MAX_OUTPUT_TOKENS"
  --mcq_seed 123
  --resume
)

echo "GPT-5: non-thinking"
python -m spatial_eval.cli \
  "${COMMON_ARGS[@]}" \
  --out_csv "$RESULTS_DIR/mcq_long_gpt_5.csv" \
  --openai_api_mode chat_completions \
  --openai_timeout "${OPENAI_TIMEOUT:-120}" \
  --openai_max_retries "${OPENAI_MAX_RETRIES:-5}" \
  --openai_reasoning_jsonl "$RESULTS_DIR/mcq_long_gpt_5_reasoning.jsonl"

echo "GPT-5: thinking"
python -m spatial_eval.cli \
  "${COMMON_ARGS[@]}" \
  --out_csv "$RESULTS_DIR/mcq_long_gpt_5_thinking.csv" \
  --openai_api_mode responses \
  --openai_reasoning_effort "${OPENAI_REASONING_EFFORT:-high}" \
  --openai_reasoning_summary "${OPENAI_REASONING_SUMMARY:-auto}" \
  --openai_timeout "${OPENAI_TIMEOUT:-120}" \
  --openai_max_retries "${OPENAI_MAX_RETRIES:-5}" \
  --openai_reasoning_jsonl "$RESULTS_DIR/mcq_long_gpt_5_thinking_reasoning.jsonl"
