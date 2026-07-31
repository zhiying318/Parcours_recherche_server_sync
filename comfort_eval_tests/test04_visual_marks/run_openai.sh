#!/usr/bin/env bash
set -euo pipefail

# GPT-5 OpenAI-compatible evaluation for Test 04 visual marks.
# Usage:
#   ./comfort_eval_tests/test04_visual_marks/run_openai.sh [short|middle|long]
#
# Required: OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL_ID
# Select:   OPENAI_RUN_MODE=instruct|thinking

ANSWER_LENGTH="${1:-long}"
case "$ANSWER_LENGTH" in
  short|middle|long) ;;
  *) echo "Usage: $0 [short|middle|long]" >&2; exit 2 ;;
esac

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script.}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL before running this script.}"
: "${OPENAI_MODEL_ID:?Set OPENAI_MODEL_ID before running this script.}"

RUN_MODE="${OPENAI_RUN_MODE:-instruct}"
MODEL_TAG="${OPENAI_MODEL_ID//[^[:alnum:]_]/_}"
REASONING_ARGS=()

case "$RUN_MODE" in
  instruct)
    API_MODE="${OPENAI_API_MODE:-chat_completions}"
    RESULT_TAG="$MODEL_TAG"
    ;;
  thinking)
    API_MODE="${OPENAI_API_MODE:-responses}"
    RESULT_TAG="${MODEL_TAG}_thinking"
    REASONING_ARGS=(
      --openai_reasoning_effort "${OPENAI_REASONING_EFFORT:-high}"
    )
    if [[ "$API_MODE" == "responses" ]]; then
      REASONING_ARGS+=(
        --openai_reasoning_summary "${OPENAI_REASONING_SUMMARY:-auto}"
      )
    fi
    ;;
  *)
    echo "OPENAI_RUN_MODE must be instruct or thinking." >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
STEM="mcq_${ANSWER_LENGTH}_${RESULT_TAG}"

mkdir -p "$SCRIPT_DIR/results"
cd "$PROJECT_ROOT"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

echo "Test04: model=$OPENAI_MODEL_ID mode=$RUN_MODE length=$ANSWER_LENGTH api=$API_MODE"
python -m spatial_eval.cli \
  --backend openai \
  --model_id "$OPENAI_MODEL_ID" \
  --image_json "$SCRIPT_DIR/image_paths.json" \
  --out_csv "$SCRIPT_DIR/results/${STEM}.csv" \
  --ask_mode mcq \
  --answer_length "$ANSWER_LENGTH" \
  --max_new_tokens_mcq "${OPENAI_MAX_OUTPUT_TOKENS:-81920}" \
  --mcq_seed 123 \
  --openai_api_mode "$API_MODE" \
  --openai_timeout "${OPENAI_TIMEOUT:-120}" \
  --openai_max_retries "${OPENAI_MAX_RETRIES:-5}" \
  --openai_reasoning_jsonl "$SCRIPT_DIR/results/${STEM}_reasoning.jsonl" \
  --resume \
  "${REASONING_ARGS[@]}"
