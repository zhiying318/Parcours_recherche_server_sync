#!/usr/bin/env bash
set -euo pipefail

# GPT-5 OpenAI-compatible evaluation on all WhatsUp images.
# Required: OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL_ID
# Select:   OPENAI_RUN_MODE=instruct|thinking

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script.}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL before running this script.}"
: "${OPENAI_MODEL_ID:?Set OPENAI_MODEL_ID before running this script.}"

RUN_MODE="${OPENAI_RUN_MODE:-instruct}"
MAX_OUTPUT_TOKENS="${OPENAI_MAX_OUTPUT_TOKENS:-81920}"
TIMEOUT="${OPENAI_TIMEOUT:-120}"
MAX_RETRIES="${OPENAI_MAX_RETRIES:-5}"
MODEL_TAG="${OPENAI_MODEL_ID//[^[:alnum:]_]/_}"

case "$RUN_MODE" in
  instruct)
    API_MODE="${OPENAI_API_MODE:-chat_completions}"
    RESULT_TAG="$MODEL_TAG"
    REASONING_ARGS=()
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
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
STEM="mcq_long_${RESULT_TAG}"

cd "$PROJECT_ROOT"
python "$SCRIPT_DIR/eval_mcq.py" \
  --backend openai \
  --model_id "$OPENAI_MODEL_ID" \
  --image_json "$SCRIPT_DIR/whatsup_image_validation/valide_image_paths.json" \
  --out_csv "$SCRIPT_DIR/results/${STEM}.csv" \
  --max_new_tokens_mcq "$MAX_OUTPUT_TOKENS" \
  --mcq_seed 123 \
  --openai_api_mode "$API_MODE" \
  --openai_timeout "$TIMEOUT" \
  --openai_max_retries "$MAX_RETRIES" \
  --openai_reasoning_jsonl "$SCRIPT_DIR/results/${STEM}_reasoning.jsonl" \
  --resume \
  "${REASONING_ARGS[@]}"
