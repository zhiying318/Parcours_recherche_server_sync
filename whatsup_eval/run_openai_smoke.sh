#!/usr/bin/env bash
set -euo pipefail

# Send exactly one paid WhatsUp request.
# Usage: ./whatsup_eval/run_openai_smoke.sh [sample_index]

SAMPLE_INDEX="${1:-0}"
: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script.}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL before running this script.}"
: "${OPENAI_MODEL_ID:?Set OPENAI_MODEL_ID before running this script.}"

RUN_MODE="${OPENAI_RUN_MODE:-instruct}"
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
STEM="smoke_mcq_long_${RESULT_TAG}_sample_${SAMPLE_INDEX}"

cd "$PROJECT_ROOT"
python "$SCRIPT_DIR/eval_mcq.py" \
  --backend openai \
  --model_id "$OPENAI_MODEL_ID" \
  --image_json "$SCRIPT_DIR/whatsup_image_validation/valide_image_paths.json" \
  --sample_index "$SAMPLE_INDEX" \
  --out_csv "$SCRIPT_DIR/results/${STEM}.csv" \
  --max_new_tokens_mcq "${OPENAI_MAX_OUTPUT_TOKENS:-81920}" \
  --mcq_seed 123 \
  --openai_api_mode "$API_MODE" \
  --openai_timeout "${OPENAI_TIMEOUT:-120}" \
  --openai_max_retries "${OPENAI_SMOKE_MAX_RETRIES:-0}" \
  --openai_reasoning_jsonl "$SCRIPT_DIR/results/${STEM}_reasoning.jsonl" \
  "${REASONING_ARGS[@]}"
