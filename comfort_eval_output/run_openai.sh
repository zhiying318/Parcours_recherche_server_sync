#!/usr/bin/env bash
set -euo pipefail

# OpenAI-compatible API evaluation on COMFORT.
# Usage:
#   ./comfort_eval_output/run_openai.sh [short|middle|long]
#
# Required environment variables:
#   OPENAI_API_KEY
#   OPENAI_BASE_URL
#   OPENAI_MODEL_ID
#
# Main optional variables:
#   OPENAI_RUN_MODE=instruct|thinking        (default: instruct)
#   OPENAI_API_MODE=chat_completions|responses (optional override)
#   OPENAI_REASONING_EFFORT=high
#   OPENAI_REASONING_SUMMARY=auto            (Responses API only)
#   OPENAI_MAX_OUTPUT_TOKENS=81920
#   OPENAI_TIMEOUT=120
#   OPENAI_MAX_RETRIES=5

ANSWER_LENGTH="${1:-long}"
case "$ANSWER_LENGTH" in
  short|middle|long) ;;
  *)
    echo "Usage: $0 [short|middle|long]" >&2
    exit 2
    ;;
esac

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script.}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL before running this script.}"
: "${OPENAI_MODEL_ID:?Set OPENAI_MODEL_ID before running this script.}"

RUN_MODE="${OPENAI_RUN_MODE:-instruct}"
MAX_OUTPUT_TOKENS="${OPENAI_MAX_OUTPUT_TOKENS:-81920}"
REASONING_EFFORT="${OPENAI_REASONING_EFFORT:-high}"
REASONING_SUMMARY="${OPENAI_REASONING_SUMMARY:-auto}"
TIMEOUT="${OPENAI_TIMEOUT:-120}"
MAX_RETRIES="${OPENAI_MAX_RETRIES:-5}"
MODEL_TAG="${OPENAI_MODEL_ID//[^[:alnum:]_]/_}"

case "$RUN_MODE" in
  instruct)
    # Matches the successful gpt-5 instruct smoke test.
    API_MODE="${OPENAI_API_MODE:-chat_completions}"
    RESULT_TAG="$MODEL_TAG"
    REASONING_ARGS=()
    ;;
  thinking)
    # Matches the successful gpt-5 high-effort + summary smoke test.
    API_MODE="${OPENAI_API_MODE:-responses}"
    RESULT_TAG="${MODEL_TAG}_thinking"
    REASONING_ARGS=(--openai_reasoning_effort "$REASONING_EFFORT")
    if [[ "$API_MODE" == "responses" ]]; then
      REASONING_ARGS+=(--openai_reasoning_summary "$REASONING_SUMMARY")
    fi
    ;;
  *)
    echo "OPENAI_RUN_MODE must be instruct or thinking." >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
IMAGES_JSON="$PROJECT_ROOT/comfort_image_paths.json"
IMAGE_PAIRS_JSON="$PROJECT_ROOT/comfort_image_pairs.json"
SINGLE_STEM="mcq_${ANSWER_LENGTH}_${RESULT_TAG}"
PAIR_STEM="pair_mcq_${ANSWER_LENGTH}_${RESULT_TAG}"

cd "$PROJECT_ROOT"
# The prompt code currently derives shuffles from Python hash().
# Pinning the hash seed keeps option order stable across separate runs.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

COMMON_ARGS=(
  --backend openai
  --model_id "$OPENAI_MODEL_ID"
  --ask_mode mcq
  --answer_length "$ANSWER_LENGTH"
  --max_new_tokens_mcq "$MAX_OUTPUT_TOKENS"
  --mcq_seed 123
  --openai_api_mode "$API_MODE"
  --openai_timeout "$TIMEOUT"
  --openai_max_retries "$MAX_RETRIES"
  --resume
)

echo "Single-image: model=$OPENAI_MODEL_ID mode=$RUN_MODE length=$ANSWER_LENGTH api=$API_MODE"
python -m spatial_eval.cli \
  "${COMMON_ARGS[@]}" \
  "${REASONING_ARGS[@]}" \
  --image_json "$IMAGES_JSON" \
  --out_csv "$SCRIPT_DIR/${SINGLE_STEM}.csv" \
  --openai_reasoning_jsonl "$SCRIPT_DIR/${SINGLE_STEM}_reasoning.jsonl"

echo "Pair-image: model=$OPENAI_MODEL_ID mode=$RUN_MODE length=$ANSWER_LENGTH api=$API_MODE"
python -m spatial_eval.cli \
  "${COMMON_ARGS[@]}" \
  "${REASONING_ARGS[@]}" \
  --image_pairs_json "$IMAGE_PAIRS_JSON" \
  --pair_mode \
  --out_csv "$SCRIPT_DIR/${PAIR_STEM}.csv" \
  --openai_reasoning_jsonl "$SCRIPT_DIR/${PAIR_STEM}_reasoning.jsonl"
