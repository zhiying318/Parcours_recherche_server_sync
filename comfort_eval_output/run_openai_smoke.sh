#!/usr/bin/env bash
set -euo pipefail

# Send exactly one paid API request.
# Usage:
#   ./comfort_eval_output/run_openai_smoke.sh [single|pair] [short|middle|long] [sample_index]

MODE="${1:-single}"
ANSWER_LENGTH="${2:-long}"
SAMPLE_INDEX="${3:-0}"
RUN_MODE="${OPENAI_RUN_MODE:-instruct}"
MAX_OUTPUT_TOKENS="${OPENAI_MAX_OUTPUT_TOKENS:-81920}"

case "$MODE" in single|pair) ;; *) echo "Mode must be single or pair." >&2; exit 2 ;; esac
case "$ANSWER_LENGTH" in short|middle|long) ;; *) echo "Answer length must be short, middle, or long." >&2; exit 2 ;; esac

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script.}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL before running this script.}"
: "${OPENAI_MODEL_ID:?Set OPENAI_MODEL_ID before running this script.}"

case "$RUN_MODE" in
  instruct) API_MODE="${OPENAI_API_MODE:-chat_completions}" ;;
  thinking) API_MODE="${OPENAI_API_MODE:-responses}" ;;
  *) echo "OPENAI_RUN_MODE must be instruct or thinking." >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

python -m comfort_eval_output.smoke_test_openai \
  --mode "$MODE" \
  --answer_length "$ANSWER_LENGTH" \
  --sample_index "$SAMPLE_INDEX" \
  --run_mode "$RUN_MODE" \
  --api_mode "$API_MODE" \
  --reasoning_effort "${OPENAI_REASONING_EFFORT:-high}" \
  --reasoning_summary "${OPENAI_REASONING_SUMMARY:-auto}" \
  --max_output_tokens "$MAX_OUTPUT_TOKENS" \
  --timeout "${OPENAI_TIMEOUT:-120}" \
  --max_retries "${OPENAI_SMOKE_MAX_RETRIES:-0}"
