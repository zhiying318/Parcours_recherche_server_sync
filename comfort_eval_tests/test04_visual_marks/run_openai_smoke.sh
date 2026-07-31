#!/usr/bin/env bash
set -euo pipefail

# Send exactly one paid Test 04 request.
# Usage: ./comfort_eval_tests/test04_visual_marks/run_openai_smoke.sh \
#          [sample_index] [short|middle|long]

SAMPLE_INDEX="${1:-0}"
ANSWER_LENGTH="${2:-long}"

: "${OPENAI_API_KEY:?Set OPENAI_API_KEY before running this script.}"
: "${OPENAI_BASE_URL:?Set OPENAI_BASE_URL before running this script.}"
: "${OPENAI_MODEL_ID:?Set OPENAI_MODEL_ID before running this script.}"

RUN_MODE="${OPENAI_RUN_MODE:-instruct}"
case "$RUN_MODE" in
  instruct) API_MODE="${OPENAI_API_MODE:-chat_completions}" ;;
  thinking) API_MODE="${OPENAI_API_MODE:-responses}" ;;
  *) echo "OPENAI_RUN_MODE must be instruct or thinking." >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"

cd "$PROJECT_ROOT"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

python -m comfort_eval_tests.test04_visual_marks.smoke_test_openai \
  --sample_index "$SAMPLE_INDEX" \
  --answer_length "$ANSWER_LENGTH" \
  --run_mode "$RUN_MODE" \
  --api_mode "$API_MODE" \
  --reasoning_effort "${OPENAI_REASONING_EFFORT:-high}" \
  --reasoning_summary "${OPENAI_REASONING_SUMMARY:-auto}" \
  --max_output_tokens "${OPENAI_MAX_OUTPUT_TOKENS:-81920}" \
  --timeout "${OPENAI_TIMEOUT:-120}" \
  --max_retries "${OPENAI_SMOKE_MAX_RETRIES:-0}"
