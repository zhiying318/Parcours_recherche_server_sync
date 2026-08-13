#!/usr/bin/env bash
# Shared setup for the model-specific experiment runners. Source this file.

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 EXPERIMENT_DIR" >&2
  exit 2
fi

EXPERIMENT_DIR="$1"
IMAGES_JSON="$EXPERIMENT_DIR/data/image_paths.json"
PROMPT_JSON="$EXPERIMENT_DIR/data/prompt_info.json"
RESULTS_DIR="$EXPERIMENT_DIR/results_preciseprompt"

export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

mkdir -p "$RESULTS_DIR"
if [[ ! -f "$IMAGES_JSON" ]]; then
  echo "Missing $IMAGES_JSON; run generate_data.py first." >&2
  exit 2
fi

TOTAL_SAMPLES="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' "$IMAGES_JSON")"

python - "$IMAGES_JSON" <<'PY'
import json
import sys
from pathlib import Path

image_paths = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
missing = [path for path in image_paths if not Path(path).is_file()]
if missing:
    examples = "\n".join(f"  - {path}" for path in missing[:5])
    raise FileNotFoundError(
        f"{len(missing)}/{len(image_paths)} evaluation images are missing. Examples:\n{examples}"
    )
print(f"Validated {len(image_paths)} evaluation images.")
PY

completed_samples() {
  local output_csv="$1"
  if [[ ! -s "$output_csv" ]]; then
    echo 0
    return
  fi
  python -c 'import csv,sys; rows=sum(1 for _ in csv.reader(open(sys.argv[1], newline="", encoding="utf-8"))); print(max(0, rows-1))' "$output_csv"
}

needs_run() {
  local label="$1"
  local output_csv="$2"
  local completed
  completed="$(completed_samples "$output_csv")"
  if (( completed >= TOTAL_SAMPLES )); then
    echo "[$label] already complete (${completed}/${TOTAL_SAMPLES}); skipping." >&2
    return 1
  fi
  echo "[$label] resuming from ${completed}/${TOTAL_SAMPLES} completed samples." >&2
  return 0
}

PROMPT_ARGS=()
if [[ -f "$PROMPT_JSON" ]]; then
  PROMPT_ARGS=(--mcq_prompt_info_json "$PROMPT_JSON")
fi
if [[ "${PROMPT_INFO_BEFORE_QUESTION:-0}" == "1" ]]; then
  PROMPT_ARGS+=(--mcq_prompt_info_before_question)
fi

COMMON_ARGS=(
  --image_json "$IMAGES_JSON"
  --device_map cuda:0
  --ask_mode mcq
  --answer_length long
  --mcq_seed 123
  --resume
)
