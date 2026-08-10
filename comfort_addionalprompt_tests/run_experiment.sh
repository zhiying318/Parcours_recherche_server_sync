#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 EXPERIMENT_DIR" >&2
  exit 2
fi

EXPERIMENT_DIR="$1"
IMAGES_JSON="$EXPERIMENT_DIR/data/image_paths.json"
PROMPT_JSON="$EXPERIMENT_DIR/data/prompt_info.json"
RESULTS_DIR="$EXPERIMENT_DIR/results_preciseprompt"
QWEN_GPU="${QWEN_GPU:-0}"
GEMMA_GPU="${GEMMA_GPU:-1}"
RUN_GEMMA="${RUN_GEMMA:-0}"

# MCQAsker intentionally remains unchanged and uses hash(image_path) in its
# existing randomizer. Fix Python's per-process hash salt so every model and
# every ablation receives the same option order for a given image.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

mkdir -p "$RESULTS_DIR"
if [[ ! -f "$IMAGES_JSON" ]]; then
  echo "Missing $IMAGES_JSON; run generate_data.py first." >&2
  exit 2
fi

TOTAL_SAMPLES="$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1], encoding="utf-8"))))' "$IMAGES_JSON")"

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

COMMON_ARGS=(
  --image_json "$IMAGES_JSON"
  --device_map cuda:0
  --ask_mode mcq
  --answer_length long
  --mcq_seed 123
  --resume
)

run_qwen_queue() {
  local output_csv="$RESULTS_DIR/mcq_long_qwen3_5vl.csv"
  if needs_run "Qwen3.5-VL" "$output_csv"; then
    echo "[Qwen3.5-VL] GPU=${QWEN_GPU}: starting non-thinking evaluation" >&2
    CUDA_VISIBLE_DEVICES="$QWEN_GPU" python -u -m spatial_eval.cli \
      --backend qwen3.5vl \
      --model_id Qwen/Qwen3.5-9B \
      --out_csv "$output_csv" \
      "${COMMON_ARGS[@]}" "${PROMPT_ARGS[@]}" || return $?
  fi

  output_csv="$RESULTS_DIR/mcq_long_qwen3_5vl_thinking.csv"
  if needs_run "Qwen3.5-VL thinking" "$output_csv"; then
    echo "[Qwen3.5-VL] GPU=${QWEN_GPU}: starting thinking evaluation" >&2
    CUDA_VISIBLE_DEVICES="$QWEN_GPU" python -u -m spatial_eval.cli \
      --backend qwen3.5vl-thinking \
      --model_id Qwen/Qwen3.5-9B \
      --out_csv "$output_csv" \
      --max_new_tokens_mcq 20480 \
      "${COMMON_ARGS[@]}" "${PROMPT_ARGS[@]}" || return $?
  fi
}

run_gemma_queue() {
  local output_csv="$RESULTS_DIR/mcq_long_gemma4.csv"
  if needs_run "Gemma4" "$output_csv"; then
    echo "[Gemma4] GPU=${GEMMA_GPU}: starting non-thinking evaluation" >&2
    CUDA_VISIBLE_DEVICES="$GEMMA_GPU" python -u -m spatial_eval.cli \
      --backend gemma4 \
      --model_id google/gemma-4-E4B-it \
      --out_csv "$output_csv" \
      "${COMMON_ARGS[@]}" "${PROMPT_ARGS[@]}" || return $?
  fi

  output_csv="$RESULTS_DIR/mcq_long_gemma4_thinking.csv"
  if needs_run "Gemma4 thinking" "$output_csv"; then
    echo "[Gemma4] GPU=${GEMMA_GPU}: starting thinking evaluation" >&2
    CUDA_VISIBLE_DEVICES="$GEMMA_GPU" python -u -m spatial_eval.cli \
      --backend gemma4 \
      --model_id google/gemma-4-E4B-it \
      --enable_thinking \
      --out_csv "$output_csv" \
      --max_new_tokens_mcq 20480 \
      "${COMMON_ARGS[@]}" "${PROMPT_ARGS[@]}" || return $?
  fi
}

status=0
run_qwen_queue || status=$?
if [[ "$RUN_GEMMA" == "1" && "$status" -eq 0 ]]; then
  run_gemma_queue || status=$?
elif [[ "$RUN_GEMMA" == "1" ]]; then
  echo "[Gemma4] skipped because the Qwen queue failed." >&2
fi
exit "$status"
