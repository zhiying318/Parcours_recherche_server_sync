#!/usr/bin/env bash
set -euo pipefail

# Run all COMFORT evaluation tests with Mistral API instruct and reasoning modes.
# Required:
#   export MISTRAL_API_KEY=...
# Optional:
#   export MISTRAL_MODEL_ID=mistral-medium-3-5
#   export MISTRAL_INSTRUCT_TEMPERATURE=0.0
#   export MISTRAL_REASONING_TEMPERATURE=0.7
#   export MISTRAL_REASONING_TOP_P=0.95

MODEL_ID="${MISTRAL_MODEL_ID:-mistral-medium-3-5}"
INSTRUCT_TEMPERATURE="${MISTRAL_INSTRUCT_TEMPERATURE:-0.0}"
REASONING_TEMPERATURE="${MISTRAL_REASONING_TEMPERATURE:-0.7}"
REASONING_TOP_P="${MISTRAL_REASONING_TOP_P:-0.95}"

TEST01_DIR="$(dirname "$0")/test01_aligned_first"
TEST02_DIR="$(dirname "$0")/test02_pov"
TEST03_DIR="$(dirname "$0")/test03_pov_four"
TEST04_DIR="$(dirname "$0")/test04_visual_marks"
TEST04_ALIGNED_DIR="$(dirname "$0")/test04_visual_marks_with_aligned"

run_all_tests_for_mode() {
  local mode_name="$1"
  local reasoning_effort="$2"
  local temperature="$3"
  local top_p="${4:-}"
  local top_p_args=()

  if [[ -n "$top_p" ]]; then
    top_p_args=(--mistral_top_p "$top_p")
  fi

  echo "Running Mistral ${mode_name}: model=${MODEL_ID}, reasoning_effort=${reasoning_effort}, temperature=${temperature}, top_p=${top_p:-none}"

  # ---------- Test 01: pair MCQ ----------
  for answer_length in short middle long; do
    python -m spatial_eval.cli \
      --backend mistral \
      --model_id "$MODEL_ID" \
      --mistral_reasoning_effort "$reasoning_effort" \
      --mistral_temperature "$temperature" \
      "${top_p_args[@]}" \
      --image_pairs_json "$TEST01_DIR/image_pairs.json" \
      --pair_mode \
      --out_csv "$TEST01_DIR/results/pair_mcq_${answer_length}_mistral_${mode_name}.csv" \
      --ask_mode mcq \
      --answer_length "$answer_length" \
      --mcq_seed 123
  done

  # ---------- Test 02: two-option POV MCQ ----------
  python -m spatial_eval.cli \
    --backend mistral \
    --model_id "$MODEL_ID" \
    --mistral_reasoning_effort "$reasoning_effort" \
    --mistral_temperature "$temperature" \
    "${top_p_args[@]}" \
    --image_triples_json "$TEST02_DIR/image_triples.json" \
    --pov_mode \
    --out_csv "$TEST02_DIR/results/pov_mistral_${mode_name}.csv" \
    --ask_mode mcq \
    --mcq_seed 123

  # ---------- Test 03: three-option POV MCQ ----------
  python -m spatial_eval.cli \
    --backend mistral \
    --model_id "$MODEL_ID" \
    --mistral_reasoning_effort "$reasoning_effort" \
    --mistral_temperature "$temperature" \
    "${top_p_args[@]}" \
    --image_quads_json "$TEST03_DIR/image_quads.json" \
    --pov4_mode \
    --out_csv "$TEST03_DIR/results/pov4_mistral_${mode_name}.csv" \
    --ask_mode mcq \
    --mcq_seed 123

  # ---------- Test 04: single-image visual-mark MCQ ----------
  for answer_length in short middle long; do
    python -m spatial_eval.cli \
      --backend mistral \
      --model_id "$MODEL_ID" \
      --mistral_reasoning_effort "$reasoning_effort" \
      --mistral_temperature "$temperature" \
      "${top_p_args[@]}" \
      --image_json "$TEST04_DIR/image_paths.json" \
      --out_csv "$TEST04_DIR/results/mcq_${answer_length}_mistral_${mode_name}.csv" \
      --ask_mode mcq \
      --answer_length "$answer_length" \
      --mcq_seed 123
  done

  # ---------- Test 04 with aligned arrow note ----------
  for answer_length in short middle long; do
    python -m spatial_eval.cli \
      --backend mistral \
      --model_id "$MODEL_ID" \
      --mistral_reasoning_effort "$reasoning_effort" \
      --mistral_temperature "$temperature" \
      "${top_p_args[@]}" \
      --image_json "$TEST04_ALIGNED_DIR/image_paths.json" \
      --out_csv "$TEST04_ALIGNED_DIR/results/mcq_${answer_length}_mistral_${mode_name}_with_aligned.csv" \
      --ask_mode mcq \
      --answer_length "$answer_length" \
      --mcq_prompt_note "The arrows in the image indicate the person's face orientation." \
      --mcq_seed 123
  done
}

run_all_tests_for_mode "instruct" "none" "$INSTRUCT_TEMPERATURE"
run_all_tests_for_mode "reasoning" "high" "$REASONING_TEMPERATURE" "$REASONING_TOP_P"

echo "All Mistral instruct and reasoning evaluations done."
