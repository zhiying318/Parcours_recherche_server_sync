#!/usr/bin/env bash
set -euo pipefail

# Test 04: Single-image MCQ on comfort_human_car_visual_marks
# Each query: 1 image, 4-choice spatial relation MCQ (short / middle / long answer lengths)
# 144 images total (4 cam views x 9 objects x 4 relations)
#
# Step 0 (one-time): regenerate image list if dataset changes
#   cd /home/zzou
#   python comfort_eval_tests/test04_visual_marks/generate_image_paths.py
#
# Step 1: run evaluation (below)

IMAGES_JSON="$(dirname "$0")/image_paths.json"
RESULTS_DIR="$(dirname "$0")/results"
LENGTHS=(short middle long)

run_single_mcq() {
  local cuda_device="$1"
  local backend="$2"
  local model_id="$3"
  local model_name="$4"
  local length="$5"
  shift 5

  CUDA_VISIBLE_DEVICES="$cuda_device" python -m spatial_eval.cli \
    --backend "$backend" \
    --model_id "$model_id" \
    --image_json "$IMAGES_JSON" \
    --out_csv "$RESULTS_DIR/mcq_${length}_${model_name}.csv" \
    --ask_mode mcq \
    --answer_length "$length" \
    --mcq_seed 123 \
    "$@"
}

run_gpu1_queue() {
  for length in "${LENGTHS[@]}"; do
    run_single_mcq 1 internvl OpenGVLab/InternVL3_5-8B-HF internvl "$length"
  done

  for length in "${LENGTHS[@]}"; do
    run_single_mcq 1 qwen3vl Qwen/Qwen3-VL-8B-Instruct qwen3vl "$length"
  done

  for length in "${LENGTHS[@]}"; do
    run_single_mcq 1 gemma4 google/gemma-4-E4B-it gemma4 "$length"
  done

}

run_gpu0_queue() {
  for length in "${LENGTHS[@]}"; do
    run_single_mcq 0 qwen3.5vl Qwen/Qwen3.5-9B qwen3_5vl "$length"
  done

  for length in "${LENGTHS[@]}"; do
    run_single_mcq 0 gemma4 google/gemma-4-E4B-it gemma4_thinking "$length" \
      --enable_thinking \
      --max_new_tokens_mcq 81920
  done

  for length in "${LENGTHS[@]}"; do
    run_single_mcq 0 internvl OpenGVLab/InternVL3_5-8B-HF internvl_thinking "$length" \
      --enable_thinking \
      --max_new_tokens_mcq 81920
  done

  for length in "${LENGTHS[@]}"; do
    run_single_mcq 0 qwen3-vl-thinking Qwen/Qwen3-VL-8B-Thinking qwen3vl_thinking "$length" \
      --max_new_tokens_mcq 81920
  done

  for length in "${LENGTHS[@]}"; do
    run_single_mcq 0 qwen3.5vl-thinking Qwen/Qwen3.5-9B qwen3_5vl_thinking "$length" \
      --max_new_tokens_mcq 81920
  done
}

run_gpu1_queue &
pid_gpu1=$!

run_gpu0_queue &
pid_gpu0=$!

status=0
wait "$pid_gpu1" || status=$?
wait "$pid_gpu0" || status=$?

if [[ "$status" -eq 0 ]]; then
  echo "All done."
else
  echo "One or more evaluation queues failed." >&2
fi
exit "$status"
