#!/usr/bin/env bash
set -euo pipefail

# Test 01: Aligned-first dual-image MCQ
# img1 = cam_back (camera behind person, same facing direction = aligned perspective)
# img2 = one of cam_front / cam_left / cam_right
# 108 pairs total (3 second-views x 36 object-relation scenes)

PAIRS_JSON="$(dirname "$0")/image_pairs.json"
RESULTS_DIR="$(dirname "$0")/results"
LENGTHS=(short middle long)

run_pair_mcq() {
  local cuda_device="$1"
  local backend="$2"
  local model_id="$3"
  local model_name="$4"
  local length="$5"
  shift 5

  CUDA_VISIBLE_DEVICES="$cuda_device" python -m spatial_eval.cli \
    --backend "$backend" \
    --model_id "$model_id" \
    --image_pairs_json "$PAIRS_JSON" \
    --pair_mode \
    --out_csv "$RESULTS_DIR/pair_mcq_${length}_${model_name}.csv" \
    --ask_mode mcq \
    --answer_length "$length" \
    --mcq_seed 123 \
    "$@"
}

run_gpu1_queue() {
  for length in "${LENGTHS[@]}"; do
    run_pair_mcq 1 internvl OpenGVLab/InternVL3_5-8B-HF internvl "$length"
  done

  for length in "${LENGTHS[@]}"; do
    run_pair_mcq 1 qwen3vl Qwen/Qwen3-VL-8B-Instruct qwen3vl "$length"
  done

  for length in "${LENGTHS[@]}"; do
    run_pair_mcq 1 gemma4 google/gemma-4-E4B-it gemma4 "$length"
  done

}

run_gpu0_queue() {
  for length in "${LENGTHS[@]}"; do
    run_pair_mcq 0 qwen3.5vl Qwen/Qwen3.5-9B qwen3_5vl "$length"
  done

  for length in "${LENGTHS[@]}"; do
    run_pair_mcq 0 gemma4 google/gemma-4-E4B-it gemma4_thinking "$length" \
      --enable_thinking \
      --max_new_tokens_mcq 81920
  done

  for length in "${LENGTHS[@]}"; do
    run_pair_mcq 0 internvl OpenGVLab/InternVL3_5-8B-HF internvl_thinking "$length" \
      --enable_thinking \
      --max_new_tokens_mcq 81920
  done

  for length in "${LENGTHS[@]}"; do
    run_pair_mcq 0 qwen3-vl-thinking Qwen/Qwen3-VL-8B-Thinking qwen3vl_thinking "$length" \
      --max_new_tokens_mcq 81920
  done

  for length in "${LENGTHS[@]}"; do
    run_pair_mcq 0 qwen3.5vl-thinking Qwen/Qwen3.5-9B qwen3_5vl_thinking "$length" \
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
