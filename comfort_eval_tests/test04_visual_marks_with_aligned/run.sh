#!/usr/bin/env bash
set -euo pipefail

# Test 04: Single-image MCQ on comfort_human_car_visual_marks
# Each query: 1 image, 4-choice spatial relation MCQ (short / middle / long answer lengths)
# 144 images total (4 cam views × 9 objects × 4 relations)
#
# Step 0 (one-time): regenerate image list if dataset changes
#   cd /home/zzou
#   python comfort_eval_tests/test04_visual_marks/generate_image_paths.py
#
# Step 1: run evaluation (below)

IMAGES_JSON="$(dirname "$0")/image_paths.json"
RESULTS_DIR="$(dirname "$0")/results"


run_gpu1_queue() {
# # ---------- Qwen3-VL long ----------
# CUDA_VISIBLE_DEVICES=1 python -m spatial_eval.cli \
#   --backend qwen3vl \
#   --model_id Qwen/Qwen3-VL-8B-Instruct \
#   --image_json "$IMAGES_JSON" \
#   --out_csv "$RESULTS_DIR/mcq_long_qwen3vl_with_aligned.csv" \
#   --ask_mode mcq \
#   --answer_length long \
#   --mcq_prompt_note "The arrows in the image indicate the person's face orientation." \
#   --mcq_seed 123

# # ---------- InternVL long ----------
# CUDA_VISIBLE_DEVICES=1 python -m spatial_eval.cli \
#   --backend internvl \
#   --model_id OpenGVLab/InternVL3_5-8B-HF \
#   --image_json "$IMAGES_JSON" \
#   --out_csv "$RESULTS_DIR/mcq_long_internvl_with_aligned.csv" \
#   --ask_mode mcq \
#   --answer_length long \
#   --mcq_prompt_note "The arrows in the image indicate the person's face orientation." \
#   --mcq_seed 123

  # # ---------- Gemma4 long ----------
  # CUDA_VISIBLE_DEVICES=1 python -m spatial_eval.cli \
  #   --backend gemma4 \
  #   --model_id google/gemma-4-E4B-it \
  #   --image_json "$IMAGES_JSON" \
  #   --out_csv "$RESULTS_DIR/mcq_long_gemma4_with_aligned.csv" \
  #   --ask_mode mcq \
  #   --answer_length long \
  #   --mcq_prompt_note "The arrows in the image indicate the person's face orientation." \
  #   --mcq_seed 123

  # ---------- InternVL thinking long ----------
  CUDA_VISIBLE_DEVICES=1 python -m spatial_eval.cli \
    --backend internvl \
    --model_id OpenGVLab/InternVL3_5-8B-HF \
    --image_json "$IMAGES_JSON" \
    --out_csv "$RESULTS_DIR/mcq_long_internvl_thinking_with_aligned.csv" \
    --ask_mode mcq \
    --answer_length long \
    --enable_thinking \
    --max_new_tokens_mcq 16384 \
    --mcq_prompt_note "The arrows in the image indicate the person's face orientation." \
    --mcq_seed 123

  # ---------- Gemma4 thinking long ----------
  CUDA_VISIBLE_DEVICES=1 python -m spatial_eval.cli \
    --backend gemma4 \
    --model_id google/gemma-4-E4B-it \
    --image_json "$IMAGES_JSON" \
    --out_csv "$RESULTS_DIR/mcq_long_gemma4_thinking_with_aligned.csv" \
    --ask_mode mcq \
    --answer_length long \
    --enable_thinking \
    --max_new_tokens_mcq 16384 \
    --mcq_prompt_note "The arrows in the image indicate the person's face orientation." \
    --mcq_seed 123
}

run_gpu0_thinking() {
  # ---------- Qwen3.5-VL long ----------
  CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
  --backend qwen3.5vl \
  --model_id Qwen/Qwen3.5-9B \
  --image_json "$IMAGES_JSON" \
  --out_csv "$RESULTS_DIR/mcq_long_qwen3_5vl_with_aligned.csv" \
  --ask_mode mcq \
  --answer_length long \
  --mcq_prompt_note "The arrows in the image indicate the person's face orientation." \
  --mcq_seed 123

  # # ---------- Qwen3.5-VL thinking short ----------
  # CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
  #   --backend qwen3.5vl-thinking \
  #   --model_id Qwen/Qwen3.5-9B \
  #   --image_json "$IMAGES_JSON" \
  #   --out_csv "$RESULTS_DIR/mcq_short_qwen3_5vl_thinking_with_aligned.csv" \
  #   --ask_mode mcq \
  #   --answer_length short \
  #   --max_new_tokens_mcq 81920 \
  #   --mcq_prompt_note "The arrows in the image indicate the person's face orientation." \
  #   --mcq_seed 123

  # # ---------- Qwen3.5-VL thinking middle ----------
  # CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
  #   --backend qwen3.5vl-thinking \
  #   --model_id Qwen/Qwen3.5-9B \
  #   --image_json "$IMAGES_JSON" \
  #   --out_csv "$RESULTS_DIR/mcq_middle_qwen3_5vl_thinking_with_aligned.csv" \
  #   --ask_mode mcq \
  #   --answer_length middle \
  #   --max_new_tokens_mcq 81920 \
  #   --mcq_prompt_note "The arrows in the image indicate the person's face orientation." \
  #   --mcq_seed 123

  # ---------- Qwen3.5-VL thinking long ---------- # didn't do this one, stoped 
  CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
    --backend qwen3.5vl-thinking \
    --model_id Qwen/Qwen3.5-9B \
    --image_json "$IMAGES_JSON" \
    --out_csv "$RESULTS_DIR/mcq_long_qwen3_5vl_thinking_with_aligned.csv" \
    --ask_mode mcq \
    --answer_length long \
    --max_new_tokens_mcq 20480 \
    --mcq_prompt_note "The arrows in the image indicate the person's face orientation." \
    --mcq_seed 123

  # ---------- Qwen3-VL thinking long ----------
  CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
    --backend qwen3-vl-thinking \
    --model_id Qwen/Qwen3-VL-8B-Thinking \
    --image_json "$IMAGES_JSON" \
    --out_csv "$RESULTS_DIR/mcq_long_qwen3vl_thinking_with_aligned.csv" \
    --ask_mode mcq \
    --answer_length long \
    --max_new_tokens_mcq 20480 \
    --mcq_prompt_note "The arrows in the image indicate the person's face orientation." \
    --mcq_seed 123
}

run_gpu1_queue &
pid_gpu1=$!

run_gpu0_thinking &
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
