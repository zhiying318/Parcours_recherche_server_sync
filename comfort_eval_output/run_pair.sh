#!/usr/bin/env bash
set -euo pipefail

# Pair-image MCQ on COMFORT comfort_human_car
# Each query: 2 images from the same scene with different camera views,
# 4-choice spatial relation MCQ (short / middle / long answer lengths).
#
# Expected pair list:
#   216 pairs total = 9 objects x 4 relations x C(4 camera views, 2)
#
# Step 0 (one-time): regenerate pair list if dataset changes
#   cd /home/zzou
#   python comfort_eval_output/get_comfort_pair_paths_json.py
#
# Step 1: run evaluation (below) run directly
#   ./comfort_eval_output/run_pair.sh

IMAGE_PAIRS_JSON="$(dirname "$0")/../comfort_image_pairs.json"
RESULTS_DIR="$(dirname "$0")"


run_gpu1_non_thinking() {
  # ---------- InternVL short ----------
  CUDA_VISIBLE_DEVICES=1 python -m spatial_eval.cli \
    --backend internvl \
    --model_id OpenGVLab/InternVL3_5-8B-HF \
    --image_pairs_json "$IMAGE_PAIRS_JSON" \
    --out_csv "$RESULTS_DIR/pair_mcq_short_internvl.csv" \
    --pair_mode \
    --ask_mode mcq \
    --answer_length short \
    --mcq_seed 123

  # ---------- InternVL middle ----------
  CUDA_VISIBLE_DEVICES=1 python -m spatial_eval.cli \
    --backend internvl \
    --model_id OpenGVLab/InternVL3_5-8B-HF \
    --image_pairs_json "$IMAGE_PAIRS_JSON" \
    --out_csv "$RESULTS_DIR/pair_mcq_middle_internvl.csv" \
    --pair_mode \
    --ask_mode mcq \
    --answer_length middle \
    --mcq_seed 123

  # ---------- InternVL long ----------
  CUDA_VISIBLE_DEVICES=1 python -m spatial_eval.cli \
    --backend internvl \
    --model_id OpenGVLab/InternVL3_5-8B-HF \
    --image_pairs_json "$IMAGE_PAIRS_JSON" \
    --out_csv "$RESULTS_DIR/pair_mcq_long_internvl.csv" \
    --pair_mode \
    --ask_mode mcq \
    --answer_length long \
    --mcq_seed 123

  # ---------- Qwen3-VL short ----------
  CUDA_VISIBLE_DEVICES=1 python -m spatial_eval.cli \
    --backend qwen3vl \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --image_pairs_json "$IMAGE_PAIRS_JSON" \
    --out_csv "$RESULTS_DIR/pair_mcq_short_qwen3vl.csv" \
    --pair_mode \
    --ask_mode mcq \
    --answer_length short \
    --mcq_seed 123

  # ---------- Qwen3-VL middle ----------
  CUDA_VISIBLE_DEVICES=1 python -m spatial_eval.cli \
    --backend qwen3vl \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --image_pairs_json "$IMAGE_PAIRS_JSON" \
    --out_csv "$RESULTS_DIR/pair_mcq_middle_qwen3vl.csv" \
    --pair_mode \
    --ask_mode mcq \
    --answer_length middle \
    --mcq_seed 123

  # ---------- Qwen3-VL long ----------
  CUDA_VISIBLE_DEVICES=1 python -m spatial_eval.cli \
    --backend qwen3vl \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --image_pairs_json "$IMAGE_PAIRS_JSON" \
    --out_csv "$RESULTS_DIR/pair_mcq_long_qwen3vl.csv" \
    --pair_mode \
    --ask_mode mcq \
    --answer_length long \
    --mcq_seed 123

  # ---------- Gemma4 short ----------
  CUDA_VISIBLE_DEVICES=1 python -m spatial_eval.cli \
    --backend gemma4 \
    --model_id google/gemma-4-E4B-it \
    --image_pairs_json "$IMAGE_PAIRS_JSON" \
    --out_csv "$RESULTS_DIR/pair_mcq_short_gemma4.csv" \
    --pair_mode \
    --ask_mode mcq \
    --answer_length short \
    --mcq_seed 123

  # ---------- Gemma4 middle ----------
  CUDA_VISIBLE_DEVICES=1 python -m spatial_eval.cli \
    --backend gemma4 \
    --model_id google/gemma-4-E4B-it \
    --image_pairs_json "$IMAGE_PAIRS_JSON" \
    --out_csv "$RESULTS_DIR/pair_mcq_middle_gemma4.csv" \
    --pair_mode \
    --ask_mode mcq \
    --answer_length middle \
    --mcq_seed 123

  # ---------- Gemma4 long ----------
  CUDA_VISIBLE_DEVICES=1 python -m spatial_eval.cli \
    --backend gemma4 \
    --model_id google/gemma-4-E4B-it \
    --image_pairs_json "$IMAGE_PAIRS_JSON" \
    --out_csv "$RESULTS_DIR/pair_mcq_long_gemma4.csv" \
    --pair_mode \
    --ask_mode mcq \
    --answer_length long \
    --mcq_seed 123
}

run_gpu0_thinking() {
  # ---------- Qwen3.5-VL short ----------
  CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
    --backend qwen3.5vl \
    --model_id Qwen/Qwen3.5-9B \
    --image_pairs_json "$IMAGE_PAIRS_JSON" \
    --out_csv "$RESULTS_DIR/pair_mcq_short_qwen3_5vl.csv" \
    --pair_mode \
    --ask_mode mcq \
    --answer_length short \
    --mcq_seed 123

  # ---------- Qwen3.5-VL middle ----------
  CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
    --backend qwen3.5vl \
    --model_id Qwen/Qwen3.5-9B \
    --image_pairs_json "$IMAGE_PAIRS_JSON" \
    --out_csv "$RESULTS_DIR/pair_mcq_middle_qwen3_5vl.csv" \
    --pair_mode \
    --ask_mode mcq \
    --answer_length middle \
    --mcq_seed 123

  # ---------- Qwen3.5-VL long ----------
  CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
    --backend qwen3.5vl \
    --model_id Qwen/Qwen3.5-9B \
    --image_pairs_json "$IMAGE_PAIRS_JSON" \
    --out_csv "$RESULTS_DIR/pair_mcq_long_qwen3_5vl.csv" \
    --pair_mode \
    --ask_mode mcq \
    --answer_length long \
    --mcq_seed 123

  # ---------- InternVL thinking short ----------
  CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
    --backend internvl \
    --model_id OpenGVLab/InternVL3_5-8B-HF \
    --image_pairs_json "$IMAGE_PAIRS_JSON" \
    --out_csv "$RESULTS_DIR/pair_mcq_short_internvl_thinking.csv" \
    --pair_mode \
    --ask_mode mcq \
    --answer_length short \
    --enable_thinking \
    --max_new_tokens_mcq 81920 \
    --mcq_seed 123

  # ---------- InternVL thinking middle ----------
  CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
    --backend internvl \
    --model_id OpenGVLab/InternVL3_5-8B-HF \
    --image_pairs_json "$IMAGE_PAIRS_JSON" \
    --out_csv "$RESULTS_DIR/pair_mcq_middle_internvl_thinking.csv" \
    --pair_mode \
    --ask_mode mcq \
    --answer_length middle \
    --enable_thinking \
    --max_new_tokens_mcq 81920 \
    --mcq_seed 123

  # ---------- InternVL thinking long ----------
  CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
    --backend internvl \
    --model_id OpenGVLab/InternVL3_5-8B-HF \
    --image_pairs_json "$IMAGE_PAIRS_JSON" \
    --out_csv "$RESULTS_DIR/pair_mcq_long_internvl_thinking.csv" \
    --pair_mode \
    --ask_mode mcq \
    --answer_length long \
    --enable_thinking \
    --max_new_tokens_mcq 81920 \
    --mcq_seed 123

  # ---------- Qwen3-VL thinking short ----------
  CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
    --backend qwen3-vl-thinking \
    --model_id Qwen/Qwen3-VL-8B-Thinking \
    --image_pairs_json "$IMAGE_PAIRS_JSON" \
    --out_csv "$RESULTS_DIR/pair_mcq_short_qwen3vl_thinking.csv" \
    --pair_mode \
    --ask_mode mcq \
    --answer_length short \
    --max_new_tokens_mcq 81920 \
    --mcq_seed 123

  # ---------- Qwen3-VL thinking middle ----------
  CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
    --backend qwen3-vl-thinking \
    --model_id Qwen/Qwen3-VL-8B-Thinking \
    --image_pairs_json "$IMAGE_PAIRS_JSON" \
    --out_csv "$RESULTS_DIR/pair_mcq_middle_qwen3vl_thinking.csv" \
    --pair_mode \
    --ask_mode mcq \
    --answer_length middle \
    --max_new_tokens_mcq 81920 \
    --mcq_seed 123

  # ---------- Qwen3-VL thinking long ----------
  CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
    --backend qwen3-vl-thinking \
    --model_id Qwen/Qwen3-VL-8B-Thinking \
    --image_pairs_json "$IMAGE_PAIRS_JSON" \
    --out_csv "$RESULTS_DIR/pair_mcq_long_qwen3vl_thinking.csv" \
    --pair_mode \
    --ask_mode mcq \
    --answer_length long \
    --max_new_tokens_mcq 81920 \
    --mcq_seed 123

  # ---------- Qwen3.5-VL thinking short ----------
  CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
    --backend qwen3.5vl-thinking \
    --model_id Qwen/Qwen3.5-9B \
    --image_pairs_json "$IMAGE_PAIRS_JSON" \
    --out_csv "$RESULTS_DIR/pair_mcq_short_qwen3_5vl_thinking.csv" \
    --pair_mode \
    --ask_mode mcq \
    --answer_length short \
    --max_new_tokens_mcq 81920 \
    --mcq_seed 123

  # ---------- Qwen3.5-VL thinking middle ----------
  CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
    --backend qwen3.5vl-thinking \
    --model_id Qwen/Qwen3.5-9B \
    --image_pairs_json "$IMAGE_PAIRS_JSON" \
    --out_csv "$RESULTS_DIR/pair_mcq_middle_qwen3_5vl_thinking.csv" \
    --pair_mode \
    --ask_mode mcq \
    --answer_length middle \
    --max_new_tokens_mcq 81920 \
    --mcq_seed 123

  # ---------- Qwen3.5-VL thinking long ----------
  CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
    --backend qwen3.5vl-thinking \
    --model_id Qwen/Qwen3.5-9B \
    --image_pairs_json "$IMAGE_PAIRS_JSON" \
    --out_csv "$RESULTS_DIR/pair_mcq_long_qwen3_5vl_thinking.csv" \
    --pair_mode \
    --ask_mode mcq \
    --answer_length long \
    --max_new_tokens_mcq 81920 \
    --mcq_seed 123

  # ---------- Gemma4 thinking short ----------
  CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
    --backend gemma4 \
    --model_id google/gemma-4-E4B-it \
    --image_pairs_json "$IMAGE_PAIRS_JSON" \
    --out_csv "$RESULTS_DIR/pair_mcq_short_gemma4_thinking.csv" \
    --pair_mode \
    --ask_mode mcq \
    --answer_length short \
    --enable_thinking \
    --max_new_tokens_mcq 81920 \
    --mcq_seed 123

  # ---------- Gemma4 thinking middle ----------
  CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
    --backend gemma4 \
    --model_id google/gemma-4-E4B-it \
    --image_pairs_json "$IMAGE_PAIRS_JSON" \
    --out_csv "$RESULTS_DIR/pair_mcq_middle_gemma4_thinking.csv" \
    --pair_mode \
    --ask_mode mcq \
    --answer_length middle \
    --enable_thinking \
    --max_new_tokens_mcq 81920 \
    --mcq_seed 123

  # ---------- Gemma4 thinking long ----------
  CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
    --backend gemma4 \
    --model_id google/gemma-4-E4B-it \
    --image_pairs_json "$IMAGE_PAIRS_JSON" \
    --out_csv "$RESULTS_DIR/pair_mcq_long_gemma4_thinking.csv" \
    --pair_mode \
    --ask_mode mcq \
    --answer_length long \
    --enable_thinking \
    --max_new_tokens_mcq 81920 \
    --mcq_seed 123
}

run_gpu1_non_thinking &
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
