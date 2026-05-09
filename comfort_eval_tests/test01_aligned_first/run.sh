#!/usr/bin/env bash
# Test 01: Aligned-first dual-image MCQ
# img1 = cam_back (camera behind person, same facing direction = aligned perspective)
# img2 = one of cam_front / cam_left / cam_right
# 120 pairs total (3 second-views × 40 object-relation scenes)

PAIRS_JSON="$(dirname "$0")/image_pairs.json"
RESULTS_DIR="$(dirname "$0")/results"

# # ---------- Qwen3-VL short ----------
# CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
#   --backend qwen3vl \
#   --model_id Qwen/Qwen3-VL-8B-Instruct \
#   --image_pairs_json "$PAIRS_JSON" \
#   --pair_mode \
#   --out_csv "$RESULTS_DIR/pair_mcq_short_qwen3vl.csv" \
#   --ask_mode mcq \
#   --answer_length short \
#   --mcq_seed 123

# # ---------- Qwen3-VL middle ----------
# CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
#   --backend qwen3vl \
#   --model_id Qwen/Qwen3-VL-8B-Instruct \
#   --image_pairs_json "$PAIRS_JSON" \
#   --pair_mode \
#   --out_csv "$RESULTS_DIR/pair_mcq_middle_qwen3vl.csv" \
#   --ask_mode mcq \
#   --answer_length middle \
#   --mcq_seed 123

# # ---------- Qwen3-VL long ----------
# CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
#   --backend qwen3vl \
#   --model_id Qwen/Qwen3-VL-8B-Instruct \
#   --image_pairs_json "$PAIRS_JSON" \
#   --pair_mode \
#   --out_csv "$RESULTS_DIR/pair_mcq_long_qwen3vl.csv" \
#   --ask_mode mcq \
#   --answer_length long \
#   --mcq_seed 123


# ---------- Qwen3.5-VL thinking short ----------
CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
  --backend qwen3.5vl-thinking \
  --model_id Qwen/Qwen3.5-9B \
  --image_pairs_json "$PAIRS_JSON" \
  --pair_mode \
  --out_csv "$RESULTS_DIR/pair_mcq_short_qwen3_5vl_thinking.csv" \
  --ask_mode mcq \
  --answer_length short \
  --max_new_tokens_mcq 81920 \
  --mcq_seed 123

# ---------- Qwen3.5-VL thinking middle ----------
CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
  --backend qwen3.5vl-thinking \
  --model_id Qwen/Qwen3.5-9B \
  --image_pairs_json "$PAIRS_JSON" \
  --pair_mode \
  --out_csv "$RESULTS_DIR/pair_mcq_middle_qwen3_5vl_thinking.csv" \
  --ask_mode mcq \
  --answer_length middle \
  --max_new_tokens_mcq 81920 \
  --mcq_seed 123

# ---------- Qwen3.5-VL thinking long ----------
CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
  --backend qwen3.5vl-thinking \
  --model_id Qwen/Qwen3.5-9B \
  --image_pairs_json "$PAIRS_JSON" \
  --pair_mode \
  --out_csv "$RESULTS_DIR/pair_mcq_long_qwen3_5vl_thinking.csv" \
  --ask_mode mcq \
  --answer_length long \
  --max_new_tokens_mcq 81920 \
  --mcq_seed 123

# ---------- Gemma4 short ----------
CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
  --backend gemma4 \
  --model_id google/gemma-4-E2B-it \
  --image_pairs_json "$PAIRS_JSON" \
  --pair_mode \
  --out_csv "$RESULTS_DIR/pair_mcq_short_gemma4.csv" \
  --ask_mode mcq \
  --answer_length short \
  --mcq_seed 123

# ---------- Gemma4 middle ----------
CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
  --backend gemma4 \
  --model_id google/gemma-4-E2B-it \
  --image_pairs_json "$PAIRS_JSON" \
  --pair_mode \
  --out_csv "$RESULTS_DIR/pair_mcq_middle_gemma4.csv" \
  --ask_mode mcq \
  --answer_length middle \
  --mcq_seed 123

# ---------- Gemma4 long ----------
CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
  --backend gemma4 \
  --model_id google/gemma-4-E2B-it \
  --image_pairs_json "$PAIRS_JSON" \
  --pair_mode \
  --out_csv "$RESULTS_DIR/pair_mcq_long_gemma4.csv" \
  --ask_mode mcq \
  --answer_length long \
  --mcq_seed 123
