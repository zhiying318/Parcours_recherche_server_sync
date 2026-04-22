#!/usr/bin/env bash
# Test 03: 4-choice POV image MCQ with two objects in the scene
#
# Step 0 (one-time): generate the dataset
#   cd /home/zzou/COMFORT
#   python data_generation/generate_dataset_pov_two.py --save_path ../COMFORT/data
#
# Step 1 (one-time): generate the quads JSON
#   cd /home/zzou
#   python comfort_eval_tests/test03_pov_four/generate_quads.py \
#     --data_root COMFORT/data/comfort_human_car_pov_two \
#     --output comfort_eval_tests/test03_pov_four/image_quads.json
#
# Step 2: run evaluation (below)

QUADS_JSON="$(dirname "$0")/image_quads.json"
RESULTS_DIR="$(dirname "$0")/results"

# ---------- Qwen3-VL ----------
CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
  --backend qwen3vl \
  --model_id Qwen/Qwen3-VL-8B-Instruct \
  --image_quads_json "$QUADS_JSON" \
  --pov4_mode \
  --out_csv "$RESULTS_DIR/pov4_qwen3vl.csv" \
  --ask_mode mcq \
  --mcq_seed 123

# ---------- Qwen3.5-VL thinking (uncomment when needed) ----------
# CUDA_VISIBLE_DEVICES=1 python -m spatial_eval.cli \
#   --backend qwen3.5vl-thinking \
#   --model_id Qwen/Qwen3.5-9B \
#   --image_quads_json "$QUADS_JSON" \
#   --pov4_mode \
#   --out_csv "$RESULTS_DIR/pov4_qwen3_5vl_thinking.csv" \
#   --ask_mode mcq \
#   --max_new_tokens_mcq 81920 \
#   --mcq_seed 123

echo "All done."
