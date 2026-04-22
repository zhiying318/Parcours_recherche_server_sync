#!/usr/bin/env bash
# Test 02: POV image-choice evaluation
#
# Step 0 (one-time): generate the dataset
#   cd /home/zzou/COMFORT
#   python data_generation/generate_dataset_pov.py --save_path ../COMFORT/data
#
# Step 1 (one-time): generate the triples JSON
#   cd /home/zzou
#   python comfort_eval_tests/test02_pov/generate_triples.py \
#     --data_root COMFORT/data/comfort_human_car_pov \
#     --output comfort_eval_tests/test02_pov/image_triples.json
#
# Step 2: run evaluation (below)

TRIPLES_JSON="$(dirname "$0")/image_triples.json"
RESULTS_DIR="$(dirname "$0")/results"

# ---------- Qwen3-VL ----------
CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
  --backend qwen3vl \
  --model_id Qwen/Qwen3-VL-8B-Instruct \
  --image_triples_json "$TRIPLES_JSON" \
  --pov_mode \
  --out_csv "$RESULTS_DIR/pov_qwen3vl.csv" \
  --ask_mode mcq \
  --mcq_seed 123

# ---------- Qwen3.5-VL thinking (uncomment when needed) ----------
# CUDA_VISIBLE_DEVICES=1 python -m spatial_eval.cli \
#   --backend qwen3.5vl-thinking \
#   --model_id Qwen/Qwen3.5-9B \
#   --image_triples_json "$TRIPLES_JSON" \
#   --pov_mode \
#   --out_csv "$RESULTS_DIR/pov_qwen3_5vl_thinking.csv" \
#   --ask_mode mcq \
#   --max_new_tokens_mcq 81920 \
#   --mcq_seed 123

echo "All done."
