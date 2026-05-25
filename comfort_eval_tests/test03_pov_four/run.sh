#!/usr/bin/env bash
set -euo pipefail

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


# run_gpu1_queue() {
#   # ---------- InternVL ----------
#   CUDA_VISIBLE_DEVICES=1 python -m spatial_eval.cli \
#     --backend internvl \
#     --model_id OpenGVLab/InternVL3_5-8B-HF \
#     --image_quads_json "$QUADS_JSON" \
#     --pov4_mode \
#     --out_csv "$RESULTS_DIR/pov4_internvl.csv" \
#     --ask_mode mcq \
#     --mcq_seed 123

#   # # ---------- Qwen3-VL ----------
#   # CUDA_VISIBLE_DEVICES=1 python -m spatial_eval.cli \
#   #   --backend qwen3vl \
#   #   --model_id Qwen/Qwen3-VL-8B-Instruct \
#   #   --image_quads_json "$QUADS_JSON" \
#   #   --pov4_mode \
#   #   --out_csv "$RESULTS_DIR/pov4_qwen3vl.csv" \
#   #   --ask_mode mcq \
#   #   --mcq_seed 123

#   # ---------- Gemma4 ----------
#   CUDA_VISIBLE_DEVICES=1 python -m spatial_eval.cli \
#     --backend gemma4 \
#     --model_id google/gemma-4-E4B-it \
#     --image_quads_json "$QUADS_JSON" \
#     --pov4_mode \
#     --out_csv "$RESULTS_DIR/pov4_gemma4.csv" \
#     --ask_mode mcq \
#     --mcq_seed 123

#   # ---------- Gemma4 thinking ----------
#   CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
#     --backend gemma4 \
#     --model_id google/gemma-4-E4B-it \
#     --image_quads_json "$QUADS_JSON" \
#     --pov4_mode \
#     --out_csv "$RESULTS_DIR/pov4_gemma4_thinking.csv" \
#     --ask_mode mcq \
#     --enable_thinking \
#     --max_new_tokens_mcq 81920 \
#     --mcq_seed 123
# }

run_gpu0_queue() {
  # # ---------- Qwen3.5-VL ----------
  # CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
  #   --backend qwen3.5vl \
  #   --model_id Qwen/Qwen3.5-9B \
  #   --image_quads_json "$QUADS_JSON" \
  #   --pov4_mode \
  #   --out_csv "$RESULTS_DIR/pov4_qwen3_5vl.csv" \
  #   --ask_mode mcq \
  #   --mcq_seed 123

  # ---------- InternVL thinking ----------
  CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
    --backend internvl \
    --model_id OpenGVLab/InternVL3_5-8B-HF \
    --image_quads_json "$QUADS_JSON" \
    --pov4_mode \
    --out_csv "$RESULTS_DIR/pov4_internvl_thinking.csv" \
    --ask_mode mcq \
    --enable_thinking \
    --max_new_tokens_mcq 16384 \
    --mcq_seed 123

  # ---------- Qwen3-VL thinking ----------
  CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
    --backend qwen3-vl-thinking \
    --model_id Qwen/Qwen3-VL-8B-Thinking \
    --image_quads_json "$QUADS_JSON" \
    --pov4_mode \
    --out_csv "$RESULTS_DIR/pov4_qwen3vl_thinking.csv" \
    --ask_mode mcq \
    --max_new_tokens_mcq 81920 \
    --mcq_seed 123

  # ---------- Qwen3.5-VL thinking ----------
  CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
    --backend qwen3.5vl-thinking \
    --model_id Qwen/Qwen3.5-9B \
    --image_quads_json "$QUADS_JSON" \
    --pov4_mode \
    --out_csv "$RESULTS_DIR/pov4_qwen3_5vl_thinking.csv" \
    --ask_mode mcq \
    --max_new_tokens_mcq 81920 \
    --mcq_seed 123
}

# run_gpu1_queue &
# pid_gpu1=$!

run_gpu0_queue &
pid_gpu0=$!

status=0
# wait "$pid_gpu1" || status=$?
wait "$pid_gpu0" || status=$?

if [[ "$status" -eq 0 ]]; then
  echo "All done."
else
  echo "One or more evaluation queues failed." >&2
fi
exit "$status"
