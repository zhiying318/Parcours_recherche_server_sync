#!/usr/bin/env bash
set -euo pipefail

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


run_gpu1_queue() {
  # ---------- InternVL ----------
  CUDA_VISIBLE_DEVICES=1 python -m spatial_eval.cli \
    --backend internvl \
    --model_id OpenGVLab/InternVL3_5-8B-HF \
    --image_triples_json "$TRIPLES_JSON" \
    --pov_mode \
    --out_csv "$RESULTS_DIR/pov_internvl.csv" \
    --ask_mode mcq \
    --mcq_seed 123

  # ---------- Qwen3-VL ----------
  CUDA_VISIBLE_DEVICES=1 python -m spatial_eval.cli \
    --backend qwen3vl \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --image_triples_json "$TRIPLES_JSON" \
    --pov_mode \
    --out_csv "$RESULTS_DIR/pov_qwen3vl.csv" \
    --ask_mode mcq \
    --mcq_seed 123

  # ---------- Gemma4 ----------
  CUDA_VISIBLE_DEVICES=1 python -m spatial_eval.cli \
    --backend gemma4 \
    --model_id google/gemma-4-E4B-it \
    --image_triples_json "$TRIPLES_JSON" \
    --pov_mode \
    --out_csv "$RESULTS_DIR/pov_gemma4.csv" \
    --ask_mode mcq \
    --mcq_seed 123

}

run_gpu0_queue() {
  # ---------- Qwen3.5-VL ----------
  CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
    --backend qwen3.5vl \
    --model_id Qwen/Qwen3.5-9B \
    --image_triples_json "$TRIPLES_JSON" \
    --pov_mode \
    --out_csv "$RESULTS_DIR/pov_qwen3_5vl.csv" \
    --ask_mode mcq \
    --mcq_seed 123

  # ---------- Gemma4 thinking ----------
  CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
    --backend gemma4 \
    --model_id google/gemma-4-E4B-it \
    --image_triples_json "$TRIPLES_JSON" \
    --pov_mode \
    --out_csv "$RESULTS_DIR/pov_gemma4_thinking.csv" \
    --ask_mode mcq \
    --enable_thinking \
    --max_new_tokens_mcq 81920 \
    --mcq_seed 123

  # ---------- InternVL thinking ----------
  CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
    --backend internvl \
    --model_id OpenGVLab/InternVL3_5-8B-HF \
    --image_triples_json "$TRIPLES_JSON" \
    --pov_mode \
    --out_csv "$RESULTS_DIR/pov_internvl_thinking.csv" \
    --ask_mode mcq \
    --enable_thinking \
    --max_new_tokens_mcq 81920 \
    --mcq_seed 123

  # ---------- Qwen3-VL thinking ----------
  CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
    --backend qwen3-vl-thinking \
    --model_id Qwen/Qwen3-VL-8B-Thinking \
    --image_triples_json "$TRIPLES_JSON" \
    --pov_mode \
    --out_csv "$RESULTS_DIR/pov_qwen3vl_thinking.csv" \
    --ask_mode mcq \
    --max_new_tokens_mcq 81920 \
    --mcq_seed 123

  # ---------- Qwen3.5-VL thinking ----------
  CUDA_VISIBLE_DEVICES=0 python -m spatial_eval.cli \
    --backend qwen3.5vl-thinking \
    --model_id Qwen/Qwen3.5-9B \
    --image_triples_json "$TRIPLES_JSON" \
    --pov_mode \
    --out_csv "$RESULTS_DIR/pov_qwen3_5vl_thinking.csv" \
    --ask_mode mcq \
    --max_new_tokens_mcq 81920 \
    --mcq_seed 123
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
