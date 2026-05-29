#!/usr/bin/env bash
set -euo pipefail

# WhatsUp human-centered spatial reasoning.
# Main evaluation: single validated image + long 4-choice MCQ.
# Ground truth is computed from the edited WhatsUp filename.
#
# Step 0:
#   cd /home/zzou/whatsup_eval
#   python generate_image_paths.py
#
# Step 1:
#   ./run.sh

IMAGES_JSON="$(dirname "$0")/image_paths.json"
RESULTS_DIR="$(dirname "$0")/results"

mkdir -p "$RESULTS_DIR"


# run_gpu1_queue() {
#   # ---------- InternVL3.5 instruct ----------
#   CUDA_VISIBLE_DEVICES=1 python "$(dirname "$0")/eval_mcq.py" \
#     --backend internvl \
#     --model_id OpenGVLab/InternVL3_5-8B-HF \
#     --image_json "$IMAGES_JSON" \
#     --out_csv "$RESULTS_DIR/mcq_long_internvl.csv" \
#     --max_new_tokens_mcq 8 \
#     --mcq_seed 123

#   # ---------- InternVL3.5 thinking ----------
#   CUDA_VISIBLE_DEVICES=1 python "$(dirname "$0")/eval_mcq.py" \
#     --backend internvl \
#     --model_id OpenGVLab/InternVL3_5-8B-HF \
#     --image_json "$IMAGES_JSON" \
#     --out_csv "$RESULTS_DIR/mcq_long_internvl_thinking.csv" \
#     --enable_thinking \
#     --max_new_tokens_mcq 16384 \
#     --mcq_seed 123

#   # ---------- Gemma4 instruct ----------
#   CUDA_VISIBLE_DEVICES=1 python "$(dirname "$0")/eval_mcq.py" \
#     --backend gemma4 \
#     --model_id google/gemma-4-E4B-it \
#     --image_json "$IMAGES_JSON" \
#     --out_csv "$RESULTS_DIR/mcq_long_gemma4.csv" \
#     --max_new_tokens_mcq 8 \
#     --mcq_seed 123

#   # ---------- Gemma4 thinking ----------
#   CUDA_VISIBLE_DEVICES=1 python "$(dirname "$0")/eval_mcq.py" \
#     --backend gemma4 \
#     --model_id google/gemma-4-E4B-it \
#     --image_json "$IMAGES_JSON" \
#     --out_csv "$RESULTS_DIR/mcq_long_gemma4_thinking.csv" \
#     --enable_thinking \
#     --max_new_tokens_mcq 81920 \
#     --mcq_seed 123
# }


run_gpu0_queue() {
  # ---------- Qwen3.5-VL instruct ----------
  CUDA_VISIBLE_DEVICES=0 python "$(dirname "$0")/eval_mcq.py" \
    --backend qwen3.5vl \
    --model_id Qwen/Qwen3.5-9B \
    --image_json "$IMAGES_JSON" \
    --out_csv "$RESULTS_DIR/mcq_long_qwen3_5vl.csv" \
    --max_new_tokens_mcq 8 \
    --mcq_seed 123

  # ---------- Qwen3.5-VL thinking ----------
  CUDA_VISIBLE_DEVICES=0 python "$(dirname "$0")/eval_mcq.py" \
    --backend qwen3.5vl-thinking \
    --model_id Qwen/Qwen3.5-9B \
    --image_json "$IMAGES_JSON" \
    --out_csv "$RESULTS_DIR/mcq_long_qwen3_5vl_thinking.csv" \
    --max_new_tokens_mcq 20480 \
    --mcq_seed 123

  # ---------- Qwen3-VL instruct ----------
  CUDA_VISIBLE_DEVICES=0 python "$(dirname "$0")/eval_mcq.py" \
    --backend qwen3vl \
    --model_id Qwen/Qwen3-VL-8B-Instruct \
    --image_json "$IMAGES_JSON" \
    --out_csv "$RESULTS_DIR/mcq_long_qwen3vl.csv" \
    --max_new_tokens_mcq 8 \
    --mcq_seed 123

  # ---------- Qwen3-VL thinking ----------
  CUDA_VISIBLE_DEVICES=0 python "$(dirname "$0")/eval_mcq.py" \
    --backend qwen3-vl-thinking \
    --model_id Qwen/Qwen3-VL-8B-Thinking \
    --image_json "$IMAGES_JSON" \
    --out_csv "$RESULTS_DIR/mcq_long_qwen3vl_thinking.csv" \
    --max_new_tokens_mcq 20480 \
    --mcq_seed 123
}

run_gpu0_queue &
pid_gpu0=$!

# run_gpu1_queue &
# pid_gpu1=$!

status=0
wait "$pid_gpu0" || status=$?
# wait "$pid_gpu1" || status=$?

if [[ "$status" -eq 0 ]]; then
  echo "All done."
else
  echo "One or more evaluation queues failed." >&2
fi
exit "$status"
