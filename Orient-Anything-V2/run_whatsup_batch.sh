#!/usr/bin/env bash
set -euo pipefail

python -u infer_human_datasets.py \
  --dataset whatsup \
  --data-root /workspace/project/Version0_dataset/Qwen_Image_Edit_2509_v0 \
  --output-dir "outputs/whatsup_qwen_image_edit_2509_v0/${ORIANY_OUTPUT_VARIANT:-human_crop}" \
  "$@"
