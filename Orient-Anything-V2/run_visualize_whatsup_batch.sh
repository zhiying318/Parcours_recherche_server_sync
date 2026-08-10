#!/usr/bin/env bash
set -euo pipefail

python -u visualize_results.py \
  --results "outputs/whatsup_qwen_image_edit_2509_v0/${ORIANY_VARIANT:-human_crop}/results.json" \
  --output-dir "outputs/whatsup_qwen_image_edit_2509_v0/${ORIANY_VARIANT:-human_crop}/visualizations_official" \
  --renderer blender
