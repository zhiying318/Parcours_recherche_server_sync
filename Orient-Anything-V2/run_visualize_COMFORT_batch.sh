#!/usr/bin/env bash
set -euo pipefail

python -u visualize_results.py \
  --results "outputs/comfort_human_car/${ORIANY_VARIANT:-human_crop}/results.json" \
  --output-dir "outputs/comfort_human_car/${ORIANY_VARIANT:-human_crop}/visualizations_official" \
  --renderer blender
