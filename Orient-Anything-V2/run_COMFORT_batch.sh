#!/usr/bin/env bash
set -euo pipefail

python -u infer_human_datasets.py \
  --dataset comfort \
  --data-root /workspace/project/COMFORT/data/comfort_human_car \
  --output-dir "outputs/comfort_human_car/${ORIANY_OUTPUT_VARIANT:-human_crop}" \
  "$@"
