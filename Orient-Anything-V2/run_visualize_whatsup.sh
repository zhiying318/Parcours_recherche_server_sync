#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(realpath "$(dirname "${BASH_SOURCE[0]}")")"
ORIANY_ENV="${ORIANY_CONDA_ENV:-/hpc_stor03/sjtu_home/zhiying.zou/miniconda3/envs/orianyv2}"
VARIANT="${ORIANY_VARIANT:-human_crop}"
if [[ "$VARIANT" != "human_crop" && "$VARIANT" != "full_image" ]]; then
  echo "ORIANY_VARIANT must be human_crop or full_image" >&2
  exit 2
fi
[[ -x "${ORIANY_ENV}/bin/python" ]] || { echo "Python not found: ${ORIANY_ENV}/bin/python" >&2; exit 1; }

export LD_LIBRARY_PATH="${ORIANY_ENV}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
cd "$SCRIPT_DIR"
exec "${ORIANY_ENV}/bin/python" -u visualize_results.py \
  --results "outputs/whatsup_qwen_image_edit_2509_v0/${VARIANT}/results.json" \
  --output-dir "outputs/whatsup_qwen_image_edit_2509_v0/${VARIANT}/visualizations_official" \
  --renderer blender
