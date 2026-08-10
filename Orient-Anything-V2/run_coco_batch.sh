#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/orianyv2_numba"
export LD_LIBRARY_PATH="${CONDA_PREFIX:-/opt/conda}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

nvidia-smi
python -c 'import torch; print("torch:", torch.__version__); print("CUDA build:", torch.version.cuda); print("CUDA available:", torch.cuda.is_available())'

python infer_batch.py \
  test_data/coco_person_samples/person_crops \
  test_data/coco_object_samples/object_crops \
  --output-dir outputs/coco_smoke
