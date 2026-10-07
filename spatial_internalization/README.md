# Spatial internalization

This package contains two sequential experiments. Stage 1 trains Qwen3.5-9B
with a deterministic geometry-reasoning template supplied by the researcher.
Stage 2 converts correct historical model trajectories into one consolidated
and verbalized reasoning dataset with Qwen3.5-397B-A17B. Stage 2 is one
experiment, not an answer-only/template/verbalized ablation.

## Layout

```text
core/                         shared geometry, serialization, local-Qwen helpers
stage1_fixed_template/        Stage 1 dataset builders and fixed verbalizer
stage2_trajectory_guided/     Stage 2 consolidation/verbalization pipeline
training/                     SFT training and internal evaluation
datasets/
  stage1_fixed_split/         original deterministic split
  stage1_object_cv/           nine leave-one-object-out folds
  stage2_trajectory_guided/   consolidated evidence, pilot, and future full data
checkpoints/
  stage1_fixed_split_nonthinking/
  stage1_object_cv_thinking/fold_N/
results/
  stage1_fixed_split_nonthinking/
  stage1_object_cv_thinking/fold_N/
tests/
```

The only Stage 1 variant name is `fixed_template_reasoning`. The only Stage 2
variant name is `verbalized_reasoning`.

## Current status

- Stage 1 fixed-split non-thinking training is complete. Its 16-example
  internal test accuracy is 100%.
- All nine object-held-out datasets are prepared.
- Thinking SFT checkpoints exist for folds 0 and 1. An internal evaluation is
  currently stored for fold 0.
- Stage 2 source filtering retained 113 of 144 historical trajectories and
  full consolidation is complete.
- Stage 2 verbalization has a pilot with three complete families. Full
  verbalization and Stage 2 SFT are not complete.

## Stage 1 data

Build the fixed split:

```bash
python -m spatial_internalization.stage1_fixed_template.build_fixed_split
```

Build object-held-out folds into a new empty output root:

```bash
python -m spatial_internalization.stage1_fixed_template.build_object_cv \
  --output-root spatial_internalization/datasets/stage1_object_cv_new
```

The existing artifacts are under `datasets/stage1_object_cv/fold_0` through
`fold_8`.

## Stage 1 thinking SFT

```bash
FOLD=2
bash spatial_internalization/training/run_train_docker.sh 0 \
  --data-root "spatial_internalization/datasets/stage1_object_cv/fold_${FOLD}" \
  --variant fixed_template_reasoning \
  --output-dir "spatial_internalization/checkpoints/stage1_object_cv_thinking/fold_${FOLD}" \
  --num-train-epochs 5 \
  --per-device-batch-size 16 \
  --gradient-accumulation-steps 1 \
  --learning-rate 5e-6 \
  --enable-thinking \
  --run-name "stage1-fixed-template-thinking-object-fold-${FOLD}"
```

Evaluate the matching held-out fold:

```bash
FOLD=0
bash spatial_internalization/training/run_eval_docker.sh 0 \
  --variant fixed_template_reasoning \
  --adapter "spatial_internalization/checkpoints/stage1_object_cv_thinking/fold_${FOLD}/final" \
  --data-jsonl "spatial_internalization/datasets/stage1_object_cv/fold_${FOLD}/fixed_template_reasoning/test.jsonl" \
  --output-jsonl "spatial_internalization/results/stage1_object_cv_thinking/fold_${FOLD}/test_predictions.jsonl" \
  --enable-thinking \
  --max-new-tokens 32768
```

## Stage 2 data

The historical source is
`comfort_addionalprompt_tests/test07_camera_geometry_before_question/results_preciseprompt/mcq_long_qwen3_5vl_thinking.csv`.
Rows must have a closed trajectory, a correct final answer, a relation matching
Blender GT, and prompt geometry matching Blender GT.

The formal provider is one frozen Qwen3.5-397B-A17B accessed through an
OpenAI-compatible Chat Completions endpoint. It is used first as consolidator
and then as verbalizer. The student target is the verbalized reasoning plus
the final answer letter.

Run the API smoke test:

```bash
export OPENAI_API_KEY='...'
export OPENAI_BASE_URL='https://api-2.xi-ai.cn/v1'
bash spatial_internalization/stage2_trajectory_guided/run_api_smoke.sh
```

Run a pilot in a new output directory:

```bash
python -m spatial_internalization.stage2_trajectory_guided.build_dataset \
  --stage all \
  --limit-families 8 \
  --output-root spatial_internalization/datasets/stage2_trajectory_guided/pilot_new \
  --clean-evidence-jsonl spatial_internalization/datasets/stage2_trajectory_guided/pilot_new/consolidated_evidence.jsonl \
  --api-base-url 'https://api-2.xi-ai.cn/v1' \
  --api-model 'Qwen3.5-397B-A17B'
```

Train Stage 2 after full verbalization exists:

```bash
python -m spatial_internalization.training.train_sft \
  --variant verbalized_reasoning \
  --data-root spatial_internalization/datasets/stage2_trajectory_guided/full \
  --model-path /path/to/Qwen3.5-9B \
  --output-dir spatial_internalization/checkpoints/stage2_trajectory_guided
```

## Tests

```bash
python -m unittest discover -s spatial_internalization/tests -v
```
