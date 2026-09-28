# Spatial capability-internalization SFT

This directory is an isolated experiment. It does not import the existing
`self_distillation`/OPSD data, collator, trainer, JSONL, teacher output,
adapter, or checkpoint.

## Source audit

The default source is the original Blender export at
`COMFORT/data/comfort_human_car_geometry_gt/comfort_human_car`. Each sample has
`scene_gt.json` plus `0.png`. The export contract is v1.0 and uses the
component-wise median of valid masked depth points for
`human_visible_center_camera` and `object_visible_center_camera`; this is not a
foot, pelvis, or model-origin substitution. `human_frame_camera.forward_axis`
and `right_axis` are read directly. The camera convention is +X image-right,
+Y downward, +Z into the scene.

The raw file uses `relation: "back"` for the semantic class that the task calls
`behind`; `configured_relation` is the directory-style value such as
`"behind"`. This implementation records both and verifies the computed class
against the canonicalized independent raw `relation` field. It does not use the
computed class as its own GT.

The generated audit currently reports 144 valid, 0 excluded, 1 near-boundary
sample, and 36 object-relation view groups. The source contract does not expose
an independent scene or layout ID; each group is constructed from one
`relation/object` directory stem and keeps its four camera views together.
Therefore this split is not evidence of unseen-scene generalization. The
deterministic stratified split is 28/4/4 object-relation view groups for
train/validation/test. No real-image intermediate geometry label set is
provided; evaluation on a real set therefore supports answer accuracy only
unless the set supplies compatible geometry fields.

## Modules and artifacts

- `geometry.py`: raw loading, axis checks, projection rule, common positive
  depth scaling, and adaptive 3–6 digit quantization.
- `data_pipeline.py`: object-relation view-group split, deterministic baseline-style A–D
  option randomization, local Gaussian position/axis noise, rejection of
  label-changing noise, and example construction.
- `verbalizer.py`: fixed, sample-dependent geometry template.
- `sft_collator.py`: Qwen processor chat-template serialization and
  assistant-only labels; JSONL keeps `question` and `choices` separate, then
  serializes them once into the user input.
- `train.py`: ordinary token-level causal SFT with language-decoder LoRA.
- `run_docker.sh`: independent single-GPU Docker and W&B entry point.
- `evaluate.py`: free generation, strict final-letter parsing, confusion
  matrix, invalid/truncation/output-length metrics, and B intermediate geometry
  metrics when the template can be parsed.
- `chat_template_check.py`: inspect the installed model tokenizer/template.
- `tests/`: CPU tests for directions, ties, scaling, signs, quantization flips,
  invalid GT, group leakage, and loss masks.

Generated data is stored in `data/answer_only/` and
`data/geometry_reasoning/`, with shared IDs and splits. `audit.json` retains
raw/full-precision evidence, displayed evidence, source relation provenance,
noise provenance, and boundary metadata. `checkpoints/` and
`evaluation_results/` are reserved for later runs.

For the first clean generated basketball sample, the user message is:

```text
Where is the basketball in the perspective of the person?
Choose ONE option and respond with ONLY the letter.
A. From the person's perspective, the basketball is in front of them.
B. From the person's perspective, the basketball is behind them.
C. From the person's perspective, the basketball is on their right.
D. From the person's perspective, the basketball is on their left.
```

Its complete geometry-reasoning assistant target is:

```text
Consider that the picture was taken from the origin [0.000, 0.000, 0.000] of a camera coordinate system, where +X points to the image's right, +Y points downward, and +Z points forward into the scene.

The person's estimated position is [0.003, -0.148, 1.000], and the basketball's estimated position is [0.000, 0.025, 0.781].
The person-to-object displacement is therefore [-0.003, 0.173, -0.219].

The person's estimated forward axis is [0.000, -0.196, 0.981], and their right axis is [1.000, 0.000, 0.000].

The displacement projects approximately -0.249 onto the forward axis and -0.003 onto the right axis. The forward component has the larger absolute magnitude. The forward component has the negative sign, which places the basketball behind the person.
B
```

The last line is the independently generated answer letter. The current
Qwen3.5 configuration uses `enable_thinking=False` and a plain assistant
target, so no manual `<think>` wrapper is added or duplicated. Training and
evaluation use the same policy.

The comparison run above is intentionally non-thinking. To train the same
geometry target as Qwen reasoning content, pass `--enable-thinking`; the
collator places all lines before the final answer in `reasoning_content` inside
the template's `<think>` block and keeps the final A/B/C/D letter outside it.
Evaluation of that adapter must use the same flag.

## Commands

### Object-held-out cross-validation

Generate all nine folds manually (CPU only; no training is started):

```bash
python -m spatial_internalization.generate_object_folds \
  --output-root spatial_internalization/data_object_cv \
  --seed 20260919 \
  --augmentations 8
```

The seeded object order is fixed across folds. Fold `i` holds object `i` out
for test and object `(i + 1) % 9` out for validation; the other seven objects
are used for training. Every relation, camera view and augmentation of an
object stays in its assigned split within that fold. Each object is tested
exactly once over nine separately initialized training runs. Fold numbers are
zero-based (`0` through `8`); pass `--fold 0` to generate only one fold.
Existing fold directories are rejected to prevent accidental replacement.
The original `data/` is unchanged. Each fold's `manifest.json` lists the actual
objects and record counts. Coordinate augmentation uses the existing pipeline:
112 original training images plus eight perturbed targets per image (1008
records per variant); validation and test contain 16 original records each.
This command regenerates targets from source GT, rather than moving augmented
rows from the old split. It does not introduce a new noise ablation.

Select a fold through the existing training `--data-root` argument. The path
must point to `fold_N`, which contains the variant directories:

```bash
FOLD=0
bash spatial_internalization/run_docker.sh 5 \
  --data-root "spatial_internalization/data_object_cv/fold_${FOLD}" \
  --variant geometry_reasoning \
  --model-name Qwen/Qwen3.5-9B \
  --output-dir "spatial_internalization/checkpoints/geometry_reasoning_thinking_object_cv/fold_${FOLD}" \
  --num-train-epochs 5 \
  --per-device-batch-size 2 \
  --gradient-accumulation-steps 1 \
  --learning-rate 5e-6 \
  --enable-thinking \
  --run-name "geometry-reasoning-thinking-object-fold-${FOLD}"

bash spatial_internalization/evaluate_docker.sh 5 \
  --variant geometry_reasoning \
  --model-name Qwen/Qwen3.5-9B \
  --adapter "spatial_internalization/checkpoints/geometry_reasoning_thinking_object_cv/fold_${FOLD}/final" \
  --data-jsonl "spatial_internalization/data_object_cv/fold_${FOLD}/geometry_reasoning/test.jsonl" \
  --output-jsonl "spatial_internalization/evaluation_results/object_cv_fold_${FOLD}.jsonl" \
  --enable-thinking
```

For the full cross-evaluation, run both commands in a shell loop
`for FOLD in {0..8}; do ...; done`, stopping on failures (`set -e`). Each run
starts from the base model and selects its checkpoint using only that fold's
validation data. Always supply `--data-jsonl` for evaluation; its default is
the original split. Report all nine held-out accuracies and their macro mean;
do not select the best test fold or tune hyperparameters against these scores.

Regenerate the independent data and audit:

```bash
python -m spatial_internalization.generate_data
```

Run all CPU checks:

```bash
python -m unittest discover -s spatial_internalization/tests -v
```

Inspect the Qwen tokenizer/chat template before a GPU run:

```bash
python -m spatial_internalization.chat_template_check \
  --model-name Qwen/Qwen3.5-9B
```

Train A and B separately from the base model. These commands are provided but
were not started in this implementation task; both use the same seed, data
order, LoRA defaults, batch settings, and validation-based best-checkpoint
selection.

```bash
python -m spatial_internalization.train \
  --variant answer_only \
  --model-name Qwen/Qwen3.5-9B \
  --output-dir spatial_internalization/checkpoints/answer_only

python -m spatial_internalization.train \
  --variant geometry_reasoning \
  --model-name Qwen/Qwen3.5-9B \
  --output-dir spatial_internalization/checkpoints/geometry_reasoning

python -m spatial_internalization.train \
  --variant geometry_reasoning \
  --model-name Qwen/Qwen3.5-9B \
  --output-dir spatial_internalization/checkpoints/geometry_reasoning_thinking \
  --num-train-epochs 5 \
  --enable-thinking
```

The current defaults are: 3 epochs, per-device micro-batch 1, gradient
accumulation 4 (effective batch 4 on one GPU), learning rate `5e-6`, bf16,
gradient checkpointing, validation every 100 steps, and LoRA `r=64`,
`alpha=128`, dropout 0.0. The base Qwen3.5-9B weights are loaded directly;
an existing adapter is never loaded. LoRA is applied to language-decoder
attention, linear-attention, and MLP projections matched by
`LANGUAGE_LORA_TARGETS`; the vision tower is not targeted.
No sequence truncation is enabled by default. If `--max-length` is supplied,
the collator raises an error when the serialized target would exceed it, so
the final answer cannot be silently removed.

For an A100-80GB, start with micro-batch 1/effective batch 4. After a short
GPU smoke test, micro-batch 2/effective batch 4 or 8 can be tried if the actual
image-token count leaves enough memory. The geometry-reasoning target is much
longer than answer-only, so its safe micro-batch may be smaller.

The Docker entry point uses one explicitly selected GPU and enables W&B
reporting. It assumes the selected image already contains the training
dependencies listed in `requirements-docker.txt`. Launch one A100 (for
example GPU 0):

```bash
export WANDB_API_KEY=...
export WANDB_PROJECT=spatial-internalization
export WANDB_MODE=online

bash spatial_internalization/run_docker.sh 0 \
  --variant geometry_reasoning \
  --model-name Qwen/Qwen3.5-9B \
  --output-dir spatial_internalization/checkpoints/geometry_reasoning \
  --num-train-epochs 3 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --run-name geometry-reasoning-a100
```

Use `--variant answer_only` and a different `--output-dir`/`--run-name` for
the A run. `WANDB_MODE=offline` is available when only local W&B files are
desired. The Docker scripts mount only this repository, a Hugging Face cache,
and a user-owned Python package cache; they do not modify the OPSD environment.

Evaluate only after a `final/` adapter exists; test is not used for checkpoint
selection:

```bash
python -m spatial_internalization.evaluate \
  --variant answer_only \
  --model-name Qwen/Qwen3.5-9B \
  --adapter spatial_internalization/checkpoints/answer_only/final \
  --output-jsonl spatial_internalization/evaluation_results/answer_only_test.jsonl

python -m spatial_internalization.evaluate \
  --variant geometry_reasoning \
  --model-name Qwen/Qwen3.5-9B \
  --adapter spatial_internalization/checkpoints/geometry_reasoning/final \
  --output-jsonl spatial_internalization/evaluation_results/geometry_reasoning_test.jsonl \
  --max-new-tokens 32768

python -m spatial_internalization.evaluate \
  --variant geometry_reasoning \
  --model-name Qwen/Qwen3.5-9B \
  --adapter spatial_internalization/checkpoints/geometry_reasoning_thinking/final \
  --output-jsonl spatial_internalization/evaluation_results/geometry_reasoning_thinking_test.jsonl \
  --max-new-tokens 32768 \
  --enable-thinking
```

The same evaluation can be launched through the independent Docker entry point
(use this when running from the host):

```bash
bash spatial_internalization/evaluate_docker.sh 0 \
  --variant geometry_reasoning \
  --model-name Qwen/Qwen3.5-9B \
  --adapter spatial_internalization/checkpoints/geometry_reasoning/final \
  --output-jsonl spatial_internalization/evaluation_results/geometry_reasoning_test.jsonl

bash spatial_internalization/evaluate_docker.sh 0 \
  --variant geometry_reasoning \
  --model-name Qwen/Qwen3.5-9B \
  --adapter spatial_internalization/checkpoints/geometry_reasoning_thinking/final \
  --output-jsonl spatial_internalization/evaluation_results/geometry_reasoning_thinking_test.jsonl \
  --max-new-tokens 32768 \
  --enable-thinking
```

The evaluator writes the complete decoded generation to `raw_response` in the
JSONL file, one row per test sample, and writes aggregate metrics to the same
path with `.summary.json` replacing `.jsonl`. It also stores parsing failures,
invalid-answer and truncation flags. It does not store logits or token-level
log probabilities.

For a real-image adapter, pass `--data-jsonl` containing at least
`image_path`, `question`, and `choices`; include `answer_letter` for supported accuracy and
`geometry` fields for intermediate geometry metrics. The evaluator never adds
the correct relation, geometry, or a reasoning prefix to the user prompt.

## Version 2: trajectory-guided consolidation and verbalization

`generate_trajectory_data.py` implements the second experiment. Its only raw
trajectory source is:

```text
comfort_addionalprompt_tests/test07_camera_geometry_before_question/results_preciseprompt/mcq_long_qwen3_5vl_thinking.csv
```

Rows are retained only when the trajectory is non-empty and closed, its parsed
answer is correct, the CSV relation agrees with Blender GT, and the three
geometry vectors in the prompt agree with Blender GT. The fixed
object-relation view-group split is reused from `data/manifest.json`. The
source CSV does not contain a
verifiable historical checkpoint size, so its audit record says
`Qwen3.5-VL thinking; size not recorded` rather than inferring 9B.

The formal v2 generator uses one frozen `Qwen3.5-397B-A17B` through the
OpenAI-compatible Chat Completions endpoint for both consolidation and
verbalization. It does not load the first-version adapter. The API client sends
the unchanged image as a data URL and consumes only `message.content`; a
provider `reasoning_content` field is never copied into the student target.
`enable_thinking` is configured separately for the two API stages and recorded
separately from the student's fixed `enable_thinking=False`. API `usage` is
recorded when the provider returns it. Requests are never silently truncated;
an explicit context/length rejection fails the relevant sample/family.

First run a small pilot covering different relations/views:

```bash
export OPENAI_API_KEY='...'
export OPENAI_BASE_URL='https://api-2.xi-ai.cn/v1'
bash spatial_internalization/run_api_smoke.sh

python -m spatial_internalization.generate_trajectory_data \
  --stage all --limit-families 8 \
  --output-root spatial_internalization/data_v2/pilot \
  --clean-evidence-jsonl spatial_internalization/data_v2/pilot/clean_evidence.jsonl \
  --api-base-url 'https://api-2.xi-ai.cn/v1' \
  --api-model 'Qwen3.5-397B-A17B'
```

The smoke entry makes one small vision request, validates the JSON response,
prints provider usage, and writes no data. It retries transient 502/503/504
responses; use `--text-only` to isolate model/account routing from the vision
request. This uses the same OpenAI Python SDK path as `whatsup_eval/run_openai.sh`:
`OPENAI_BASE_URL` ends at `/v1`, and the SDK appends `/chat/completions`. To use
another secret variable, pass `--api-key-env NAME`. The same API client is
reused by both stages in one `--stage all` process. If the provider requires a
known context limit, pass `--api-context-length`; otherwise the manifest
records that the endpoint did not expose one.

To inspect consolidation before verbalization, run:

```bash
python -m spatial_internalization.generate_trajectory_data \
  --stage consolidate --limit-families 8 \
  --output-root spatial_internalization/data_v2/pilot \
  --clean-evidence-jsonl spatial_internalization/data_v2/pilot/clean_evidence.jsonl
```

Each training family uses aug00 plus eight deterministic, label-preserving
geometry perturbations with the first-version seed rule, `position_sigma=0.05`,
and `angle_sigma_degrees=2.0`. The image is unchanged. Perturbed records are
marked `trajectory-derived, geometry-perturbed supervision`, and clean GT is
stored separately. A family is admitted to the paired A/B/C output only when
all required records pass verbalizer validation; failed families remove their
answer-only and template counterparts as well.

The generated variants are `answer_only` (A), `template_reasoning` (B), and
`trajectory_reasoning` (C). Train independently from the same local base:

```bash
BASE=/aistor/hpc_stor03/sjtu_home/zhiying.zou/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a
python -m spatial_internalization.train --variant answer_only \
  --data-root spatial_internalization/data_v2/data --model-path "$BASE" \
  --output-dir spatial_internalization/checkpoints/v2_answer_only
python -m spatial_internalization.train --variant template_reasoning \
  --data-root spatial_internalization/data_v2/data --model-path "$BASE" \
  --output-dir spatial_internalization/checkpoints/v2_template_reasoning
python -m spatial_internalization.train --variant trajectory_reasoning \
  --data-root spatial_internalization/data_v2/data --model-path "$BASE" \
  --output-dir spatial_internalization/checkpoints/v2_trajectory_reasoning
```

`manifest.json` and `audit.json` report source filtering, complete-family
counts, failure reasons, geometry changes and dependencies, raw/clean/target
length distributions, API generation time, actual provider input/output token
counts when available, context length, and generation budgets. For debugging
the old local offline backend, pass `--provider local` and the local
`--consolidator-model-path`/`--verbalizer-model-path`; the formal setting is
the API provider above.
