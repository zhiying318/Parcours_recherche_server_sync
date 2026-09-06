# Geometry-verified self-distillation data

`generate_geometry_data.py` joins the test07 prompt metadata, Qwen3.5-9B thinking
CSV, and COMFORT `scene_gt.json` files. Only the 113 rows whose predicted letter
equals the ground-truth letter are eligible for distillation.

The split unit is `object_name`, not an image. This keeps every view, relation,
and noisy copy of one object category in exactly one split. Only the training
split is augmented; `--train-augmentations N` means N additional noisy records
for every clean training record. Person and object camera positions receive independent
Gaussian noise on each coordinate. The noisy geometry is rounded to three decimal
places before classification and teacher prompting. A Gaussian
yaw perturbation is applied around
the person's unchanged upright axis to both person-forward and person-right.
This keeps the noisy right direction recoverable from the noisy forward direction
and the upright-person convention, while preserving the unit-length and
orthogonality assumptions required by the projection rule.
Because the source image remains unchanged, augmentation uses label-preserving
rejection sampling: a noisy draw is accepted only when the recomputed relation
matches the clean relation. `noise_attempts` records how many draws were needed.

The student always receives the image and the baseline MCQ beginning with
`Where is ...`; it never receives coordinates. The fixed teacher receives the
same image and exactly the same MCQ, followed by a reference-solution block that
contains only `teacher_geometry`. No stored clean or noisy CoT is used.

Clean records retain the CSV's original A-D order. The eight noisy copies use a
deterministic balanced shuffle, placing the correct relation twice at each letter
position while updating the MCQ's correct letter. With the defaults, the
training set contains 74 clean records and 592 noisy records (666 total). The
clean validation and test sets contain 15 and 24 records respectively.
The Trainer's seeded random sampler mixes clean and noisy records during every
training epoch; they are not run as two consecutive training phases.

Generate the final JSONL directly; no separate CoT generation stage is needed:

```bash
python -m self_distillation.generate_geometry_data
```

Run the CPU-only tests:

```bash
python -m unittest discover -s self_distillation/tests -v
```

The Qwen3.5-VL OPSD implementation and launch instructions are in
[`vlm_opsd/DESIGN.md`](vlm_opsd/DESIGN.md). Training uses
`Qwen/Qwen3.5-9B` in non-thinking mode by default, multimodal Transformers rollout, a fixed base
teacher, and language-decoder LoRA. Pass the physical GPU indices explicitly:

```bash
bash self_distillation/vlm_opsd/prepare_docker_env.sh
bash self_distillation/vlm_opsd/run_docker.sh 0,1 \
  --student_thinking False --teacher_thinking False \
  --max-completion-length 1024 \
  --output-dir self_distillation/output/qwen3_5_9b_opsd_nonthink
```

The first command prepares a persistent Python package directory for the fixed
Torch 2.6 Docker image. The second command accepts explicit host GPU indices,
passes them through Docker's `--gpus device=...`, and defaults to one accumulation step, so global batch is GPU count × per-device
batch size. `OPSD_GLOBAL_BATCH_SIZE` can request a larger batch through accumulation;
it must be divisible by GPU count × per-device batch size by setting gradient accumulation accordingly.

After training, evaluate the adapter on the held-out object categories:

```bash
CUDA_VISIBLE_DEVICES=0 python -m self_distillation.vlm_opsd.evaluate \
  --adapter self_distillation/output/qwen3_5_9b_opsd_nonthink --no_thinking
```

Both thinking flags accept explicit `True` / `False` values (hyphenated aliases
are also supported) and default to `False`. They control each context's chat
template independently, as in [OPSD](https://github.com/siyan-zhao/OPSD).
The teacher scores the student's generated tokens; it does not generate a separate CoT.
Evaluation also defaults to non-thinking; use `--student_thinking True` for a
thinking checkpoint.

The active trainer uses ordinary full-logit forwards and direct full-vocabulary
KL, with no hidden-state interception or loss chunks. `--loss-chunk-size` and
the previously unused `--max-total-length` are removed. Standard model gradient
checkpointing remains enabled. Non-thinking does not guarantee an early EOS;
`--max-completion-length` remains the generation cap, and full logits still consume memory.
Use a fresh output directory for the new experiment, as above.

The previous thinking/chunked implementation is preserved in
[`archive/vlm_opsd_chunked`](archive/vlm_opsd_chunked/README.md).

## A100 80GB: batch size, epochs, and W&B

The container launcher supports one GPU (plain Python inside Docker) and multiple
GPUs (Accelerate DDP inside the same Docker image). Batch arithmetic is:
`global_batch = GPUs * per_device_batch_size * gradient_accumulation_steps`.
Without `OPSD_GLOBAL_BATCH_SIZE` or explicit accumulation, accumulation is 1.
Conflicting or indivisible settings are rejected. Unset any old
`OPSD_GLOBAL_BATCH_SIZE=2` before using two GPUs with micro-batch 2.

Start with one epoch on the current 666 records. There are only 74 clean training
images, so more epochs should be justified by held-out validation accuracy, not
training KL alone. An initial 1–3 epoch exploration is reasonable; no optimal
epoch count has been measured. Validation is not automatically run by this trainer.
With global batch 4, one epoch is about 167 optimizer steps; three about 501.
With global batch 2, those counts are 333 and 999. Last batches can be partial.

```bash
# Two GPUs, micro-batch 2, global batch 4, one epoch.
unset OPSD_GLOBAL_BATCH_SIZE
bash self_distillation/vlm_opsd/run_docker.sh 0,1 \
  --per-device-batch-size 2 --num-train-epochs 1 \
  --student_thinking False --teacher_thinking False \
  --max-completion-length 1024 --report-to wandb \
  --wandb-project opsd-vlm-nonthink-20260905 \
  --run-name nonthink-2gpu-b2-1epoch \
  --output-dir self_distillation/output/nonthink_2gpu_b2_1epoch

# One GPU with accumulation 2 preserves global batch 4.
OPSD_GLOBAL_BATCH_SIZE=4 bash self_distillation/vlm_opsd/run_docker.sh 0 \
  --per-device-batch-size 2 --num-train-epochs 1 \
  --student_thinking False --teacher_thinking False \
  --max-completion-length 1024 --report-to wandb \
  --wandb-project opsd-vlm-nonthink-20260905 \
  --run-name nonthink-1gpu-b2-acc2-1epoch \
  --output-dir self_distillation/output/nonthink_1gpu_b2_acc2_1epoch
```

`--num-train-epochs` disables the old implicit 100-step cap. An explicitly positive
`--max-steps` overrides epochs. If neither is supplied, the default is still
100 steps. `--save-steps` and `--save-total-limit` default to 25 and 2;
increase the latter if you need earlier checkpoints for comparison.

W&B reporting now defaults to enabled. Export `WANDB_API_KEY` in the host shell;
the launcher passes it and optional `WANDB_ENTITY` into Docker. The new default
project is `opsd-vlm-nonthink-20260905`; override with `--wandb-project`.
The old `opsd-vlm` project is rejected, and old project/run/resume environment
variables are not forwarded. Every invocation initializes a fresh run ID.
The remote project/run is created when training starts, not during code setup.
`WANDB_MODE=offline` still means local-only logging; use online mode to upload.

Config includes thinking flags, sampling settings, completion cap, KL clipping,
LoRA parameters, dataset size, trainable parameter count, GPU count, actual global
batch, epochs/steps, seed and learning rate. It is also saved to
`experiment_config.json` in the output directory. Logged metrics include Trainer
loss, learning rate, gradient norm and epoch; raw/clipped KL and clip ratio;
mean completion/prompt lengths; no-EOS fraction; and maximum allocated CUDA peak
across ranks. Custom mean metrics average micro-batches and ranks. W&B also
collects system utilization. Accuracy is not recorded unless an evaluation is run.
Use `--report-to none` for an explicitly untracked run.
