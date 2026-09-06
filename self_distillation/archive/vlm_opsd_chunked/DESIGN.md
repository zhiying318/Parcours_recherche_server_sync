# Why OPSD needs a Qwen3.5-VL adaptation

## Model and thinking mode

This experiment uses `Qwen/Qwen3.5-9B`, the same multimodal checkpoint that
produced the successful test07 traces. Its chat template is called with
`enable_thinking=True`.

## Text OPSD versus this VLM implementation

The public OPSD code uses `AutoTokenizer`. Its collator produces only
`input_ids` and `attention_mask`, and its vLLM rollout decodes each prompt to a
plain string. That is sufficient for Qwen3, but a plain string contains no
image features.

This implementation uses `AutoProcessor.apply_chat_template` for both
contexts:

- student: image + the baseline MCQ, with no coordinates;
- teacher: the same image and MCQ + a reference-solution block containing only
  the clean/noisy test07 geometry.

The processor expands the image placeholder to the correct number of patch
tokens and returns `pixel_values`, `image_grid_thw`, and, on versions that
require it, `mm_token_type_ids`. Those tensors are passed to student rollout,
student scoring, and teacher scoring. The first implementation uses
Transformers generation because the public OPSD vLLM call accepts only text;
using it unchanged would silently remove the image from the on-policy rollout.

## What remains unchanged from OPSD

The student generates its own thinking trajectory online. The identical token
IDs are appended to the student prompt and privileged teacher prompt. The loss
is full-vocabulary forward KL, `KL(teacher || student)`, only on completion
positions. Vocabulary-entry clipping is applied before summing over the
vocabulary, matching OPSD's stabilization mechanism.

Only one 9B base model is loaded. Language-decoder LoRA is active for student
generation and student scoring. During teacher scoring, `disable_adapter()`
temporarily exposes the frozen step-0 base model. This preserves OPSD's fixed
self-teacher instead of allocating a second 9B model.

LoRA is restricted to language-model attention and MLP projections. Qwen3.5
mixes full-attention layers (`q/k/v/o_proj`) with linear-attention layers
(`in_proj_qkv`, `in_proj_z`, `in_proj_a/b`, and `out_proj`), so both types are
targeted. The older Qwen3-VL-only regex would omit most Qwen3.5 attention
layers. The vision
encoder is frozen because the target is geometry-conditioned chain-of-thought
reasoning, and the same rendered images were already understood by the base
VLM. This also prevents generic projection names from accidentally attaching
LoRA modules inside the visual tower.

Position coordinates receive independent Gaussian noise. Orientation noise is
a Gaussian yaw around the person's unchanged upright axis; both forward and
right rotate together. All four noisy vectors are rounded to three decimal places
before relation checking and teacher prompting. An arbitrary 3-D rotation is deliberately not used,
because it would introduce a hidden roll/pitch that cannot be recovered from
the student-visible forward vector and unchanged image.

## Sequence alignment

Student and teacher prompts have different lengths. Each is left-padded within
its own batch, then the exact same completion IDs are appended. Completion
token `n` is scored from logit position `prompt_length - 1 + n` on both sides.
Qwen3.5-VL's `logits_to_keep` limits the language head to those positions, so
prompt tokens do not allocate unused full-vocabulary logits.

## GPU launch rule

`run_docker.sh` accepts explicit host GPU indices and passes the same IDs through
Docker's `--gpus device=...`. One, two, four,
or eight selected cards keep global batch size 32 by changing gradient
accumulation.
