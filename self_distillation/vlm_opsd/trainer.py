"""On-policy full-vocabulary OPSD trainer for Qwen3.5-VL."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from accelerate.utils import is_peft_model
from transformers import GenerationConfig, Trainer


def completion_mask(
    completion_ids: torch.Tensor,
    eos_token_id: int | Sequence[int],
) -> torch.Tensor:
    """Mask through the first EOS, including EOS itself."""
    eos_ids = [eos_token_id] if isinstance(eos_token_id, int) else list(eos_token_id)
    eos = torch.zeros_like(completion_ids, dtype=torch.bool)
    for token_id in eos_ids:
        eos |= completion_ids.eq(token_id)
    before_first_eos = eos.cumsum(dim=1).eq(0)
    return (before_first_eos | eos & eos.cumsum(dim=1).eq(1)).long()

def forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
    vocabulary_entry_clip: float | None,
    return_metrics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """OPSD beta=0: full-vocabulary KL(teacher || student), completion only.

    Upstream clamps vocabulary entries before summing, despite naming the
    option jsd_token_clip. No temperature-squared multiplier is applied.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.detach().float() / temperature, dim=-1)
    entries = F.kl_div(
        student_log_probs, teacher_log_probs, reduction="none", log_target=True
    )
    valid = mask.bool()
    entries = entries[valid]
    token_count = valid.sum().clamp_min(1)
    clipped = entries if vocabulary_entry_clip is None else entries.clamp(max=vocabulary_entry_clip)
    loss = clipped.sum() / token_count
    if not return_metrics:
        return loss
    with torch.no_grad():
        clip_ratio = (
            (entries > vocabulary_entry_clip).float().sum()
            / (token_count * student_logits.shape[-1])
            if vocabulary_entry_clip is not None else loss.new_zeros(())
        )
        metrics = {
            "raw_kl": entries.sum().detach() / token_count,
            "clipped_kl": loss.detach(),
            "clip_ratio": clip_ratio,
        }
    return loss, metrics


class VLMOPSDTrainer(Trainer):
    """Generate student completions online, then align teacher/student token distributions."""

    def __init__(
        self,
        *args,
        max_completion_length: int,
        temperature: float,
        top_p: float,
        top_k: int,
        vocabulary_entry_clip: float | None,
        debug_diagnostics: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.vocabulary_entry_clip = vocabulary_entry_clip
        self.debug_diagnostics = debug_diagnostics
        self._metric_batches: list[dict[str, float]] = []
        self._diagnosed_forward = False
        self._diagnosed_backward = False
        tokenizer = self.processing_class.tokenizer
        eos_token_id = self.model.generation_config.eos_token_id
        self.generation_config = GenerationConfig(
            max_new_tokens=max_completion_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_token_id,
            use_cache=True,
        )

    def _is_main_process(self) -> bool:
        return not hasattr(self, "accelerator") or self.accelerator.is_main_process

    @staticmethod
    def _cuda_memory() -> str:
        if not torch.cuda.is_available():
            return "cuda=unavailable"
        gib = 1024**3
        return (
            f"allocated={torch.cuda.memory_allocated() / gib:.2f}GiB, "
            f"reserved={torch.cuda.memory_reserved() / gib:.2f}GiB, "
            f"peak={torch.cuda.max_memory_allocated() / gib:.2f}GiB"
        )

    def _debug_print(self, message: str) -> None:
        if self.debug_diagnostics and self._is_main_process():
            print(f"[OPSD diagnostic] {message}", flush=True)

    def _prefix_kwargs(self, inputs: dict, prefix: str) -> dict:
        marker = prefix + "_"
        return {
            key[len(marker) :]: value
            for key, value in inputs.items()
            if key.startswith(marker)
        }

    @staticmethod
    def _append_completion(
        prefix: dict,
        completion_ids: torch.Tensor,
        completion_attention: torch.Tensor,
    ) -> dict:
        result = {
            "input_ids": torch.cat([prefix["input_ids"], completion_ids], dim=1),
            "attention_mask": torch.cat(
                [prefix["attention_mask"], completion_attention], dim=1
            ),
            "pixel_values": prefix["pixel_values"],
            "image_grid_thw": prefix["image_grid_thw"],
        }
        if "mm_token_type_ids" in prefix:
            completion_types = torch.zeros_like(completion_ids)
            result["mm_token_type_ids"] = torch.cat(
                [prefix["mm_token_type_ids"], completion_types], dim=1
            )
        return result

    def _fixed_teacher_context(self, model):
        unwrapped = self.accelerator.unwrap_model(model)
        return unwrapped.disable_adapter() if is_peft_model(unwrapped) else nullcontext()

    def log(self, logs, *args, **kwargs):
        """Add KL diagnostics to the Trainer's normal console/W&B logs."""
        logs = dict(logs)
        batches = getattr(self, "_metric_batches", [])
        if batches and "loss" in logs:
            names = sorted(batches[0])
            values = torch.tensor(
                [sum(batch[name] for batch in batches) / len(batches) for name in names],
                device=self.accelerator.device, dtype=torch.float32,
            )
            # Average over accumulation micro-batches and ranks, not only the last batch.
            values = self.accelerator.reduce(values, reduction="mean")
            logs.update(zip(names, values.cpu().tolist()))
            self._metric_batches = []
            if torch.cuda.is_available():
                peak = torch.tensor([torch.cuda.max_memory_allocated() / 1024**3],
                                    device=self.accelerator.device)
                logs["gpu_peak_allocated_gib"] = self.accelerator.gather(peak).max().item()
        return super().log(logs, *args, **kwargs)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Evaluate through OPSD loss instead of the model's plain forward path."""
        # The collator returns student_* and teacher_* fields, not a plain
        # input_ids/labels pair. Trainer's default no-label path calls
        # model(**inputs), which cannot handle those prefixed fields.
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss, _ = self.compute_loss(model, inputs, return_outputs=True)
        return loss.detach().mean(), None, None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        student_prefix = self._prefix_kwargs(inputs, "student")
        teacher_prefix = self._prefix_kwargs(inputs, "teacher")
        student_prompt_length = student_prefix["input_ids"].shape[1]
        teacher_prompt_length = teacher_prefix["input_ids"].shape[1]

        diagnose = self.debug_diagnostics and not self._diagnosed_forward
        if diagnose and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        if diagnose:
            tokenizer = self.processing_class.tokenizer
            prompt_tail_ids = student_prefix["input_ids"][0, -128:].detach().cpu()
            prompt_tail = tokenizer.decode(prompt_tail_ids, skip_special_tokens=False)
            self._debug_print(
                f"sample_ids={inputs.get('sample_ids')}, "
                f"student_prompt_shape={tuple(student_prefix['input_ids'].shape)}, "
                f"teacher_prompt_shape={tuple(teacher_prefix['input_ids'].shape)}"
            )
            self._debug_print(f"student_prompt_tail={prompt_tail!r}")
            self._debug_print(
                f"training_args.gradient_checkpointing="
                f"{self.args.gradient_checkpointing}, "
                f"model.is_gradient_checkpointing="
                f"{getattr(self.accelerator.unwrap_model(model), 'is_gradient_checkpointing', None)}, "
                f"model.training={model.training}"
            )
            self._debug_print(
                f"generation_config.max_new_tokens="
                f"{self.generation_config.max_new_tokens}, "
                f"eos_token_id={self.generation_config.eos_token_id}, "
                f"do_sample={self.generation_config.do_sample}, "
                f"temperature={self.generation_config.temperature}"
            )

        unwrapped = self.accelerator.unwrap_model(model)
        original_use_cache = unwrapped.config.use_cache
        was_training = unwrapped.training
        unwrapped.config.use_cache = True
        # Gradient checkpointing only applies in training mode. Generation is
        # already inference-only, so temporarily use eval mode to retain the
        # KV cache without Transformers issuing an incompatibility warning.
        unwrapped.eval()
        try:
            with torch.no_grad():
                generated = unwrapped.generate( #student generate answer online: use sampling=True
                    **student_prefix,
                    generation_config=self.generation_config,
                )
        finally:
            unwrapped.config.use_cache = original_use_cache
            unwrapped.train(was_training)

        completion_ids = generated[:, student_prompt_length:]
        completion_attention = completion_mask(
            completion_ids, self.generation_config.eos_token_id
        )
        if diagnose:
            tokenizer = self.processing_class.tokenizer
            valid_lengths = completion_attention.sum(dim=1).detach().cpu().tolist()
            eos_ids = self.generation_config.eos_token_id
            eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids)
            eos_seen = [
                any(token_id in eos_ids for token_id in row[:valid_length].tolist())
                for row, valid_length in zip(completion_ids.detach().cpu(), valid_lengths)
            ]
            preview_ids = completion_ids[0, : min(256, completion_ids.shape[1])]
            completion_preview = tokenizer.decode(
                preview_ids.detach().cpu(), skip_special_tokens=False
            )
            tail_ids = completion_ids[0, -min(256, completion_ids.shape[1]) :]
            completion_tail = tokenizer.decode(
                tail_ids.detach().cpu(), skip_special_tokens=False
            )
            first_eos_positions = []
            for row in completion_ids.detach().cpu():
                eos_positions = [
                    index
                    for index, token_id in enumerate(row.tolist())
                    if token_id in eos_ids
                ]
                first_eos_positions.append(eos_positions[0] if eos_positions else None)
            self._debug_print(
                f"generated_shape={tuple(generated.shape)}, "
                f"completion_tensor_shape={tuple(completion_ids.shape)}, "
                f"valid_completion_lengths={valid_lengths}, eos_seen={eos_seen}, "
                f"first_eos_positions={first_eos_positions}, "
                f"hit_max_new_tokens="
                f"{completion_ids.shape[1] == self.generation_config.max_new_tokens}"
            )
            self._debug_print(f"completion_preview={completion_preview!r}")
            self._debug_print(f"completion_tail={completion_tail!r}")
            self._debug_print(
                f"completion_last_token_ids="
                f"{completion_ids[0, -min(16, completion_ids.shape[1]) :].tolist()}"
            )
            self._debug_print(f"after_generation_memory: {self._cuda_memory()}")
        student_inputs = self._append_completion(
            student_prefix, completion_ids, completion_attention
        )
        teacher_inputs = self._append_completion(
            teacher_prefix, completion_ids, completion_attention
        )
        # Match upstream OPSD: ordinary model forwards, then causal slicing.
        # No lm_head interception, hidden-state projection, or token chunks.
        with torch.no_grad(), self._fixed_teacher_context(model):
            teacher_outputs = model(**teacher_inputs, use_cache=False)
            teacher_logits = teacher_outputs.logits[:, teacher_prompt_length - 1 : -1, :]
        student_outputs = model(**student_inputs, use_cache=False)
        student_logits = student_outputs.logits[:, student_prompt_length - 1 : -1, :]
        if student_logits.shape[:2] != completion_ids.shape or teacher_logits.shape[:2] != completion_ids.shape:
            raise RuntimeError("Teacher/student completion logits are misaligned")
        if diagnose:
            self._debug_print(
                f"student_logits_shape={tuple(student_logits.shape)}, "
                f"teacher_logits_shape={tuple(teacher_logits.shape)}"
            )
        loss, loss_metrics = forward_kl(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            mask=completion_attention,
            temperature=self.temperature,
            vocabulary_entry_clip=self.vocabulary_entry_clip,
            return_metrics=True,
        )
        if model.training:
            batch_metrics = {
                name: value.detach().float().item() for name, value in loss_metrics.items()
            }
            lengths = completion_attention.sum(dim=1)
            eos_ids = self.generation_config.eos_token_id
            eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids)
            ended = torch.zeros_like(lengths, dtype=torch.bool)
            for eos_id in eos_ids:
                ended |= completion_ids.eq(eos_id).any(dim=1)
            batch_metrics.update(
                completion_length_mean=lengths.float().mean().item(),
                completion_no_eos_ratio=(~ended).float().mean().item(),
                student_prompt_length_mean=student_prefix["attention_mask"].sum(1).float().mean().item(),
                teacher_prompt_length_mean=teacher_prefix["attention_mask"].sum(1).float().mean().item(),
            )
            if not hasattr(self, "_metric_batches"):
                self._metric_batches = []
            self._metric_batches.append(batch_metrics)
        if diagnose:
            self._debug_print(
                f"loss={loss.detach().float().item():.8g}, "
                f"loss_requires_grad={loss.requires_grad}, "
                f"loss_grad_fn={type(loss.grad_fn).__name__ if loss.grad_fn else None}, "
                f"valid_token_count={int(completion_attention.sum().item())}"
            )
            self._debug_print(f"before_backward_memory: {self._cuda_memory()}")
            self._diagnosed_forward = True

        if return_outputs:
            return loss, {"loss": loss.detach()}
        return loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        if (
            self.debug_diagnostics
            and not self._diagnosed_backward
            and self._is_main_process()
        ):
            trainable = 0
            with_grad = 0
            finite_grad = 0
            grad_sq_sum = torch.zeros((), device=loss.device, dtype=torch.float32)
            first_grad_names = []
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                trainable += 1
                if parameter.grad is None:
                    continue
                with_grad += 1
                if torch.isfinite(parameter.grad).all():
                    finite_grad += 1
                grad_sq_sum += parameter.grad.detach().float().square().sum()
                if len(first_grad_names) < 5:
                    first_grad_names.append(name)
            self._debug_print(
                f"after_backward: trainable_parameter_tensors={trainable}, "
                f"with_grad={with_grad}, finite_grad={finite_grad}, "
                f"raw_grad_l2={grad_sq_sum.sqrt().item():.8g}, "
                f"first_grad_names={first_grad_names}"
            )
            self._debug_print(f"after_backward_memory: {self._cuda_memory()}")
            self._diagnosed_backward = True
        return loss
