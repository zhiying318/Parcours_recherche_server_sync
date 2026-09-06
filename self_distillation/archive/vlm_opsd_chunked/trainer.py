"""On-policy full-vocabulary OPSD trainer for Qwen3.5-VL."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import contextmanager, nullcontext
from types import MethodType

import torch
import torch.nn.functional as F
from accelerate.utils import is_peft_model
from torch.utils.checkpoint import checkpoint
from transformers import GenerationConfig, Trainer


@contextmanager
def capture_hidden_without_logits(lm_head):
    """Capture the input of lm_head without materializing [B, L, V] logits."""
    captured = {}
    original_forward = lm_head.forward

    def fake_forward(_module, hidden_states):
        captured["hidden_states"] = hidden_states
        # Qwen forward requires a logits-like return value, but it is not used.
        return hidden_states[..., :1] * 0.0

    lm_head.forward = MethodType(fake_forward, lm_head)
    try:
        yield captured
    finally:
        lm_head.forward = original_forward


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

def forward_kl_from_hidden(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
    lm_head,
    mask: torch.Tensor,
    temperature: float,
    vocabulary_entry_clip: float | None,
    chunk_size: int,
    return_metrics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Exact full-vocabulary KL(teacher || student), chunked over tokens."""

    weight = lm_head.weight
    bias = lm_head.bias
    total = student_hidden.new_zeros((), dtype=torch.float32)
    raw_total = student_hidden.new_zeros((), dtype=torch.float32)
    clipped_total = student_hidden.new_zeros((), dtype=torch.float32)
    clipped_entry_count = student_hidden.new_zeros((), dtype=torch.float32)
    token_count = mask.sum().clamp_min(1).float()

    def chunk_loss(
        student_hidden_chunk,
        teacher_hidden_chunk,
        mask_chunk,
        lm_head_weight,
    ):
        student_logits = F.linear(
            student_hidden_chunk,
            lm_head_weight,
            bias,
        ).float() / temperature

        teacher_logits = F.linear(
            teacher_hidden_chunk,
            lm_head_weight,
            bias,
        ).float() / temperature

        student_log_probs = F.log_softmax(student_logits, dim=-1)
        teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        entries = F.kl_div(
            student_log_probs,
            teacher_log_probs,
            reduction="none",
            log_target=True,
        )

        raw_token_loss = entries.sum(dim=-1)
        raw_sum = (raw_token_loss * mask_chunk).sum()
        if vocabulary_entry_clip is not None:
            clipped_entries = entries.clamp(max=vocabulary_entry_clip)
            clipped_count = (
                (entries > vocabulary_entry_clip).float() * mask_chunk.unsqueeze(-1)
            ).sum()
        else:
            clipped_entries = entries
            clipped_count = entries.new_zeros(())

        clipped_token_loss = clipped_entries.sum(dim=-1)
        clipped_sum = (clipped_token_loss * mask_chunk).sum()
        return (
            (clipped_sum, raw_sum.detach(), clipped_sum.detach(), clipped_count.detach())
            if return_metrics
            else clipped_sum
        )

    for start in range(0, student_hidden.shape[1], chunk_size):
        stop = min(start + chunk_size, student_hidden.shape[1])

        # Checkpoint prevents FP32 logits/log-probs/JSD tensors for every
        # chunk from being retained until the final backward.
        current = checkpoint(
            chunk_loss,
            student_hidden[:, start:stop],
            teacher_hidden[:, start:stop],
            mask[:, start:stop],
            weight,
            use_reentrant=False,
            preserve_rng_state=False,
        )
        if return_metrics:
            current_loss, current_raw, current_clipped, current_clipped_count = current
            raw_total = raw_total + current_raw
            clipped_total = clipped_total + current_clipped
            clipped_entry_count = clipped_entry_count + current_clipped_count
        else:
            current_loss = current
        total = total + current_loss

    loss = total / token_count
    if not return_metrics:
        return loss

    vocabulary_size = student_hidden.new_tensor(
        lm_head.weight.shape[0], dtype=torch.float32
    )
    entry_count = token_count * vocabulary_size
    return loss, {
        "raw_kl": raw_total / token_count,
        "clipped_kl": clipped_total / token_count,
        "clip_ratio": clipped_entry_count / entry_count,
    }

# def forward_kl(
#     student_logits: torch.Tensor,
#     teacher_logits: torch.Tensor,
#     mask: torch.Tensor,
#     temperature: float,
#     vocabulary_entry_clip: float | None,
#     chunk_size: int,
# ) -> torch.Tensor:
#     """KL(teacher || student) over all vocabulary entries and completion tokens."""
#     total = student_logits.new_zeros((), dtype=torch.float32)
#     token_count = mask.sum()
#     for start in range(0, student_logits.shape[1], chunk_size):
#         stop = min(start + chunk_size, student_logits.shape[1])
#         student_log_probs = F.log_softmax(
#             student_logits[:, start:stop].float() / temperature, dim=-1
#         )
#         teacher_log_probs = F.log_softmax(
#             teacher_logits[:, start:stop].float() / temperature, dim=-1
#         )
#         entries = F.kl_div(
#             student_log_probs,
#             teacher_log_probs,
#             reduction="none",
#             log_target=True,
#         )
#         if vocabulary_entry_clip is not None:
#             entries = entries.clamp(max=vocabulary_entry_clip)
#         token_loss = entries.sum(dim=-1)
#         total = total + (token_loss * mask[:, start:stop]).sum()
#     return total / token_count


class VLMOPSDTrainer(Trainer):
    """Generate student CoT online, then align teacher/student token distributions."""

    def __init__(
        self,
        *args,
        max_completion_length: int,
        temperature: float,
        top_p: float,
        top_k: int,
        vocabulary_entry_clip: float | None,
        loss_chunk_size: int,
        debug_diagnostics: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.temperature = temperature
        self.vocabulary_entry_clip = vocabulary_entry_clip
        self.loss_chunk_size = loss_chunk_size
        self.debug_diagnostics = debug_diagnostics
        self._last_loss_metrics: dict[str, float] = {}
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
        if "eval_loss" in logs:
            # compute_loss also runs during evaluation; do not attach the last
            # validation batch's diagnostics as if they were training metrics.
            self._last_loss_metrics = {}
        elif self._last_loss_metrics:
            logs.update(self._last_loss_metrics)
            self._last_loss_metrics = {}
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
        base_model = (
                    unwrapped.get_base_model()
                    if is_peft_model(unwrapped)
                    else unwrapped
                )
        lm_head = base_model.lm_head

        original_use_cache = unwrapped.config.use_cache
        was_training = unwrapped.training
        unwrapped.config.use_cache = True
        # Gradient checkpointing only applies in training mode. Generation is
        # already inference-only, so temporarily use eval mode to retain the
        # KV cache without Transformers issuing an incompatibility warning.
        unwrapped.eval()
        try:
            with torch.no_grad():
                generated = unwrapped.generate(
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
        completion_length = completion_ids.shape[1]
        student_positions = torch.arange(
            student_prompt_length - 1,
            student_prompt_length + completion_length - 1,
            device=completion_ids.device,
        )
        teacher_positions = torch.arange(
            teacher_prompt_length - 1,
            teacher_prompt_length + completion_length - 1,
            device=completion_ids.device,
        )

        # with torch.no_grad(), self._fixed_teacher_context(model):
        #     teacher_logits = model(
        #         **teacher_inputs,
        #         logits_to_keep=teacher_positions,
        #         use_cache=False,
        #     ).logits

        # student_logits = model(
        #     **student_inputs,
        #     logits_to_keep=student_positions,
        #     use_cache=False,
        # ).logits
    
        # loss = forward_kl(
        #     student_logits=student_logits,
        #     teacher_logits=teacher_logits,
        #     mask=completion_attention,
        #     temperature=self.temperature,
        #     vocabulary_entry_clip=self.vocabulary_entry_clip,
        #     chunk_size=self.loss_chunk_size,
        # )
        # return (loss, {"logits": student_logits}) if return_outputs else loss

        with (
            torch.no_grad(),
            self._fixed_teacher_context(model),
            capture_hidden_without_logits(lm_head) as teacher_capture,
        ):
            teacher_outputs = model(
                **teacher_inputs,
                logits_to_keep=teacher_positions,
                use_cache=False,
            )

        teacher_hidden = teacher_capture["hidden_states"].detach()
        del teacher_outputs
        if teacher_hidden.shape[:2] != completion_ids.shape:
            raise RuntimeError(
                "Teacher completion hidden states are misaligned: "
                f"hidden={tuple(teacher_hidden.shape)}, "
                f"completion_ids={tuple(completion_ids.shape)}"
            )
        if diagnose:
            self._debug_print(
                f"teacher_hidden_shape={tuple(teacher_hidden.shape)}, "
                f"dtype={teacher_hidden.dtype}, requires_grad="
                f"{teacher_hidden.requires_grad}"
            )
            self._debug_print(f"after_teacher_memory: {self._cuda_memory()}")

        with capture_hidden_without_logits(lm_head) as student_capture:
            student_outputs = model(
                **student_inputs,
                logits_to_keep=student_positions,
                use_cache=False,
            )

        student_hidden = student_capture["hidden_states"]
        del student_outputs
        if student_hidden.shape[:2] != completion_ids.shape:
            raise RuntimeError(
                "Student completion hidden states are misaligned: "
                f"hidden={tuple(student_hidden.shape)}, "
                f"completion_ids={tuple(completion_ids.shape)}"
            )
        if diagnose:
            self._debug_print(
                f"student_hidden_shape={tuple(student_hidden.shape)}, "
                f"dtype={student_hidden.dtype}, requires_grad="
                f"{student_hidden.requires_grad}, grad_fn="
                f"{type(student_hidden.grad_fn).__name__ if student_hidden.grad_fn else None}"
            )
            self._debug_print(f"after_student_memory: {self._cuda_memory()}")

        loss, loss_metrics = forward_kl_from_hidden(
            student_hidden=student_hidden,
            teacher_hidden=teacher_hidden,
            lm_head=lm_head,
            mask=completion_attention,
            temperature=self.temperature,
            vocabulary_entry_clip=self.vocabulary_entry_clip,
            chunk_size=self.loss_chunk_size,
            return_metrics=True,
        )
        self._last_loss_metrics = {
            name: value.detach().float().item() for name, value in loss_metrics.items()
        }
        if diagnose:
            self._debug_print(
                f"loss={loss.detach().float().item():.8g}, "
                f"loss_requires_grad={loss.requires_grad}, "
                f"loss_grad_fn={type(loss.grad_fn).__name__ if loss.grad_fn else None}, "
                f"loss_chunk_size={self.loss_chunk_size}, "
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
        
