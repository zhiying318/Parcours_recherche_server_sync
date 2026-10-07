"""Multimodal causal-SFT collator with assistant-only labels."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..stage1_fixed_template.dataset_pipeline import format_user_question


class CapabilitySFTCollator:
    """Apply the model's own chat template once per conversation.

    The prompt is encoded separately only to locate the assistant boundary; no
    prompt tokens, image placeholders, or padding receive a loss.  The full
    conversation is passed to the model so its multimodal image metadata stays
    aligned with the input IDs.
    """

    def __init__(
        self,
        processor,
        repo_root: str | Path,
        max_length: int | None = None,
        enable_thinking: bool = False,
    ):
        self.processor = processor
        self.repo_root = Path(repo_root).resolve()
        self.max_length = max_length
        self.enable_thinking = enable_thinking
        self.track_stats = True
        self.total_target_tokens = 0
        self.total_examples = 0
        tokenizer = getattr(processor, "tokenizer", processor)
        tokenizer.padding_side = "right"

    @staticmethod
    def _messages(
        image_path: str,
        question: str,
        target: str | None = None,
        enable_thinking: bool = False,
    ):
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ],
        }]
        if target is not None:
            if enable_thinking:
                target_lines = target.rstrip().splitlines()
                if not target_lines or not target_lines[-1].strip():
                    raise ValueError("thinking target must end with a non-empty final answer")
                reasoning = "\n".join(target_lines[:-1]).rstrip()
                answer = target_lines[-1].strip()
                messages.append({
                    "role": "assistant",
                    "reasoning_content": reasoning,
                    "content": [{"type": "text", "text": answer}],
                })
            else:
                messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
        return messages

    def _encode(self, conversations, add_generation_prompt: bool):
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
            "return_dict": True,
            "return_tensors": "pt",
            "enable_thinking": self.enable_thinking,
            "processor_kwargs": {"do_resize": False, "padding": True},
        }
        encoded = self.processor.apply_chat_template(conversations, **kwargs)
        if self.max_length is not None:
            lengths = self._lengths(encoded)
            if any(length > self.max_length for length in lengths):
                raise ValueError(
                    f"serialized sequence exceeds --max-length={self.max_length}; "
                    "increase the limit instead of silently truncating the assistant answer"
                )
        return encoded

    @staticmethod
    def _lengths(encoded) -> list[int]:
        mask = encoded.get("attention_mask")
        if mask is None:
            pad_id = getattr(encoded, "pad_token_id", None)
            return [int(row.numel()) for row in encoded["input_ids"]]
        return [int(row.sum().item()) for row in mask]

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        image_paths = [str(self.repo_root / feature["image_path"]) for feature in features]
        prompts = [
            self._messages(
                image,
                format_user_question(feature["question"], feature["choices"]),
                enable_thinking=self.enable_thinking,
            )
            for image, feature in zip(image_paths, features)
        ]
        full = [
            self._messages(
                image,
                format_user_question(feature["question"], feature["choices"]),
                feature["assistant_target"],
                enable_thinking=self.enable_thinking,
            )
            for image, feature in zip(image_paths, features)
        ]
        prompt_encoded = self._encode(prompts, add_generation_prompt=True)
        encoded = self._encode(full, add_generation_prompt=False)
        input_ids = encoded["input_ids"]
        labels = input_ids.clone()
        lengths = self._lengths(encoded)
        prompt_lengths = self._lengths(prompt_encoded)
        if len(lengths) != len(prompt_lengths):
            raise ValueError("processor returned mismatched batch lengths")
        target_token_counts = []
        for row, (full_len, prompt_len) in enumerate(zip(lengths, prompt_lengths)):
            if prompt_len > full_len:
                raise ValueError(f"prompt length {prompt_len} exceeds full length {full_len}")
            prompt_ids = prompt_encoded["input_ids"][row, :prompt_len]
            full_prefix = input_ids[row, :prompt_len]
            if not torch.equal(prompt_ids, full_prefix):
                raise ValueError("chat-template prompt is not a prefix of the full conversation; refusing an unsafe loss mask")
            labels[row, :prompt_len] = -100
            labels[row, full_len:] = -100
            target_count = int((labels[row] != -100).sum().item())
            if target_count <= 0:
                raise ValueError("assistant target produced no loss tokens")
            target_token_counts.append(target_count)
        if "attention_mask" in encoded:
            labels[encoded["attention_mask"] == 0] = -100
        encoded["labels"] = labels
        encoded["sample_ids"] = [feature["id"] for feature in features]
        encoded["target_token_counts"] = target_token_counts
        if self.track_stats:
            self.total_target_tokens += sum(target_token_counts)
            self.total_examples += len(features)
        return encoded
