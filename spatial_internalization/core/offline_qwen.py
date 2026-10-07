"""One-process local Qwen3.5-VL generator used by both offline stages."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import time
from typing import Any


DEFAULT_MODEL_ID = "Qwen/Qwen3.5-9B"


def _candidate_model_roots(model_id: str) -> list[Path]:
    if "/" not in model_id:
        raise ValueError(f"expected a Hugging Face model id such as {DEFAULT_MODEL_ID!r}")
    owner, name = model_id.split("/", 1)
    cache_roots = []
    for variable in ("HF_HOME", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(variable)
        if value:
            root = Path(value).expanduser()
            cache_roots.append(root / "hub" if variable == "HF_HOME" else root)
    cache_roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    return [root / f"models--{owner}--{name}" for root in cache_roots]


def _valid_snapshot(path: Path) -> bool:
    return (
        (path / "config.json").is_file()
        and (path / "preprocessor_config.json").is_file()
        and (
            (path / "model.safetensors.index.json").is_file()
            or (path / "model.safetensors").is_file()
        )
    )


def resolve_local_model_path(
    requested: str | Path | None = None,
    model_id: str = DEFAULT_MODEL_ID,
) -> Path:
    """Resolve a complete local snapshot; never silently fall back to Hub."""
    if requested is not None:
        path = Path(requested).expanduser().resolve()
        if not path.is_dir() or not _valid_snapshot(path):
            raise FileNotFoundError(
                f"local model path is not a complete Qwen snapshot: {path}"
            )
        return path

    checked = []
    for model_root in _candidate_model_roots(model_id):
        checked.append(str(model_root))
        ref = model_root / "refs" / "main"
        if ref.is_file():
            snapshot = model_root / "snapshots" / ref.read_text(encoding="utf-8").strip()
            if _valid_snapshot(snapshot):
                return snapshot.resolve()
        snapshots = sorted(
            (path for path in (model_root / "snapshots").glob("*") if path.is_dir()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for snapshot in snapshots:
            if _valid_snapshot(snapshot):
                return snapshot.resolve()
    raise FileNotFoundError(
        "no complete local Qwen3.5-9B snapshot was found; checked:\n  - "
        + "\n  - ".join(checked)
    )


def model_metadata(model_path: Path) -> dict[str, Any]:
    config = json.loads((model_path / "config.json").read_text(encoding="utf-8"))
    text_config = config.get("text_config", {})
    return {
        "model_path": str(model_path),
        "snapshot_id": model_path.name,
        "architectures": config.get("architectures", []),
        "model_type": config.get("model_type"),
        "context_length": text_config.get("max_position_embeddings")
        or config.get("max_position_embeddings"),
        "transformers_config_version": config.get("transformers_version"),
    }


class ContextOverflowError(ValueError):
    pass


class LocalQwenGenerator:
    """Load one frozen model and use it for consolidation and verbalization."""

    def __init__(
        self,
        model_path: Path,
        device_map: str = "cuda:0",
        attn_implementation: str = "flash_attention_2",
        context_length: int | None = None,
    ):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.model_path = model_path
        self.metadata = model_metadata(model_path)
        self.context_length = int(context_length or self.metadata["context_length"])
        if self.context_length <= 0:
            raise ValueError("Qwen config did not expose a positive context length")
        self.processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_path),
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map=device_map,
            attn_implementation=attn_implementation,
        )
        self.model.eval()
        self.device = self.model.device
        self.device_map = device_map
        self.attn_implementation = attn_implementation
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)

    def _inputs(self, image_path: Path, prompt: str, enable_thinking: bool):
        from qwen_vl_utils import process_vision_info

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": "file://" + str(image_path.resolve())},
                {"type": "text", "text": prompt},
            ],
        }]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        images, videos = process_vision_info(messages, image_patch_size=16)
        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            do_resize=False,
            padding=True,
            return_tensors="pt",
        )
        return inputs.to(self.device)

    def generate(
        self,
        image_path: Path,
        prompt: str,
        *,
        enable_thinking: bool,
        max_new_tokens: int,
        attempt: int = 1,
    ) -> tuple[str, dict[str, Any]]:
        import torch

        inputs = self._inputs(image_path, prompt, enable_thinking)
        input_tokens = int(inputs["input_ids"].shape[-1])
        if input_tokens + max_new_tokens > self.context_length:
            raise ContextOverflowError(
                f"input_tokens={input_tokens} + max_new_tokens={max_new_tokens} "
                f"exceeds context_length={self.context_length}; no truncation was applied"
            )
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=int(max_new_tokens),
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - started
        completion = output[:, inputs["input_ids"].shape[-1]:]
        text = self.processor.batch_decode(
            completion,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        output_tokens = int(completion.shape[-1])
        return text, {
            "model_path": str(self.model_path),
            "snapshot_id": self.model_path.name,
            "context_length": self.context_length,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "max_new_tokens": int(max_new_tokens),
            "enable_thinking": bool(enable_thinking),
            "decode": "greedy",
            "attempt": attempt,
            "elapsed_seconds": elapsed,
        }

    def count_text_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse JSON despite an optional markdown fence or thinking wrapper."""
    import json as json_module

    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE).replace("```", "").strip()
    decoder = json_module.JSONDecoder()
    for match in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned[match.start():])
        except json_module.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("local model output did not contain a JSON object")
