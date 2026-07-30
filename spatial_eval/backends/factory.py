# spatial_eval/backends/factory.py
from typing import Any
from .base import VLMBackend

def build_backend(
    name: str,
    model_id: str,
    dtype: Any,
    device_map: str,
    enable_thinking: bool = False,
    mistral_temperature: float | None = None,
    mistral_top_p: float | None = None,
    mistral_reasoning_effort: str | None = None,
    openai_reasoning_effort: str | None = None,
    openai_reasoning_summary: str | None = None,
    openai_api_mode: str = "chat_completions",
    openai_reasoning_jsonl: str | None = None,
    openai_timeout: float = 120.0,
    openai_max_retries: int = 5,
) -> VLMBackend:
    name = name.lower()
    if name in ("qwen", "qwen2", "qwen2.5", "qwen2.5vl"):
        from .qwen2_5 import Qwen2Backend
        return Qwen2Backend(model_id=model_id, dtype=dtype, device_map=device_map)
    if name in ("qwen3vl", "qwen3"):
        from .qwen3vl import Qwen3VLBackend
        return Qwen3VLBackend(model_id=model_id, dtype="auto", device_map=device_map, enable_thinking=enable_thinking)
    if name in ("internvl", "internvl3.5", "intern"):
        from .internvl import InternVLBackend
        return InternVLBackend(model_id=model_id, dtype=dtype, device_map=device_map, enable_thinking=enable_thinking)
    if name in ("qwen3-vl-thinking"): # 这个backend里面用的不是标准读图的方法。用的是PIL，应该用process_vision_info
        from .qwen3vlthinking import Qwen3ThinkingBackend
        return Qwen3ThinkingBackend(model_id=model_id, dtype="auto", device_map=device_map)
    if name in ("gemma3"):
        from .gemma3 import Gemma3Backend
        return Gemma3Backend(model_id=model_id, dtype=dtype, device_map=device_map)
    if name in ("gemma4"):
        from .gemma4 import Gemma4Backend
        return Gemma4Backend(model_id=model_id, dtype=dtype, device_map=device_map, enable_thinking=enable_thinking)
    if name in ("qwen3.5vl", "qwen3.5"):
        from .qwen3_5vl import Qwen35VLBackend, Qwen35VLThinkingBackend
        if enable_thinking:
            return Qwen35VLThinkingBackend(model_id=model_id, dtype="auto", device_map=device_map)
        return Qwen35VLBackend(model_id=model_id, dtype="auto", device_map=device_map)
    if name in ("qwen3.5vl-thinking", "qwen3.5-thinking"):
        from .qwen3_5vl import Qwen35VLThinkingBackend
        return Qwen35VLThinkingBackend(model_id=model_id, dtype="auto", device_map=device_map)

    if name in ("qwen3vl-logits", "qwen3-logits"):
        from .qwen3vl_logits import Qwen3VLLogitsBackend
        return Qwen3VLLogitsBackend(model_id=model_id, dtype="auto", device_map=device_map)
    if name in ("mistral", "mistralai"):
        from .mistral import MistralBackend
        return MistralBackend(
            model_id=model_id,
            temperature=mistral_temperature,
            top_p=mistral_top_p,
            reasoning_effort=mistral_reasoning_effort,
        )
    if name == "openai":
        from .openai import OpenAIBackend
        return OpenAIBackend(
            model_id=model_id,
            reasoning_effort=openai_reasoning_effort,
            reasoning_summary=openai_reasoning_summary,
            api_mode=openai_api_mode,
            reasoning_jsonl=openai_reasoning_jsonl,
            timeout=openai_timeout,
            max_retries=openai_max_retries,
        )
    raise ValueError(f"Unknown backend '{name}'.")
