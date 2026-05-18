# spatial_eval/backends/factory.py
from typing import Any
from .base import VLMBackend

def build_backend(
    name: str,
    model_id: str,
    dtype: Any,
    device_map: str,
    mistral_temperature: float | None = None,
    mistral_top_p: float | None = None,
    mistral_reasoning_effort: str | None = None,
) -> VLMBackend:
    name = name.lower()
    if name in ("qwen", "qwen2", "qwen2.5", "qwen2.5vl"):
        from .qwen2_5 import Qwen2Backend
        return Qwen2Backend(model_id=model_id, dtype=dtype, device_map=device_map)
    if name in ("qwen3vl", "qwen3"):
        from .qwen3vl import Qwen3VLBackend
        return Qwen3VLBackend(model_id=model_id, dtype="auto", device_map=device_map)
    if name in ("internvl", "internvl3.5", "intern"):
        from .internvl import InternVLBackend
        return InternVLBackend(model_id=model_id, dtype=dtype, device_map=device_map)
    if name in ("qwen3-vl-thinking"): # 这个backend里面用的不是标准读图的方法。用的是PIL，应该用process_vision_info
        from .qwen3vlthinking import Qwen3ThinkingBackend
        return Qwen3ThinkingBackend(model_id=model_id, dtype="auto", device_map=device_map)
    if name in ("gemma3"):
        from .gemma3 import Gemma3Backend
        return Gemma3Backend(model_id=model_id, dtype=dtype, device_map=device_map)
    if name in ("gemma4"):
        from .gemma4 import Gemma4Backend
        return Gemma4Backend(model_id=model_id, dtype=dtype, device_map=device_map)
    if name in ("qwen3.5vl", "qwen3.5"):
        from .qwen3_5vl import Qwen35VLBackend
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
    raise ValueError(f"Unknown backend '{name}'.")
