# spatial_eval/backends/factory.py
import torch
from .base import VLMBackend

def build_backend(name: str, model_id: str, dtype: torch.dtype, device_map: str) -> VLMBackend:
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
    if name in ("qwen3-vl-thinking"):
        from .qwen3vlthinking import Qwen3ThinkingBackend
        return Qwen3ThinkingBackend(model_id=model_id, dtype="auto", device_map=device_map)
    if name in ("gemma3"):
        from .gemma3 import Gemma3Backend
        return Gemma3Backend(model_id=model_id, dtype=dtype, device_map=device_map)
    if name in ("qwen3.5vl", "qwen3.5"):
        from .qwen3_5vl import Qwen35VLBackend
        return Qwen35VLBackend(model_id=model_id, dtype="auto", device_map=device_map)

    if name in ("qwen3vl-logits", "qwen3-logits"):
        from .qwen3vl_logits import Qwen3VLLogitsBackend
        return Qwen3VLLogitsBackend(model_id=model_id, dtype="auto", device_map=device_map)
    raise ValueError(f"Unknown backend '{name}'.")