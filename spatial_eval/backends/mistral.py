# spatial_eval/backends/mistral.py
from __future__ import annotations

import base64
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from .base import VLMBackend


def _import_mistral_client():
    try:
        from mistralai.client import Mistral
    except ImportError:
        try:
            from mistralai import Mistral
        except ImportError as exc:
            raise ImportError(
                "The official Mistral SDK is required for the mistral backend. "
                "Install it with: pip install mistralai"
            ) from exc
    return Mistral


def _image_data_url(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists() and image_path.startswith("./"):
        path = Path(image_path[2:])
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _response_text(response: Any) -> str:
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, KeyError, TypeError):
        content = response["choices"][0]["message"]["content"]

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(getattr(item, "text", "")))
        return "".join(parts).strip()
    return str(content).strip()


def _redact_key(api_key: str) -> str:
    if len(api_key) <= 10:
        return f"{api_key[:2]}...{api_key[-2:]}"
    return f"{api_key[:6]}...{api_key[-4:]}"


@dataclass
class MistralBackend(VLMBackend):
    model_id: str
    api_key_env: str = "MISTRAL_API_KEY"
    temperature: float | None = None
    top_p: float | None = None
    reasoning_effort: str | None = None

    def __post_init__(self):
        api_key = (os.environ.get(self.api_key_env) or "").strip()
        if not api_key:
            raise RuntimeError(f"Set {self.api_key_env} before using the mistral backend.")

        self._api_key_hint = _redact_key(api_key)
        Mistral = _import_mistral_client()
        self.client = Mistral(api_key=api_key)

    def _complete(self, content: List[dict], max_tokens: int) -> str:
        params = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": int(max_tokens),
        }
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.top_p is not None:
            params["top_p"] = self.top_p
        if self.reasoning_effort is not None:
            params["reasoning_effort"] = self.reasoning_effort

        try:
            response = self.client.chat.complete(**params)
        except Exception as exc:
            message = str(exc)
            if "Status 401" in message or "Unauthorized" in message:
                raise RuntimeError(
                    "Mistral API returned 401 Unauthorized. "
                    f"Check that {self.api_key_env} is a valid Mistral Console API key "
                    f"visible to this shell; current key looks like {self._api_key_hint}."
                ) from exc
            raise
        return _response_text(response)

    def ask(self, image_path: str, prompt: str, max_new_tokens: int = 512) -> str:
        content = [
            {"type": "image_url", "image_url": _image_data_url(image_path)},
            {"type": "text", "text": prompt},
        ]
        return self._complete(content, max_new_tokens)

    def ask_multi(self, image_paths, prompt: str, max_new_tokens: int = 512) -> str:
        content = []
        for idx, image_path in enumerate(image_paths, start=1):
            content.append({"type": "text", "text": f"Image {idx}:"})
            content.append({"type": "image_url", "image_url": _image_data_url(image_path)})
        content.append({"type": "text", "text": prompt})
        return self._complete(content, max_new_tokens)
