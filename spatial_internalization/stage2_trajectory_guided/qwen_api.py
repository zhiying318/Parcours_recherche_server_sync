"""OpenAI-SDK Qwen3.5-397B-A17B generator for v2 data creation.

This follows the repository's working ``spatial_eval.backends.openai`` path:
the base URL ends at ``/v1`` and the SDK appends ``/chat/completions``.
"""

from __future__ import annotations

import base64
import mimetypes
import os
from pathlib import Path
import time
from typing import Any

from ..core.offline_qwen import ContextOverflowError


DEFAULT_API_BASE_URL = "https://api-2.xi-ai.cn/v1"
DEFAULT_API_MODEL = "qwen3.5-397b-a17b"
DEFAULT_API_KEY_ENV = "OPENAI_API_KEY"
DEFAULT_API_BASE_URL_ENV = "OPENAI_BASE_URL"


def _import_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The official OpenAI SDK is required. Install it with: "
            "uv pip install -r requirements_api.txt"
        ) from exc
    return OpenAI


def _clean_api_key(value: str) -> str:
    cleaned = value.strip().strip("\"'").strip()
    if cleaned.lower().startswith("bearer "):
        cleaned = cleaned[7:].strip().strip("\"'").strip()
    return cleaned


def _normalize_base_url(value: str) -> str:
    """Normalize the user-facing base URL to the SDK's ``.../v1`` form."""
    normalized = value.strip().rstrip("/")
    suffix = "/chat/completions"
    if normalized.endswith(suffix):
        normalized = normalized[: -len(suffix)]
    return normalized.rstrip("/")


def _image_data_url(image_path: Path) -> str:
    if not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _text_from_content(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    parts = []
    for block in value or []:
        text = _value(block, "text", "")
        if _value(block, "type") in (None, "text", "output_text") and text:
            parts.append(str(text).strip())
    return "\n".join(parts).strip()


def _usage_dict(usage: Any) -> dict[str, Any]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    return {
        name: value
        for name in ("prompt_tokens", "completion_tokens", "total_tokens")
        if (value := _value(usage, name)) is not None
    }


def _usage_value(usage: dict[str, Any], name: str) -> int | None:
    value = usage.get(name)
    return int(value) if isinstance(value, (int, float)) else None


class APIRequestError(RuntimeError):
    """An SDK request failed without exposing the API key."""


class OpenAICompatibleQwenGenerator:
    """One shared OpenAI SDK client for consolidation and verbalization."""

    def __init__(
        self,
        *,
        api_key_env: str | None = None,
        base_url: str | None = None,
        model: str = DEFAULT_API_MODEL,
        timeout: float = 120.0,
        request_retries: int = 5,
        context_length: int | None = None,
        client: Any | None = None,
    ):
        requested_key_env = api_key_env or DEFAULT_API_KEY_ENV
        key_candidates = [requested_key_env]
        if requested_key_env == "OPENAI_API_KEY":
            key_candidates.append("XI_AI_API_KEY")
        actual_key_env = next(
            (name for name in key_candidates if _clean_api_key(os.environ.get(name, ""))),
            None,
        )
        if actual_key_env is None:
            raise RuntimeError(f"Set {requested_key_env} before using the API generator")
        api_key = _clean_api_key(os.environ[actual_key_env])
        actual_base_url = base_url or os.environ.get(DEFAULT_API_BASE_URL_ENV) or DEFAULT_API_BASE_URL
        actual_base_url = _normalize_base_url(actual_base_url)
        if not actual_base_url.startswith(("http://", "https://")):
            raise ValueError("API base URL must be an HTTP(S) URL ending at /v1")
        if request_retries < 0:
            raise ValueError("request_retries must be non-negative")

        if client is None:
            OpenAI = _import_openai_client()
            client = OpenAI(
                api_key=api_key,
                base_url=actual_base_url,
                timeout=float(timeout),
                max_retries=int(request_retries),
            )
        self.client = client
        self.api_key_env = actual_key_env
        self.base_url = actual_base_url
        self.model = model
        self.timeout = float(timeout)
        self.request_retries = int(request_retries)
        self.context_length = int(context_length) if context_length else None

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "provider": "openai_sdk_chat_completions",
            "api_base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "model": self.model,
            "context_length": self.context_length,
            "context_length_source": (
                "cli" if self.context_length is not None else "not_exposed_by_endpoint"
            ),
            "sdk_request_path": "/chat/completions",
        }

    def generate(
        self,
        image_path: Path | None,
        prompt: str,
        *,
        enable_thinking: bool,
        max_new_tokens: int,
        attempt: int = 1,
    ) -> tuple[str, dict[str, Any]]:
        started = time.perf_counter()
        content = []
        if image_path is not None:
            content.append({
                "type": "image_url",
                "image_url": {"url": _image_data_url(image_path)},
            })
        # This is a normal user prompt, not an extra provider-specific body
        # parameter. It keeps the two offline generation settings explicit.
        thinking_directive = "/think" if enable_thinking else "/no_think"
        content.append({"type": "text", "text": f"{thinking_directive}\n{prompt}"})
        params: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            # This matches the working spatial_eval OpenAI backend.
            "max_completion_tokens": int(max_new_tokens),
        }
        try:
            response = self.client.chat.completions.create(**params)
        except Exception as exc:
            status_code = getattr(exc, "status_code", None)
            message = str(exc)[:1000]
            if status_code == 400 and any(word in message.lower() for word in ("context", "token", "length")):
                raise ContextOverflowError(
                    "API rejected the request for context/length; the original CoT was not truncated. "
                    + message
                ) from exc
            raise APIRequestError(
                f"OpenAI SDK request failed{f' (HTTP {status_code})' if status_code else ''}: {message}"
            ) from exc
        elapsed = time.perf_counter() - started

        choices = _value(response, "choices", []) or []
        if not choices:
            raise APIRequestError("API response did not contain choices[0]")
        message = _value(choices[0], "message", {})
        text = _text_from_content(_value(message, "content"))
        if not text:
            usage = _usage_dict(_value(response, "usage"))
            finish_reason = _value(choices[0], "finish_reason")
            raise APIRequestError(
                "API response did not contain message.content; "
                f"finish_reason={finish_reason!r}, usage={usage!r}. "
                "Increase --max-new-tokens if the completion was cut off."
            )
        usage = _usage_dict(_value(response, "usage"))
        input_tokens = _usage_value(usage, "prompt_tokens")
        output_tokens = _usage_value(usage, "completion_tokens")
        return text, {
            **self.metadata,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": _usage_value(usage, "total_tokens"),
            "usage_available": input_tokens is not None or output_tokens is not None,
            "usage_raw": usage,
            "max_new_tokens": int(max_new_tokens),
            "enable_thinking": bool(enable_thinking),
            "thinking_control": "prompt_directive",
            "attempt": attempt,
            "elapsed_seconds": elapsed,
            "response_model": _value(response, "model"),
            "finish_reason": _value(choices[0], "finish_reason"),
        }

    def count_text_tokens(self, text: str) -> int | None:
        """Do not substitute a local tokenizer for the remote tokenizer."""
        return None


__all__ = [
    "APIRequestError",
    "DEFAULT_API_BASE_URL",
    "DEFAULT_API_KEY_ENV",
    "DEFAULT_API_MODEL",
    "OpenAICompatibleQwenGenerator",
]
