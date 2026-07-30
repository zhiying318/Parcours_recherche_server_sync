from __future__ import annotations

import base64
import json
import mimetypes
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List

from .base import VLMBackend


def _import_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The official OpenAI SDK is required for the openai backend. "
            "Install it with: uv pip install -r requirements_api.txt"
        ) from exc
    return OpenAI


def _image_data_url(image_path: str) -> str:
    path = Path(image_path)
    if not path.exists() and image_path.startswith("./"):
        path = Path(image_path[2:])
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    parts = []
    for block in content or []:
        block_type = _value(block, "type")
        if block_type in (None, "text", "output_text"):
            text = _value(block, "text", "")
            if text:
                parts.append(str(text).strip())
    return "\n".join(parts).strip()


@dataclass
class OpenAIBackend(VLMBackend):
    """OpenAI SDK backend for OpenAI-compatible services.

    ``api_mode=chat_completions`` uses the widely supported compatibility
    endpoint. ``api_mode=responses`` is available only for providers that
    explicitly implement the Responses API.
    """

    model_id: str
    api_mode: str = "chat_completions"
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    reasoning_jsonl: str | None = None
    timeout: float = 120.0
    max_retries: int = 5

    def __post_init__(self):
        api_key = (os.environ.get(self.api_key_env) or "").strip()
        if not api_key:
            raise RuntimeError(
                f"Set {self.api_key_env} before using the openai backend."
            )
        base_url = (os.environ.get(self.base_url_env) or "").strip() or None
        if self.api_mode not in ("chat_completions", "responses"):
            raise ValueError(
                "api_mode must be 'chat_completions' or 'responses'."
            )

        OpenAI = _import_openai_client()
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        self.enable_thinking = self.reasoning_effort not in (None, "none")

    @staticmethod
    def _responses_summary(response: Any) -> str:
        parts = []
        for output_item in _value(response, "output", []) or []:
            if _value(output_item, "type") != "reasoning":
                continue
            for summary in _value(output_item, "summary", []) or []:
                text = _value(summary, "text", "")
                if text:
                    parts.append(str(text).strip())
        return "\n\n".join(parts)

    @staticmethod
    def _chat_summary(message: Any) -> str:
        # Aggregators use different names. Reading these fields is harmless;
        # request-side provider-specific thinking parameters are not guessed.
        for field in (
            "reasoning_content",
            "reasoning",
            "thinking",
            "reasoning_summary",
        ):
            value = _value(message, field)
            if isinstance(value, str) and value.strip():
                return value.strip()
            text = _text_from_content(value)
            if text:
                return text
        return ""

    def _write_sidecar(
        self,
        *,
        image_paths: List[str],
        summary: str,
        response_id: str | None,
    ) -> None:
        if not self.reasoning_jsonl:
            return
        path = Path(self.reasoning_jsonl)
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "image_paths": image_paths,
            "model_id": self.model_id,
            "api_mode": self.api_mode,
            "response_id": response_id,
            "reasoning_summary": summary,
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()

    def _complete_responses(
        self, content: List[dict], max_output_tokens: int
    ) -> tuple[str, str, str | None]:
        params: dict[str, Any] = {
            "model": self.model_id,
            "input": [{"role": "user", "content": content}],
            "max_output_tokens": int(max_output_tokens),
        }
        if self.reasoning_effort is not None:
            reasoning = {"effort": self.reasoning_effort}
            if self.reasoning_summary is not None:
                reasoning["summary"] = self.reasoning_summary
            params["reasoning"] = reasoning

        response = self.client.responses.create(**params)
        if _value(response, "status") == "incomplete":
            details = _value(response, "incomplete_details")
            raise RuntimeError(
                "API response was incomplete: "
                f"{_value(details, 'reason', 'unknown')}. "
                "Increase --max_new_tokens_mcq."
            )
        answer = str(_value(response, "output_text", "") or "").strip()
        return (
            answer,
            self._responses_summary(response),
            _value(response, "id"),
        )

    def _complete_chat(
        self, content: List[dict], max_output_tokens: int
    ) -> tuple[str, str, str | None]:
        chat_content = []
        for item in content:
            if item["type"] == "input_text":
                chat_content.append({"type": "text", "text": item["text"]})
            else:
                chat_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": item["image_url"]},
                    }
                )
        params: dict[str, Any] = {
            "model": self.model_id,
            "messages": [{"role": "user", "content": chat_content}],
            "max_completion_tokens": int(max_output_tokens),
        }
        if self.reasoning_effort not in (None, "none"):
            params["reasoning_effort"] = self.reasoning_effort

        response = self.client.chat.completions.create(**params)
        choices = _value(response, "choices", []) or []
        if not choices:
            raise RuntimeError("API response did not contain any choices.")
        message = _value(choices[0], "message")
        return (
            _text_from_content(_value(message, "content")),
            self._chat_summary(message),
            _value(response, "id"),
        )

    def _complete(
        self, content: List[dict], image_paths: List[str], max_output_tokens: int
    ) -> str:
        if self.api_mode == "responses":
            answer, summary, response_id = self._complete_responses(
                content, max_output_tokens
            )
        else:
            answer, summary, response_id = self._complete_chat(
                content, max_output_tokens
            )
        if not answer:
            raise RuntimeError("API response did not contain a final text answer.")
        self._write_sidecar(
            image_paths=image_paths,
            summary=summary,
            response_id=response_id,
        )
        # Keep model_answer identical in meaning to existing CSV files.
        return answer

    def ask(
        self,
        image_path: str,
        prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        content = [
            {
                "type": "input_image",
                "image_url": _image_data_url(image_path),
            },
            {"type": "input_text", "text": prompt},
        ]
        return self._complete(content, [image_path], max_new_tokens)

    def ask_multi(
        self,
        image_paths: List[str],
        prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        content = []
        for index, image_path in enumerate(image_paths, start=1):
            content.append(
                {"type": "input_text", "text": f"Image {index}:"}
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": _image_data_url(image_path),
                }
            )
        content.append({"type": "input_text", "text": prompt})
        return self._complete(content, image_paths, max_new_tokens)
