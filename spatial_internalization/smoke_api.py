#!/usr/bin/env python3
"""One-request smoke test for the v2 Qwen API generator.

This command checks credentials, image encoding, Chat Completions request
shape, JSON response parsing, and usage reporting. It does not write training
data or load the student model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .api_qwen import (
    DEFAULT_API_MODEL,
    OpenAICompatibleQwenGenerator,
)
from .offline_qwen import parse_json_object


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = REPO_ROOT / "COMFORT/data/comfort_human_car_geometry_gt/comfort_human_car/behind/basketball__behind__cam_back/0.png"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--api-key-env", default=None, help="defaults to OPENAI_API_KEY, then XI_AI_API_KEY")
    parser.add_argument("--api-base-url", "--base-url", dest="api_base_url", default=None)
    parser.add_argument("--api-model", default=DEFAULT_API_MODEL)
    parser.add_argument("--api-timeout", type=float, default=120.0)
    parser.add_argument("--api-request-retries", type=int, default=5)
    parser.add_argument("--api-context-length", type=int, default=None)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "81920")),
        help="completion budget; follows OPENAI_MAX_OUTPUT_TOKENS and defaults to 81920",
    )
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="omit the image to isolate text/model routing from the vision request",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    generator = OpenAICompatibleQwenGenerator(
        api_key_env=args.api_key_env,
        base_url=args.api_base_url,
        model=args.api_model,
        timeout=args.api_timeout,
        request_retries=args.api_request_retries,
        context_length=args.api_context_length,
    )
    prompt = (
        "Return JSON only with this shape: "
        '{"ok":true,"observation":"one short visual observation"}. '
        "Do not include markdown or hidden reasoning in the JSON."
    )
    text, generation = generator.generate(
        None if args.text_only else args.image.resolve(),
        prompt,
        enable_thinking=args.enable_thinking,
        max_new_tokens=args.max_new_tokens,
        attempt=1,
    )
    parsed = parse_json_object(text)
    if parsed.get("ok") is not True or not str(parsed.get("observation", "")).strip():
        raise ValueError(f"smoke response has unexpected JSON shape: {parsed}")
    print(json.dumps({"status": "ok", "generation": generation, "response": parsed}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
