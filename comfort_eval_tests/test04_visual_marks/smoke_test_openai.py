#!/usr/bin/env python3
"""Send exactly one OpenAI-compatible request for Test 04 visual marks."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from spatial_eval.backends.openai import OpenAIBackend
from spatial_eval.eval_runner import run_eval
from spatial_eval.prompts.MCQ import MCQAsker


TEST_DIR = Path(__file__).resolve().parent
RESULTS_DIR = TEST_DIR / "results"


def required_env(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise RuntimeError(f"Set {name} before running this smoke test.")
    return value


def model_tag(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", model_id)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Send exactly one Test 04 visual-marks API request."
    )
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument(
        "--answer_length",
        choices=["short", "middle", "long"],
        default="long",
    )
    parser.add_argument(
        "--run_mode", choices=["instruct", "thinking"], default="instruct"
    )
    parser.add_argument(
        "--api_mode",
        choices=["chat_completions", "responses"],
        default="chat_completions",
    )
    parser.add_argument("--reasoning_effort", default="high")
    parser.add_argument("--reasoning_summary", default="auto")
    parser.add_argument("--max_output_tokens", type=int, default=81920)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--max_retries",
        type=int,
        default=0,
        help="SDK retries after the first request; keep 0 for one-request smoke tests.",
    )
    args = parser.parse_args()

    required_env("OPENAI_API_KEY")
    required_env("OPENAI_BASE_URL")
    model_id = required_env("OPENAI_MODEL_ID")

    samples = json.loads((TEST_DIR / "image_paths.json").read_text(encoding="utf-8"))
    if not 0 <= args.sample_index < len(samples):
        parser.error(f"--sample_index must be between 0 and {len(samples) - 1}")

    thinking = args.run_mode == "thinking"
    tag = model_tag(model_id) + ("_thinking" if thinking else "")
    stem = (
        f"smoke_mcq_{args.answer_length}_{tag}_sample_{args.sample_index}"
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_csv = RESULTS_DIR / f"{stem}.csv"
    reasoning_jsonl = RESULTS_DIR / f"{stem}_reasoning.jsonl"

    backend = OpenAIBackend(
        model_id=model_id,
        api_mode=args.api_mode,
        reasoning_effort=args.reasoning_effort if thinking else None,
        reasoning_summary=(
            args.reasoning_summary
            if thinking and args.api_mode == "responses"
            else None
        ),
        reasoning_jsonl=str(reasoning_jsonl),
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    asker = MCQAsker(
        seed=123,
        max_new_tokens_mcq=args.max_output_tokens,
        answer_length=args.answer_length,
    )
    run_eval(
        backend=backend,
        image_paths=[samples[args.sample_index]],
        output_csv=str(output_csv),
        asker=asker,
    )

    print("Smoke test sent exactly 1 API request.")
    print(f"CSV: {output_csv}")
    print(f"Reasoning JSONL: {reasoning_jsonl}")


if __name__ == "__main__":
    main()
