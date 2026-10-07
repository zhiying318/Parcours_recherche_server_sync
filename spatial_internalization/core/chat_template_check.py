"""Inspect the configured model processor without loading model weights."""

from __future__ import annotations

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    args = parser.parse_args()
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_name)
    tokenizer = processor.tokenizer
    print(json.dumps({
        "model_name": args.model_name,
        "tokenizer_class": tokenizer.__class__.__name__,
        "chat_template_present": bool(getattr(tokenizer, "chat_template", None)),
        "chat_template_has_thinking_switch": "enable_thinking" in (getattr(tokenizer, "chat_template", "") or ""),
        "reasoning_policy": "The comparison run uses enable_thinking=False; the thinking run uses enable_thinking=True with reasoning_content supplied separately from the final answer.",
        "special_tokens": {"bos": tokenizer.bos_token, "eos": tokenizer.eos_token, "pad": tokenizer.pad_token},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
