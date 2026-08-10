#!/usr/bin/env python3
"""One-image Qwen3.5 thinking smoke test for FlashAttention 2."""

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from spatial_eval.backends.qwen3_5vl import Qwen35VLThinkingBackend
from spatial_eval.prompts.MCQ import MCQAsker
from spatial_eval.utils import parse_relation_for_COMFORT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available inside the container")

    root = Path(__file__).resolve().parent
    image_paths = json.loads(
        (root / "test01_camera_side/data/image_paths.json").read_text(encoding="utf-8")
    )
    prompt_info = json.loads(
        (root / "test01_camera_side/data/prompt_info.json").read_text(encoding="utf-8")
    )
    image_path = image_paths[0]
    second_object, relation = parse_relation_for_COMFORT(image_path)
    relation = {
        "infrontof": "front",
        "totheleft": "left",
        "totheright": "right",
    }.get(relation, relation)

    backend = Qwen35VLThinkingBackend(
        model_id=args.model_id,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    actual_attention = getattr(backend.model.config, "_attn_implementation", None)
    if actual_attention != "flash_attention_2":
        raise RuntimeError(
            f"Expected flash_attention_2, model reports {actual_attention!r}"
        )

    asker = MCQAsker(
        answer_length="long",
        seed=123,
        max_new_tokens_mcq=args.max_new_tokens,
        prompt_info_by_image=prompt_info,
    )
    result = asker.evaluate_one(
        backend=backend,
        img_path=f"./{image_path}",
        second_object=second_object,
        correct_relation=relation,
    )
    print(f"FlashAttention smoke test passed: pred={result['pred_letter']!r}")


if __name__ == "__main__":
    main()
