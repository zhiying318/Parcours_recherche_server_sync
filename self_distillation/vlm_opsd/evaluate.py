"""Evaluate a Qwen3.5-VL OPSD LoRA adapter on clean held-out JSONL."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from self_distillation.vlm_opsd.arguments import parse_bool

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument(
        "--model-name", default="Qwen/Qwen3.5-9B"
    )
    parser.add_argument(
        "--data-jsonl",
        type=Path,
        default=REPO_ROOT / "self_distillation/data/test.jsonl",
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=REPO_ROOT / "self_distillation/output/test_predictions.jsonl",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--student_thinking", "--student-thinking", type=parse_bool, default=False)
    parser.add_argument("--no_thinking", "--no-thinking", dest="student_thinking", action="store_false")
    return parser.parse_args()


def final_letter(text: str) -> str:
    answer = text.rsplit("</think>", 1)[-1]
    matches = re.findall(
        r"(?:answer|option|choice)(?:\s+is|\s*[:=])?\s*([A-D])\b",
        answer,
        flags=re.IGNORECASE,
    )
    if matches:
        return matches[-1].upper()

    answer = re.sub(r"<\|[^>]*\|>", "", answer).strip()
    bare = re.fullmatch(r"([A-D])(?:[.)])?", answer, flags=re.IGNORECASE)
    return bare.group(1).upper() if bare else ""


def main() -> None:
    args = parse_args()
    records = [json.loads(line) for line in args.data_jsonl.read_text().splitlines()]
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
    )
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    predictions = []
    for record in records:
        image_path = str(REPO_ROOT / record["image_path"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": record["problem"]},
                ],
            }
        ]
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"do_resize": False},
            enable_thinking=args.student_thinking,
        ).to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )
        completion_ids = output_ids[:, inputs["input_ids"].shape[1] :]
        response = processor.batch_decode(
            completion_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )[0]
        prediction = final_letter(response)
        correct_letter = record["correct_letter"]
        predictions.append(
            {
                "id": record["id"],
                "prediction": prediction,
                "correct_letter": correct_letter,
                "correct": prediction == correct_letter,
                "response": response,
            }
        )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for prediction in predictions:
            output.write(json.dumps(prediction, ensure_ascii=False) + "\n")
    accuracy = sum(item["correct"] for item in predictions) / len(predictions)
    print(f"accuracy={accuracy:.6f} ({sum(item['correct'] for item in predictions)}/{len(predictions)})")


if __name__ == "__main__":
    main()
