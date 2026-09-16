"""Evaluate teacher accuracy and completion length on JSONL data."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from self_distillation.vlm_opsd.arguments import parse_bool

REPO_ROOT = Path(__file__).resolve().parents[2]

# Evaluation-only prompt. Edit this template freely; it is intentionally
# independent from the teacher prompt used for distillation in collator.py.

#### Determine the spatial relation required by the question. The answer must be
# exactly one of: front, behind, left, right.
# You may reason before answering, but end with exactly this format:
# Final answer: RELATION
# Replace RELATION with exactly one of front, behind, left, or right.

#### INITIAL STANDARD TEACHER'S PROMPT
# EVAL_TEACHER_CONTEXT = """\
# Here is a reference solution.
# === Reference Solution Begin ===
# {teacher_geometry}
# === Reference Solution End ===
# After understanding the reference solution, please try to solve this problem using your own approach below:

# Answer:
# """

EVAL_TEACHER_CONTEXT = """\
Here is a reference solution.
=== Reference Solution Begin ===
{teacher_geometry}
=== Reference Solution End ===

After understanding the reference solution, determine the spatial relation required by the question.
Reason carefully but concisely:
- Identify only the geometric facts relevant to the queried relation.
- Perform only the reasoning necessary to verify the answer.
- Do not restate coordinates or other irrelevant details.
- Avoid alternative derivations, repeated verification, or unnecessary explanation.
- Keep the reasoning brief and focused.
Then provide the final spatial relation.

Answer:
"""

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument(
        "--max-new-tokens", type=int, default=32960,
        help="Maximum teacher completion length (default: 32960).",
    )
    parser.add_argument(
        "--teacher-thinking", "--teacher_thinking",
        dest="teacher_thinking", type=parse_bool, default=False,
        help="Enable thinking in the teacher chat template.",
    )
    parser.add_argument(
        "--no-thinking", "--no_thinking",
        dest="teacher_thinking", action="store_false",
    )
    parser.add_argument(
        "--include-options", "--include_options",
        type=parse_bool, default=False,
        help="Whether to include the original A-D choices in the user message.",
    )
    parser.add_argument(
        "--no-options", "--no_options",
        dest="include_options", action="store_false",
    )
    return parser.parse_args()


def final_letter(text: str) -> str:
    answer = text.rsplit("</think>", 1)[-1]
    matches = re.findall(
        r"(?:(?:final|correct)\s+)?(?:answer|option|choice)"
        r"(?:\s+is)?\s*[:=]?\s*\**([A-D])\b",
        answer,
        flags=re.IGNORECASE,
    )
    if matches:
        return matches[-1].upper()

    answer = re.sub(r"<\|[^>]*\|>", "", answer).strip()
    bare = re.fullmatch(r"([A-D])(?:[.)])?", answer, flags=re.IGNORECASE)
    return bare.group(1).upper() if bare else ""


def final_relation(text: str) -> str:
    """Extract the spatial relation, preferring the required final-answer format."""
    answer = re.sub(r"<\|[^>]*\|>", "", text.rsplit("</think>", 1)[-1]).lower()
    final = re.findall(
        r"final\s+answer\s*:\s*(front|behind|left|right)\b", answer
    )
    if final:
        return final[-1]
    # Keep a fallback so an otherwise valid answer is still measurable when
    # the model omits the requested prefix.
    matches = re.findall(r"\b(?:in\s+)?front\b|\bbehind\b|\bleft\b|\bright\b", answer)
    if not matches:
        return ""
    relation = matches[-1].strip()
    return "front" if relation == "in front" else relation


def print_summary(predictions: list[dict], total: int) -> None:
    correct = sum(item["correct"] for item in predictions)
    lengths = [item["completion_length"] for item in predictions]
    hit_limit = sum(item.get("hit_max_length", False) for item in predictions)
    print(
        f"progress={len(predictions)}/{total}, "
        f"accuracy={correct / len(predictions):.6f} ({correct}/{len(predictions)}), "
        f"completion_tokens: mean={statistics.fmean(lengths):.2f}, "
        f"median={statistics.median(lengths):.1f}, min={min(lengths)}, "
        f"max={max(lengths)}, hit_max_length={hit_limit}",
        flush=True,
    )


def load_predictions(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluation_hash(args: argparse.Namespace) -> str:
    config = {
        "context": EVAL_TEACHER_CONTEXT,
        "include_options": args.include_options,
        "max_new_tokens": args.max_new_tokens,
        "model_name": args.model_name,
        "teacher_thinking": args.teacher_thinking,
    }
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def teacher_prompt_text(record: dict, include_options: bool) -> str:
    """Build the evaluation-only privileged teacher prompt."""
    problem = record["problem"]
    if not include_options: # remove the ABCD choices if the parameter=False
        problem = re.split(r"\n\s*[A-D]\.\s", problem, maxsplit=1)[0].rstrip()
    return problem + "\n\n" + EVAL_TEACHER_CONTEXT.format(
        teacher_geometry=record["teacher_geometry"]
    )


def thinking_trace(response: str) -> str:
    """Return the full text inside the generated thinking block, if present."""
    match = re.search(r"<think>(.*?)(?:</think>|$)", response, flags=re.DOTALL)
    return match.group(1) if match else ""


def conversation(image_path: str, text: str) -> list[dict]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": text},
            ],
        }
    ]


def main() -> None:
    args = parse_args()
    run_hash = evaluation_hash(args)
    records = [json.loads(line) for line in args.data_jsonl.read_text().splitlines()]
    predictions = load_predictions(args.output_jsonl)
    prompt_hashes = {item.get("prompt_hash") for item in predictions}
    if predictions and prompt_hashes != {run_hash}:
        raise ValueError(
            "The existing output was produced with a different evaluation setup. "
            "Use a new --output-jsonl after changing the prompt or evaluation flags."
        )
    completed_ids = {item["id"] for item in predictions}
    remaining = [record for record in records if record["id"] not in completed_ids]
    if predictions:
        print(f"Resuming from {len(predictions)} existing records.", flush=True)
        print_summary(predictions, len(records))
    if not remaining:
        return

    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda:0",
    )
    model.eval()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("a", encoding="utf-8") as output:
        for record in remaining:
            image_path = str(REPO_ROOT / record["image_path"])
            prompt = teacher_prompt_text(record, args.include_options)
            messages = conversation(image_path, prompt)
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={"do_resize": False},
                enable_thinking=args.teacher_thinking,
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
            predicted_letter = final_letter(response) if args.include_options else ""
            if predicted_letter:
                prediction = predicted_letter
                target = record["correct_letter"]
                answer_type = "letter"
            else:
                prediction = final_relation(response)
                target = record["relation"]
                answer_type = "relation"
            prediction_record = {
                "id": record["id"],
                "prediction": prediction,
                "target": target,
                "correct": prediction == target,
                "response": response,
                "thinking_trace": thinking_trace(response),
                "prompt": prompt,
                "prompt_hash": run_hash,
                "include_options": args.include_options,
                "answer_type": answer_type,
                "completion_length": completion_ids.shape[1],
                "hit_max_length": completion_ids.shape[1] >= args.max_new_tokens,
                "valid_answer": bool(prediction),
                "teacher_thinking": args.teacher_thinking,
            }
            predictions.append(prediction_record)
            output.write(json.dumps(prediction_record, ensure_ascii=False) + "\n")
            output.flush()
            print_summary(predictions, len(records))

if __name__ == "__main__":
    main()
