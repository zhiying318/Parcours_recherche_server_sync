#!/usr/bin/env python3
"""WhatsUp single-image MCQ evaluation.

This evaluator mirrors the COMFORT single-image MCQ setup, but computes the
person-perspective ground truth from WhatsUp edited-image filenames.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spatial_eval.backends.factory import build_backend
from spatial_eval.utils import get_dtype


RELATIONS_LONG = {
    "front": "From the person's perspective, the {second_object} is in front of them.",
    "behind": "From the person's perspective, the {second_object} is behind them.",
    "left": "From the person's perspective, the {second_object} is on their left.",
    "right": "From the person's perspective, the {second_object} is on their right.",
}

OPPOSITE_MAP = {
    "front": "behind",
    "behind": "front",
    "left": "right",
    "right": "left",
}

CSV_HEADER = [
    "image_path",
    "second_object",
    "reference_object",
    "original_relation",
    "face_view",
    "mcq_prompt",
    "correct_relation",
    "correct_letter",
    "opposite_relation",
    "opposite_letter",
    "model_answer",
    "pred_letter",
    "correct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate WhatsUp images with long MCQ prompts.")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument(
        "--image_json",
        default=str(
            Path(__file__).resolve().parent
            / "whatsup_image_validation"
            / "valide_image_paths.json"
        ),
    )
    parser.add_argument("--out_csv", required=True)
    parser.add_argument("--cuda_visible_devices", default=None)
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--enable_thinking", action="store_true")
    parser.add_argument("--max_new_tokens_mcq", type=int, default=8)
    parser.add_argument("--mcq_seed", type=int, default=123)
    parser.add_argument("--sample_index", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--openai_api_mode",
        choices=["chat_completions", "responses"],
        default="chat_completions",
    )
    parser.add_argument("--openai_reasoning_effort", default=None)
    parser.add_argument("--openai_reasoning_summary", default=None)
    parser.add_argument("--openai_reasoning_jsonl", default=None)
    parser.add_argument("--openai_timeout", type=float, default=120.0)
    parser.add_argument("--openai_max_retries", type=int, default=5)
    return parser.parse_args()


def stable_seed(seed: int, value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return seed + int(digest[:12], 16)


def normalize_choice(raw: str, thinking: bool = False) -> str:
    text = (raw or "").strip()
    text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

    if thinking:
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        else:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if lines:
                text = lines[-1]

    upper = text.upper()
    match = re.search(r"\b([ABCD])\b", upper)
    if match:
        return match.group(1)
    match = re.match(r"^([ABCD])[\.\)\:\-]?", upper)
    if match:
        return match.group(1)
    matches = re.findall(r"\b([ABCD])\b", (raw or "").upper())
    if matches:
        return matches[-1]
    return upper[:1]


def parse_whatsup_filename(image_path: str) -> dict[str, str]:
    base = Path(image_path).stem
    if "_FACE-" not in base:
        raise ValueError(f"Missing FACE tag in filename: {image_path}")

    core, face_view = base.rsplit("_FACE-", 1)
    if "_left_of_" in core:
        second_object, reference_object = core.split("_left_of_", 1)
        original_relation = "left"
    elif "_right_of_" in core:
        second_object, reference_object = core.split("_right_of_", 1)
        original_relation = "right"
    else:
        raise ValueError(f"Expected left_of/right_of relation in filename: {image_path}")

    if face_view == "CAMERA":
        correct_relation = "right" if original_relation == "left" else "left"
    elif face_view == "LEFT":
        correct_relation = "front" if original_relation == "left" else "behind"
    elif face_view == "RIGHT":
        correct_relation = "behind" if original_relation == "left" else "front"
    else:
        raise ValueError(f"Unexpected FACE tag in filename: {image_path}")

    return {
        "second_object": second_object,
        "reference_object": reference_object,
        "original_relation": original_relation,
        "face_view": face_view,
        "correct_relation": correct_relation,
    }


def resolve_image_path(image_path: str) -> str:
    path = Path(image_path)
    if path.exists():
        return str(path)

    repo_root = Path(__file__).resolve().parents[1]
    if not path.is_absolute():
        rooted = repo_root / image_path.lstrip("./")
        if rooted.exists():
            return str(rooted)

    return image_path


def completed_paths(out_csv: Path) -> set[str]:
    if not out_csv.exists() or out_csv.stat().st_size == 0:
        return set()
    with out_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != CSV_HEADER:
            raise ValueError(f"Cannot resume {out_csv}: CSV header is not compatible.")
        return {row[0] for row in reader if row}


def record_error(out_csv: Path, image_path: str, exc: Exception) -> None:
    error_path = out_csv.with_name(f"{out_csv.stem}_errors.jsonl")
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sample": {"image_path": image_path},
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    with error_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def build_prompt(
    image_path: str,
    second_object: str,
    correct_relation: str,
    seed: int,
) -> tuple[str, dict[str, str], str, str, str]:
    rng = random.Random(stable_seed(seed, image_path))
    relation_keys = list(RELATIONS_LONG.keys())
    rng.shuffle(relation_keys)

    letters = ["A", "B", "C", "D"]
    choices = {
        letters[i]: RELATIONS_LONG[relation_keys[i]].format(second_object=second_object)
        for i in range(4)
    }
    correct_letter = letters[relation_keys.index(correct_relation)]
    opposite_relation = OPPOSITE_MAP[correct_relation]
    opposite_letter = letters[relation_keys.index(opposite_relation)]

    option_lines = "\n".join(f"{letter}. {text}" for letter, text in choices.items())
    prompt = "\n".join(
        [
            f"Where is the {second_object} in the perspective of the person?",
            "Choose ONE option and respond with ONLY the letter.",
            option_lines,
        ]
    )
    return prompt, choices, correct_letter, opposite_relation, opposite_letter


def main() -> None:
    args = parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    dtype = None if args.backend == "openai" else get_dtype()
    backend = build_backend(
        args.backend,
        args.model_id,
        dtype=dtype,
        device_map=args.device_map,
        enable_thinking=args.enable_thinking,
        openai_reasoning_effort=args.openai_reasoning_effort,
        openai_reasoning_summary=args.openai_reasoning_summary,
        openai_api_mode=args.openai_api_mode,
        openai_reasoning_jsonl=args.openai_reasoning_jsonl,
        openai_timeout=args.openai_timeout,
        openai_max_retries=args.openai_max_retries,
    )

    with open(args.image_json, "r", encoding="utf-8") as f:
        image_paths = json.load(f)
    if not isinstance(image_paths, list):
        raise ValueError(f"Expected a JSON list in {args.image_json}")
    if args.sample_index is not None:
        if args.sample_index < 0 or args.sample_index >= len(image_paths):
            raise ValueError(
                f"--sample_index must be between 0 and {len(image_paths) - 1}"
            )
        image_paths = [image_paths[args.sample_index]]

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    completed = completed_paths(out_csv) if args.resume else set()
    append = args.resume and out_csv.exists() and out_csv.stat().st_size > 0
    failures = 0

    with out_csv.open("a" if append else "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not append:
            writer.writerow(CSV_HEADER)
            f.flush()

        for image_path in image_paths:
            if image_path in completed:
                continue
            try:
                meta = parse_whatsup_filename(image_path)
                prompt, _choices, correct_letter, opposite_relation, opposite_letter = build_prompt(
                    image_path=image_path,
                    second_object=meta["second_object"],
                    correct_relation=meta["correct_relation"],
                    seed=args.mcq_seed,
                )
                resolved_image_path = resolve_image_path(image_path)
                raw_answer = backend.ask(resolved_image_path, prompt, args.max_new_tokens_mcq)
                pred_letter = normalize_choice(
                    raw_answer,
                    thinking=getattr(backend, "enable_thinking", False) or args.enable_thinking,
                )
                is_correct = pred_letter == correct_letter

                writer.writerow(
                    [
                        image_path,
                        meta["second_object"],
                        meta["reference_object"],
                        meta["original_relation"],
                        meta["face_view"],
                        prompt,
                        meta["correct_relation"],
                        correct_letter,
                        opposite_relation,
                        opposite_letter,
                        raw_answer,
                        pred_letter,
                        is_correct,
                    ]
                )
                f.flush()
            except Exception as exc:
                failures += 1
                record_error(out_csv, image_path, exc)
                print(f"Failed: {image_path}: {type(exc).__name__}: {exc}")

    print(f"Saved: {out_csv}")
    if failures:
        raise RuntimeError(
            f"{failures} sample(s) failed; rerun with --resume. "
            f"Details: {out_csv.with_name(f'{out_csv.stem}_errors.jsonl')}"
        )


if __name__ == "__main__":
    main()
