#!/usr/bin/env python3
"""Free-generation evaluation for the independent SFT variants."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any

import torch

from ..core.geometry import classify_projections
from ..stage1_fixed_template.dataset_pipeline import OPTION_TEXT, format_user_question

REPO_ROOT = Path(__file__).resolve().parents[2]
NUMBER = r"[-+]?\d+(?:\.\d+)?"
VECTOR = rf"\[\s*{NUMBER}(?:\s*,\s*{NUMBER}){{2}}\s*\]"


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _vector(text: str, label: str):
    match = re.search(label + rf"\s*({VECTOR})", text, re.I)
    if not match:
        return None
    return tuple(float(x) for x in re.findall(NUMBER, match.group(1)))


def parse_final_letter(text: str) -> str | None:
    """Parse only a final answer marker/standalone line, never direction words."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        match = re.fullmatch(r"(?:answer\s*[:\-]?\s*)?([ABCD])(?:[.)])?", line, re.I)
        if match:
            return match.group(1).upper()
    matches = list(re.finditer(r"(?:final\s+answer|answer)\s*[:\-]\s*([ABCD])\b", text, re.I))
    return matches[-1].group(1).upper() if matches else None


def _norm(v):
    return math.sqrt(sum(x * x for x in v))


def _angle_degrees(a, b):
    na, nb = _norm(a), _norm(b)
    if na == 0 or nb == 0:
        return None
    cosine = max(-1.0, min(1.0, sum(x * y for x, y in zip(a, b)) / (na * nb)))
    return math.degrees(math.acos(cosine))


def parse_geometry(text: str, target: dict[str, Any] | None) -> dict[str, Any]:
    result = {"parsed": False}
    if target is None:
        return result
    # The fixed opening sentence contains the camera origin as the first
    # vector. Exclude it before reading person, object, displacement, forward
    # and right; otherwise every geometry metric is shifted by one vector.
    vectors = [tuple(float(x) for x in re.findall(NUMBER, match)) for match in re.findall(VECTOR, text)]
    if vectors and vectors[0] == (0.0, 0.0, 0.0):
        vectors = vectors[1:]
    if len(vectors) >= 5:
        person, obj, displacement, forward, right = vectors[:5]
    else:
        displacement = forward = right = None
    projection_match = re.search(rf"projects approximately\s*({NUMBER}).*?and\s*({NUMBER})\s+onto the right axis", text, re.I | re.S)
    if not (person and obj and displacement and forward and right and projection_match):
        result["reason"] = "template fields missing"
        return result
    f_proj, r_proj = float(projection_match.group(1)), float(projection_match.group(2))
    geometry = target
    gt_person = tuple(geometry["person_position"])
    gt_obj = tuple(geometry["object_position"])
    gt_disp = tuple(geometry["relative_vector"])
    gt_forward = tuple(geometry["person_forward"])
    gt_right = tuple(geometry["person_right"])
    predicted_disp = tuple(obj[i] - person[i] for i in range(3))
    consistency_f = sum(predicted_disp[i] * forward[i] for i in range(3))
    consistency_r = sum(predicted_disp[i] * right[i] for i in range(3))
    try:
        predicted_relation = classify_projections(consistency_f, consistency_r)
    except ValueError:
        predicted_relation = None
    result.update({
        "parsed": True,
        "person_position_l2": _norm(tuple(person[i] - gt_person[i] for i in range(3))),
        "object_position_l2": _norm(tuple(obj[i] - gt_obj[i] for i in range(3))),
        "relative_displacement_l2": _norm(tuple(displacement[i] - gt_disp[i] for i in range(3))),
        "forward_axis_angle_degrees": _angle_degrees(forward, gt_forward),
        "right_axis_angle_degrees": _angle_degrees(right, gt_right),
        "forward_projection_error": abs(f_proj - sum(displacement[i] * forward[i] for i in range(3))),
        "right_projection_error": abs(r_proj - sum(displacement[i] * right[i] for i in range(3))),
        "predicted_relation": predicted_relation,
        "predicted_geometry_final_consistent": predicted_relation is not None,
        "predicted_projection_forward": consistency_f,
        "predicted_projection_right": consistency_r,
    })
    return result


def _model_device(model):
    return next(model.parameters()).device


@torch.inference_mode()
def generate_one(
    model,
    processor,
    repo_root: Path,
    record: dict[str, Any],
    max_new_tokens: int,
    enable_thinking: bool,
):
    image_path = str(repo_root / record["image_path"])
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": format_user_question(record["question"], record["choices"])},
    ]}]
    inputs = processor.apply_chat_template(
        [messages], tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt", enable_thinking=enable_thinking,
        processor_kwargs={"do_resize": False, "padding": True},
    )
    device = _model_device(model)
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}
    input_len = int(inputs["input_ids"].shape[1])
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    generated = output[:, input_len:]
    text = processor.batch_decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    eos_ids = {x for x in (getattr(processor.tokenizer, "eos_token_id", None),) if x is not None}
    has_eos = any(int(x) in eos_ids for x in generated[0].tolist())
    return text, not has_eos


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("fixed_template_reasoning", "verbalized_reasoning"),
        required=True,
    )
    parser.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--data-jsonl", type=Path, default=None)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32768,
        help="hard generation safety cap; generation may stop earlier at EOS",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="use the Qwen reasoning chat-template mode for a thinking-trained adapter",
    )
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.data_jsonl:
        data_path = args.data_jsonl
    else:
        data_root = (
            REPO_ROOT / "spatial_internalization/datasets/stage2_trajectory_guided/full"
            if args.variant == "verbalized_reasoning"
            else REPO_ROOT / "spatial_internalization/datasets/stage1_fixed_split"
        )
        data_path = data_root / args.variant / "test.jsonl"
    records = load_jsonl(data_path)
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError(
            "spatial_internalization.training.evaluate_internal requires a CUDA GPU for the configured "
            f"attention implementation ({args.attn_implementation!r}); "
            "the model was not loaded on CPU. Run evaluate_docker.sh with a visible GPU."
        )

    if args.model_path is not None:
        from ..core.offline_qwen import resolve_local_model_path

        model_source = str(resolve_local_model_path(args.model_path))
        local_files_only = True
    elif args.model_name == "Qwen/Qwen3.5-9B":
        from ..core.offline_qwen import resolve_local_model_path

        model_source = str(resolve_local_model_path())
        local_files_only = True
    else:
        model_source = args.model_name
        local_files_only = False

    try:
        processor = AutoProcessor.from_pretrained(args.adapter)
    except Exception:
        processor = AutoProcessor.from_pretrained(model_source, local_files_only=local_files_only)
    base = AutoModelForImageTextToText.from_pretrained(
        model_source,
        local_files_only=local_files_only,
        dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
    )
    model = PeftModel.from_pretrained(base, str(args.adapter))
    # Trainer/Accelerate moves the model automatically during SFT, but this
    # standalone generation entry point does not. Explicitly move both the
    # base model and LoRA adapter to the selected CUDA device before the first
    # FlashAttention call.
    model = model.to(torch.device("cuda"))
    model.eval()
    print(f"evaluation device: {next(model.parameters()).device}")
    results = []
    for index, record in enumerate(records):
        try:
            response, truncated = generate_one(
                model, processor, REPO_ROOT, record, args.max_new_tokens, args.enable_thinking
            )
            letter = parse_final_letter(response)
            geometry_target = None
            if args.variant == "fixed_template_reasoning":
                geometry_target = record.get("geometry") or record.get("perturbed_supervision_geometry")
            geometry = parse_geometry(response, geometry_target)
            final_relation = None
            if letter:
                option_text = record.get("choices", {}).get(letter, "")
                final_relation = next((relation for relation, phrase in OPTION_TEXT.items() if phrase in option_text), None)
            results.append({
                "id": record.get("id", str(index)),
                "image_path": record["image_path"],
                "raw_response": response,
                "predicted_letter": letter,
                "target_letter": record.get("answer_letter"),
                "relation": record.get("relation"),
                "correct": letter == record.get("answer_letter") if record.get("answer_letter") else None,
                "invalid_answer": letter is None,
                "truncated": truncated,
                "output_chars": len(response),
                "geometry": geometry,
                "final_relation_from_option": final_relation,
            })
        except Exception as exc:
            results.append({"id": record.get("id", str(index)), "image_path": record["image_path"], "error": repr(exc), "invalid_answer": True})
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
    summary = summarize(results)
    (args.output_jsonl.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    labelled = [r for r in results if r.get("target_letter")]
    correct = [r for r in labelled if r.get("correct")]
    confusion = {gt: {pred: 0 for pred in "ABCD"} for gt in "ABCD"}
    for row in labelled:
        pred = row.get("predicted_letter")
        if pred in confusion[row["target_letter"]]:
            confusion[row["target_letter"]][pred] += 1
    relation_counts = {relation: {"correct": 0, "total": 0} for relation in ("front", "behind", "left", "right")}
    for row in labelled:
        relation = row.get("relation")
        if relation in relation_counts:
            relation_counts[relation]["total"] += 1
            relation_counts[relation]["correct"] += int(bool(row.get("correct")))
    parsed = [r["geometry"] for r in results if r.get("geometry", {}).get("parsed")]
    return {
        "count": len(results),
        "labelled_count": len(labelled),
        "accuracy": len(correct) / len(labelled) if labelled else None,
        "invalid_answer_rate": sum(bool(r.get("invalid_answer")) for r in results) / len(results) if results else None,
        "truncation_rate": sum(bool(r.get("truncated")) for r in results) / len(results) if results else None,
        "mean_output_chars": sum(r.get("output_chars", 0) for r in results) / len(results) if results else 0,
        "per_relation": relation_counts,
        "confusion_matrix_by_letter": confusion,
        "geometry_parse_count": len(parsed),
        "geometry_parse_failure_count": sum(not r.get("geometry", {}).get("parsed", False) for r in results),
        "geometry_metrics": {
            key: sum(float(x[key]) for x in parsed if x.get(key) is not None) / sum(x.get(key) is not None for x in parsed)
            for key in ("person_position_l2", "object_position_l2", "relative_displacement_l2", "forward_axis_angle_degrees", "right_axis_angle_degrees", "forward_projection_error", "right_projection_error")
            if any(x.get(key) is not None for x in parsed)
        },
        "geometry_relation_accuracy": sum(x.get("predicted_relation") == r.get("relation") for r in results for x in [r.get("geometry", {})] if x.get("predicted_relation") and r.get("relation")) / sum(bool(r.get("geometry", {}).get("predicted_relation")) for r in results) if any(r.get("geometry", {}).get("predicted_relation") for r in results) else None,
        "geometry_final_answer_consistency": sum(r.get("final_relation_from_option") == r.get("geometry", {}).get("predicted_relation") for r in results if r.get("geometry", {}).get("predicted_relation") is not None) / sum(r.get("geometry", {}).get("predicted_relation") is not None for r in results) if any(r.get("geometry", {}).get("predicted_relation") is not None for r in results) else None,
    }


if __name__ == "__main__":
    main()
