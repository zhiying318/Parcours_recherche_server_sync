"""Inspect Grounding DINO candidates without loading any other teacher model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="IDEA-Research/grounding-dino-base")
    parser.add_argument("--memory-fraction", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    if not 0.0 < args.memory_fraction <= 0.1:
        raise ValueError("memory-fraction must be in (0, 0.1]")

    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=0)
    free_before, total = torch.cuda.mem_get_info(0)
    if free_before < 8 * 1024**3:
        raise RuntimeError(f"Refusing to run with less than 8 GiB free: {free_before / 1024**3:.2f} GiB")

    image = Image.open(args.image).convert("RGB")
    processor = AutoProcessor.from_pretrained(args.model_id, local_files_only=True)
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        args.model_id, local_files_only=True
    ).to("cuda").eval()

    prompts = ("person", "person.", "a person.", "human.", "car", "car.")
    report: dict = {
        "model_id": args.model_id,
        "image": str(args.image),
        "image_size_wh": list(image.size),
        "gpu": torch.cuda.get_device_name(0),
        "memory_fraction_limit": args.memory_fraction,
        "total_memory_mib": round(total / 1024**2),
        "free_memory_before_mib": round(free_before / 1024**2),
        "prompts": {},
    }

    for prompt in prompts:
        inputs = processor(images=image, text=prompt, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            outputs = model(**inputs)
        result = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.0,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]],
        )[0]
        scores = result["scores"].detach().cpu()
        boxes = result["boxes"].detach().cpu()
        order = scores.argsort(descending=True)[: args.top_k]
        candidates = [
            {
                "score": float(scores[index]),
                "box_xyxy": [round(float(value), 3) for value in boxes[index]],
            }
            for index in order
        ]
        report["prompts"][prompt] = {
            "top_score": candidates[0]["score"] if candidates else None,
            "count_at_box_threshold_0.35": int((scores >= 0.35).sum()),
            "count_at_box_threshold_0.25": int((scores >= 0.25).sum()),
            "count_at_box_threshold_0.20": int((scores >= 0.20).sum()),
            "top_candidates": candidates,
        }
        print(prompt, json.dumps(report["prompts"][prompt]), flush=True)
        del inputs, outputs, result

    report["peak_memory_allocated_mib"] = round(torch.cuda.max_memory_allocated(0) / 1024**2)
    report["peak_memory_reserved_mib"] = round(torch.cuda.max_memory_reserved(0) / 1024**2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
