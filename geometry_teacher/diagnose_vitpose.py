"""Compare ViTPose post-processing for absolute COCO boxes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def serialize_result(result: dict) -> dict:
    return {
        key: value.detach().cpu().numpy().tolist()
        for key, value in result.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--person-box-xyxy", type=float, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--memory-fraction", type=float, default=0.05)
    args = parser.parse_args()

    import torch
    from transformers import AutoImageProcessor, VitPoseForPoseEstimation

    torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device=0)
    free_before, total = torch.cuda.mem_get_info(0)
    if free_before < 8 * 1024**3:
        raise RuntimeError(f"Refusing to run with less than 8 GiB free: {free_before / 1024**3:.2f} GiB")

    image = Image.open(args.image).convert("RGB")
    box_xyxy = np.asarray(args.person_box_xyxy, dtype=np.float32)
    box_xywh = box_xyxy.copy()
    box_xywh[2:] -= box_xywh[:2]
    boxes = [box_xywh[None, :]]

    processor = AutoImageProcessor.from_pretrained(
        "usyd-community/vitpose-base", local_files_only=True
    )
    model = VitPoseForPoseEstimation.from_pretrained(
        "usyd-community/vitpose-base", local_files_only=True
    ).to("cuda").eval()
    inputs = processor(image, boxes=boxes, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        outputs = model(**inputs)

    absolute_result = processor.post_process_pose_estimation(outputs, boxes=boxes)[0][0]
    scaled_again_result = processor.post_process_pose_estimation(
        outputs, boxes=boxes, target_sizes=[(image.height, image.width)]
    )[0][0]
    report = {
        "image_size_wh": list(image.size),
        "input_box_xyxy_absolute": box_xyxy.tolist(),
        "processor_box_xywh_absolute": box_xywh.tolist(),
        "heatmap_shape": list(outputs.heatmaps.shape),
        "without_target_sizes": serialize_result(absolute_result),
        "with_target_sizes_current_adapter": serialize_result(scaled_again_result),
        "peak_memory_allocated_mib": round(torch.cuda.max_memory_allocated(0) / 1024**2),
        "peak_memory_reserved_mib": round(torch.cuda.max_memory_reserved(0) / 1024**2),
        "total_memory_mib": round(total / 1024**2),
        "free_memory_before_mib": round(free_before / 1024**2),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
