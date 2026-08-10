"""Command-line entry point for the single-image geometry teacher."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from geometry_teacher.pipeline import run_geometry_teacher
from geometry_teacher.reasoning import (
    RELATIONS,
    generate_reasoning_trace,
    validate_reasoning_trace,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Geometry-grounded teacher pipeline")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--object", dest="object_name", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vitpose_model_id", default="usyd-community/vitpose-base")
    parser.add_argument("--expected_answer", choices=RELATIONS)
    args = parser.parse_args()

    memory_fraction = os.environ.get("GEOMETRY_CUDA_MEMORY_FRACTION")
    torch_module = None
    memory_device = None
    if args.device.startswith("cuda"):
        import torch

        torch_module = torch
        memory_device = 0 if args.device == "cuda" else args.device
    if memory_fraction and args.device.startswith("cuda"):
        fraction = float(memory_fraction)
        if not 0.0 < fraction <= 1.0:
            raise ValueError("GEOMETRY_CUDA_MEMORY_FRACTION must be in (0, 1]")
        torch_module.cuda.set_per_process_memory_fraction(fraction, device=memory_device)
    if torch_module is not None:
        torch_module.cuda.reset_peak_memory_stats(memory_device)

    primitive, artifacts = run_geometry_teacher(
        image_path=args.image,
        object_name=args.object_name,
        device=args.device,
        vitpose_model_id=args.vitpose_model_id,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "primitive.json").write_text(
        json.dumps({"image": str(args.image), "primitive": primitive}, indent=2) + "\n",
        encoding="utf-8",
    )
    trace = generate_reasoning_trace(primitive)
    (args.output_dir / "trace.json").write_text(
        json.dumps({"image": str(args.image), **trace}, indent=2) + "\n",
        encoding="utf-8",
    )
    validation = validate_reasoning_trace(primitive, trace, args.expected_answer)
    if torch_module is not None:
        validation["runtime"] = {
            "device": args.device,
            "cuda_memory_fraction_limit": float(memory_fraction) if memory_fraction else None,
            "peak_cuda_memory_allocated_mib": round(
                torch_module.cuda.max_memory_allocated(memory_device) / 1024**2
            ),
            "peak_cuda_memory_reserved_mib": round(
                torch_module.cuda.max_memory_reserved(memory_device) / 1024**2
            ),
        }
    (args.output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n",
        encoding="utf-8",
    )
    cv2.imwrite(
        str(args.output_dir / "person_mask.png"), artifacts["person_mask"].astype(np.uint8) * 255
    )
    cv2.imwrite(
        str(args.output_dir / "object_mask.png"), artifacts["object_mask"].astype(np.uint8) * 255
    )
    np.savez_compressed(
        args.output_dir / "geometry.npz",
        depth=artifacts["depth"],
        depth_confidence=artifacts["depth_confidence"],
        point_map_camera=artifacts["point_map_camera"],
        intrinsic_processed=artifacts["intrinsic_processed"],
        keypoints=artifacts["keypoints"],
        person_mask_processed=artifacts["person_mask_processed"],
        object_mask_processed=artifacts["object_mask_processed"],
    )
    if not validation["valid"]:
        raise RuntimeError("Generated trace failed validation; see validation.json")


if __name__ == "__main__":
    main()
