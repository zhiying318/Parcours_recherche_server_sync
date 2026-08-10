#!/usr/bin/env python3
"""Batch absolute-orientation inference using the official Orient Anything V2 core."""

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import torch
from PIL import Image

from utils.app_utils import background_preprocess, inf_single_case
from utils.paths import LOCAL_CKPT_PATH
from vision_tower import VGGT_OriAny_Ref


EXPECTED_CHECKPOINT_BYTES = 5_048_116_892
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the official Orient Anything V2 inference core on files/directories."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Image files or directories")
    parser.add_argument("--checkpoint", type=Path, default=Path(LOCAL_CKPT_PATH))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/coco_smoke"))
    parser.add_argument("--remove-background", action="store_true")
    parser.add_argument("--recursive", action="store_true")
    return parser.parse_args()


def discover_images(inputs, recursive):
    images = []
    for item in inputs:
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES:
            images.append(item)
        elif item.is_dir():
            iterator = item.rglob("*") if recursive else item.glob("*")
            images.extend(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        else:
            raise SystemExit(f"Input does not exist or is not a supported image: {item}")
    unique = sorted({path.resolve() for path in images})
    if not unique:
        raise SystemExit("No supported images found.")
    return unique


def scalar(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    if hasattr(value, "item"):
        return value.item()
    return value


def load_model(checkpoint):
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    size = checkpoint.stat().st_size
    if size != EXPECTED_CHECKPOINT_BYTES:
        raise SystemExit(f"Unexpected checkpoint size: {size}; expected {EXPECTED_CHECKPOINT_BYTES}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Run inside a GPU job.")

    capability = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
    model = VGGT_OriAny_Ref(out_dim=900, dtype=dtype, nopretrain=True)
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval().to("cuda:0")
    return model, dtype


def predict(model, image_path, remove_background):
    image = Image.open(image_path).convert("RGB")
    if remove_background:
        image = background_preprocess(image, True)
    started = time.perf_counter()
    prediction = inf_single_case(model, image, None)
    torch.cuda.synchronize()
    return {
        "image": str(image_path),
        "azimuth_deg": float(scalar(prediction["ref_az_pred"])),
        "elevation_deg": float(scalar(prediction["ref_el_pred"])),
        "in_plane_rotation_deg": float(scalar(prediction["ref_ro_pred"])),
        "num_front_directions": int(scalar(prediction["ref_alpha_pred"])),
        "inference_seconds": round(time.perf_counter() - started, 4),
        "status": "ok",
        "error": "",
    }


def main():
    args = parse_args()
    images = discover_images(args.inputs, args.recursive)
    model, dtype = load_model(args.checkpoint)

    results = []
    for index, image_path in enumerate(images, start=1):
        print(f"[{index}/{len(images)}] {image_path}", flush=True)
        try:
            results.append(predict(model, image_path, args.remove_background))
        except Exception as error:
            results.append(
                {
                    "image": str(image_path),
                    "azimuth_deg": None,
                    "elevation_deg": None,
                    "in_plane_rotation_deg": None,
                    "num_front_directions": None,
                    "inference_seconds": None,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            torch.cuda.empty_cache()

    metadata = {
        "model": "Orient Anything V2",
        "checkpoint": str(args.checkpoint.resolve()),
        "official_inference_function": "utils.app_utils.inf_single_case",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "dtype": str(dtype),
        "remove_background": args.remove_background,
        "num_images": len(images),
        "num_success": sum(item["status"] == "ok" for item in results),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps({"metadata": metadata, "results": results}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    print(json.dumps(metadata, indent=2), flush=True)
    if metadata["num_success"] != len(images):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
