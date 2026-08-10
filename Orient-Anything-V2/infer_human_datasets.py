#!/usr/bin/env python3
"""Dataset-aware human orientation inference with Orient Anything V2."""

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import torch
from PIL import Image

from utils.app_utils import inf_single_case
from utils.paths import LOCAL_CKPT_PATH
from vision_tower import VGGT_OriAny_Ref


EXPECTED_CHECKPOINT_BYTES = 5_048_116_892


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=("comfort", "whatsup"))
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path(LOCAL_CKPT_PATH))
    parser.add_argument(
        "--full-image",
        action="store_true",
        help="Disable the default center crop intended to make the human the principal subject.",
    )
    return parser.parse_args()


def comfort_samples(root):
    samples = []
    for image_path in sorted(root.rglob("*.png")):
        config_path = image_path.parent / "config.json"
        if not config_path.is_file():
            raise SystemExit(f"Missing config for {image_path}: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        samples.append(
            {
                "image_path": image_path,
                "dataset": "COMFORT",
                "object_name": config["object_name"],
                "relation": config["relation"],
                "view_type": config["view_name"],
                "expected_human_view": config["view_name"].removeprefix("cam_"),
                "expected_direction": "",
                "sampled_ref_yaw_deg": config.get("sampled_ref_yaw_deg", ""),
                "config_path": str(config_path),
                "prompt": "",
            }
        )
    return samples


def whatsup_samples(root):
    metadata_path = root / "meta_v0.csv"
    if not metadata_path.is_file():
        raise SystemExit(f"Metadata CSV not found: {metadata_path}")
    samples = []
    with metadata_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            # The CSV was created on another machine, so resolve dst_path by basename.
            image_path = root / Path(row["dst_path"]).name
            if not image_path.is_file():
                raise SystemExit(f"Image listed in metadata is missing: {image_path}")
            samples.append(
                {
                    "image_path": image_path,
                    "dataset": "WhatsUp-Qwen-Image-Edit-2509-v0",
                    "object_name": row["second_object"],
                    "relation": "left_of_chair" if "_left_of_chair_" in image_path.name else "right_of_chair",
                    "view_type": row["view_type"],
                    "expected_human_view": row["view_type"].removeprefix("FACE-").lower(),
                    "expected_direction": row["expected_direction"],
                    "sampled_ref_yaw_deg": "",
                    "config_path": str(metadata_path),
                    "prompt": row["prompt"],
                }
            )
    return samples


def load_model(checkpoint):
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")
    if checkpoint.stat().st_size != EXPECTED_CHECKPOINT_BYTES:
        raise SystemExit(f"Checkpoint has unexpected size: {checkpoint.stat().st_size}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Run this script in the GPU Docker container.")
    capability = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
    model = VGGT_OriAny_Ref(out_dim=900, dtype=dtype, nopretrain=True)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval().to("cuda:0")
    return model, dtype


def human_crop(image, dataset):
    width, height = image.size
    # Both datasets place the human near the image center. COMFORT needs a tighter
    # crop because its reference object can be much larger than the person.
    ratios = (0.28, 0.12, 0.72, 0.82) if dataset == "comfort" else (0.22, 0.02, 0.78, 0.99)
    box = tuple(round(value * size) for value, size in zip(ratios, (width, height, width, height)))
    return image.crop(box), box


def scalar(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    return value.item() if hasattr(value, "item") else value


def main():
    args = parse_args()
    if not args.data_root.is_dir():
        raise SystemExit(f"Dataset directory not found: {args.data_root}")
    samples = comfort_samples(args.data_root) if args.dataset == "comfort" else whatsup_samples(args.data_root)
    if not samples:
        raise SystemExit(f"No samples found in {args.data_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    crop_dir = args.output_dir / "human_crops"
    if not args.full_image:
        crop_dir.mkdir(exist_ok=True)
    model, dtype = load_model(args.checkpoint)
    results = []

    for index, sample in enumerate(samples, 1):
        image_path = sample.pop("image_path")
        print(f"[{index}/{len(samples)}] {image_path}", flush=True)
        base = {"image": str(image_path), **sample}
        try:
            image = Image.open(image_path).convert("RGB")
            crop_box = "full"
            inference_image = image
            inference_path = image_path
            if not args.full_image:
                inference_image, box = human_crop(image, args.dataset)
                crop_box = ",".join(map(str, box))
                inference_path = crop_dir / f"{index:04d}_{image_path.name}"
                inference_image.save(inference_path)
            started = time.perf_counter()
            prediction = inf_single_case(model, inference_image, None)
            torch.cuda.synchronize()
            results.append(
                {
                    **base,
                    "inference_image": str(inference_path),
                    "crop_box_xyxy": crop_box,
                    "azimuth_deg": float(scalar(prediction["ref_az_pred"])),
                    "elevation_deg": float(scalar(prediction["ref_el_pred"])),
                    "in_plane_rotation_deg": float(scalar(prediction["ref_ro_pred"])),
                    "num_front_directions": int(scalar(prediction["ref_alpha_pred"])),
                    "inference_seconds": round(time.perf_counter() - started, 4),
                    "status": "ok",
                    "error": "",
                }
            )
        except Exception as error:
            results.append(
                {
                    **base,
                    "inference_image": "",
                    "crop_box_xyxy": "",
                    "azimuth_deg": "",
                    "elevation_deg": "",
                    "in_plane_rotation_deg": "",
                    "num_front_directions": "",
                    "inference_seconds": "",
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                }
            )
            torch.cuda.empty_cache()

    metadata = {
        "model": "Orient Anything V2",
        "target": "human",
        "dataset": args.dataset,
        "data_root": str(args.data_root.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "official_inference_function": "utils.app_utils.inf_single_case",
        "input_mode": "full_image" if args.full_image else "fixed_human_center_crop",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": str(dtype),
        "num_images": len(results),
        "num_success": sum(row["status"] == "ok" for row in results),
    }
    (args.output_dir / "results.json").write_text(
        json.dumps({"metadata": metadata, "results": results}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    if metadata["num_success"] != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
