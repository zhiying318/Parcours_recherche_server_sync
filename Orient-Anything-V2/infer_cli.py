#!/usr/bin/env python3
"""Run Orient Anything V2 absolute-orientation inference on one image."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from utils.app_utils import background_preprocess, inf_single_case
from utils.paths import LOCAL_CKPT_PATH
from vision_tower import VGGT_OriAny_Ref


EXPECTED_CHECKPOINT_BYTES = 5_048_116_892


def parse_args():
    parser = argparse.ArgumentParser(
        description="Predict the absolute 3D orientation of a single object/image crop."
    )
    parser.add_argument("image", type=Path, help="Input image containing one main object")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path(LOCAL_CKPT_PATH), help="Model checkpoint"
    )
    parser.add_argument(
        "--remove-background",
        action="store_true",
        help="Use rembg before inference (may download its own model on first use)",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path")
    return parser.parse_args()


def require_file(path: Path, label: str):
    if not path.is_file():
        raise SystemExit(f"{label} not found: {path}")


def scalar(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    if hasattr(value, "item"):
        return value.item()
    return value


def main():
    args = parse_args()
    require_file(args.image, "Input image")
    require_file(args.checkpoint, "Checkpoint")

    checkpoint_bytes = args.checkpoint.stat().st_size
    if checkpoint_bytes != EXPECTED_CHECKPOINT_BYTES:
        raise SystemExit(
            f"Checkpoint is incomplete or unexpected: {checkpoint_bytes} bytes; "
            f"expected {EXPECTED_CHECKPOINT_BYTES} bytes."
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable. Run this command inside a GPU job.")

    capability = torch.cuda.get_device_capability()
    dtype = torch.bfloat16 if capability[0] >= 8 else torch.float16
    device = torch.device("cuda:0")

    model = VGGT_OriAny_Ref(out_dim=900, dtype=dtype, nopretrain=True)
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval().to(device)

    image = Image.open(args.image).convert("RGB")
    if args.remove_background:
        image = background_preprocess(image, True)
    prediction = inf_single_case(model, image, None)

    result = {
        "image": str(args.image.resolve()),
        "checkpoint": str(args.checkpoint.resolve()),
        "device": torch.cuda.get_device_name(0),
        "dtype": str(dtype),
        "azimuth_deg": float(scalar(prediction["ref_az_pred"])),
        "elevation_deg": float(scalar(prediction["ref_el_pred"])),
        "in_plane_rotation_deg": float(scalar(prediction["ref_ro_pred"])),
        "num_front_directions": int(scalar(prediction["ref_alpha_pred"])),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
