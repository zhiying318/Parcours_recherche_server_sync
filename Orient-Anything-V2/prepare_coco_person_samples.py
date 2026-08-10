#!/usr/bin/env python3
"""Prepare a small, visually reviewed COCO val2017 person-orientation set."""

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

from PIL import Image


SAMPLES = [
    {"image_id": 42102, "annotation_id": 1728930, "manual_view_hint": "front"},
    {"image_id": 85772, "annotation_id": 429906, "manual_view_hint": "three_quarter"},
    {"image_id": 130579, "annotation_id": 439080, "manual_view_hint": "side"},
    {"image_id": 140556, "annotation_id": 425491, "manual_view_hint": "back"},
    {"image_id": 325991, "annotation_id": 422442, "manual_view_hint": "side"},
    {"image_id": 362520, "annotation_id": 429411, "manual_view_hint": "back"},
    {"image_id": 400815, "annotation_id": 474691, "manual_view_hint": "front"},
    {"image_id": 449909, "annotation_id": 430546, "manual_view_hint": "back"},
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("test_data/coco_person_samples"),
        help="Sample-set root",
    )
    parser.add_argument(
        "--padding", type=float, default=0.12, help="Fractional padding around each person box"
    )
    return parser.parse_args()


def padded_box(bbox, image_size, padding):
    x, y, width, height = bbox
    pad_x, pad_y = width * padding, height * padding
    image_width, image_height = image_size
    return [
        max(0, round(x - pad_x)),
        max(0, round(y - pad_y)),
        min(image_width, round(x + width + pad_x)),
        min(image_height, round(y + height + pad_y)),
    ]


def main():
    args = parse_args()
    annotation_path = args.root / "annotations" / "instances_val2017.json"
    if not annotation_path.is_file():
        raise SystemExit(f"Missing COCO annotations: {annotation_path}")

    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    images = {item["id"]: item for item in data["images"]}
    annotations = {item["id"]: item for item in data["annotations"]}
    licenses = {item["id"]: item for item in data["licenses"]}

    candidates_dir = args.root / "candidates"
    originals_dir = args.root / "originals"
    crops_dir = args.root / "person_crops"
    originals_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in (*originals_dir.glob("sample_*.jpg"), *crops_dir.glob("sample_*.jpg")):
        stale_path.unlink()

    manifest = []
    for index, sample in enumerate(SAMPLES, start=1):
        image_info = images[sample["image_id"]]
        annotation = annotations[sample["annotation_id"]]
        license_info = licenses[image_info["license"]]
        if annotation["image_id"] != sample["image_id"] or annotation["category_id"] != 1:
            raise RuntimeError(f"Invalid person annotation mapping: {sample}")

        file_name = image_info["file_name"]
        candidate_path = candidates_dir / file_name
        if not candidate_path.is_file():
            url = f"http://images.cocodataset.org/val2017/{file_name}"
            urllib.request.urlretrieve(url, candidate_path)

        original_name = f"sample_{index:02d}_coco_{sample['image_id']:012d}.jpg"
        original_path = originals_dir / original_name
        shutil.copyfile(candidate_path, original_path)

        with Image.open(candidate_path) as image:
            image = image.convert("RGB")
            crop_box = padded_box(annotation["bbox"], image.size, args.padding)
            crop = image.crop(crop_box)
            crop_name = f"sample_{index:02d}_person_ann_{sample['annotation_id']}.jpg"
            crop_path = crops_dir / crop_name
            crop.save(crop_path, quality=95)

        manifest.append(
            {
                "sample_index": index,
                "split": "val2017",
                "coco_image_id": sample["image_id"],
                "coco_annotation_id": sample["annotation_id"],
                "source_url": f"http://images.cocodataset.org/val2017/{file_name}",
                "coco_license_id": license_info["id"],
                "coco_license_name": license_info["name"],
                "coco_license_url": license_info["url"],
                "original_image": str(original_path.relative_to(args.root)),
                "person_crop": str(crop_path.relative_to(args.root)),
                "coco_bbox_xywh": annotation["bbox"],
                "crop_box_xyxy": crop_box,
                "manual_view_hint": sample["manual_view_hint"],
                "manual_view_hint_is_ground_truth": False,
            }
        )

    output_path = args.root / "samples.json"
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(manifest)} samples under {args.root}")


if __name__ == "__main__":
    main()
