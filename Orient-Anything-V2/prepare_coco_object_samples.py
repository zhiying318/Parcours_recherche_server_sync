#!/usr/bin/env python3
"""Prepare eight visually reviewed directional-object crops from COCO val2017."""

import argparse
import json
import shutil
import urllib.request
from pathlib import Path

from PIL import Image


SAMPLES = [
    {"image_id": 370208, "annotation_id": 125303, "category": "bicycle", "view": "side"},
    {"image_id": 263594, "annotation_id": 137053, "category": "car", "view": "side"},
    {"image_id": 144300, "annotation_id": 151726, "category": "motorcycle", "view": "side"},
    {"image_id": 137950, "annotation_id": 157562, "category": "airplane", "view": "front"},
    {"image_id": 311909, "annotation_id": 166445, "category": "bus", "view": "three_quarter"},
    {"image_id": 287874, "annotation_id": 170385, "category": "train", "view": "front"},
    {"image_id": 220732, "annotation_id": 395730, "category": "truck", "view": "side"},
    {"image_id": 365098, "annotation_id": 177900, "category": "boat", "view": "side"},
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("test_data/coco_object_samples"))
    parser.add_argument(
        "--annotations",
        type=Path,
        default=Path("test_data/coco_person_samples/annotations/instances_val2017.json"),
    )
    parser.add_argument("--padding", type=float, default=0.12)
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
    if not args.annotations.is_file():
        raise SystemExit(f"Missing COCO annotations: {args.annotations}")

    data = json.loads(args.annotations.read_text(encoding="utf-8"))
    images = {item["id"]: item for item in data["images"]}
    annotations = {item["id"]: item for item in data["annotations"]}
    categories = {item["id"]: item for item in data["categories"]}
    licenses = {item["id"]: item for item in data["licenses"]}

    candidates_dir = args.root / "candidates"
    originals_dir = args.root / "originals"
    crops_dir = args.root / "object_crops"
    for directory in (candidates_dir, originals_dir, crops_dir):
        directory.mkdir(parents=True, exist_ok=True)
    for stale_path in (*originals_dir.glob("sample_*.jpg"), *crops_dir.glob("sample_*.jpg")):
        stale_path.unlink()

    manifest = []
    for index, sample in enumerate(SAMPLES, start=1):
        image_info = images[sample["image_id"]]
        annotation = annotations[sample["annotation_id"]]
        category = categories[annotation["category_id"]]
        license_info = licenses[image_info["license"]]
        if annotation["image_id"] != sample["image_id"] or category["name"] != sample["category"]:
            raise RuntimeError(f"Invalid object annotation mapping: {sample}")

        file_name = image_info["file_name"]
        candidate_path = candidates_dir / file_name
        if not candidate_path.is_file():
            urllib.request.urlretrieve(
                f"http://images.cocodataset.org/val2017/{file_name}", candidate_path
            )

        original_name = f"sample_{index:02d}_{sample['category']}_coco_{sample['image_id']:012d}.jpg"
        original_path = originals_dir / original_name
        shutil.copyfile(candidate_path, original_path)

        with Image.open(candidate_path) as image:
            image = image.convert("RGB")
            crop_box = padded_box(annotation["bbox"], image.size, args.padding)
            crop = image.crop(crop_box)
            crop_name = f"sample_{index:02d}_{sample['category']}_ann_{sample['annotation_id']}.jpg"
            crop_path = crops_dir / crop_name
            crop.save(crop_path, quality=95)

        manifest.append(
            {
                "sample_index": index,
                "split": "val2017",
                "category": sample["category"],
                "coco_image_id": sample["image_id"],
                "coco_annotation_id": sample["annotation_id"],
                "source_url": f"http://images.cocodataset.org/val2017/{file_name}",
                "coco_license_id": license_info["id"],
                "coco_license_name": license_info["name"],
                "coco_license_url": license_info["url"],
                "original_image": str(original_path.relative_to(args.root)),
                "object_crop": str(crop_path.relative_to(args.root)),
                "coco_bbox_xywh": annotation["bbox"],
                "crop_box_xyxy": crop_box,
                "manual_view_hint": sample["view"],
                "manual_view_hint_is_ground_truth": False,
            }
        )

    (args.root / "samples.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Prepared {len(manifest)} object samples under {args.root}")


if __name__ == "__main__":
    main()
