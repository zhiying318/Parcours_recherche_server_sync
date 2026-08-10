"""Validate a rendered COMFORT geometry-GT dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


REQUIRED_FILES = {
    "0.png",
    "rgb.png",
    "person_mask.png",
    "object_mask.png",
    "depth.npy",
    "camera.json",
    "scene_gt.json",
    "config.json",
}


def validate_dataset(root: Path, expected_samples: int) -> dict:
    scene_paths = sorted(root.rglob("scene_gt.json"))
    failures = []
    relations = Counter()
    objects = Counter()
    min_margin = float("inf")

    if len(scene_paths) != expected_samples:
        failures.append(f"Expected {expected_samples} samples, found {len(scene_paths)}")

    raw_outputs = list(root.rglob("*_raw_*"))
    if raw_outputs:
        failures.append(f"Found {len(raw_outputs)} unnormalized raw render passes")

    for scene_path in scene_paths:
        sample_dir = scene_path.parent
        try:
            missing = sorted(name for name in REQUIRED_FILES if not (sample_dir / name).is_file())
            if missing:
                raise ValueError(f"missing files: {missing}")

            rgb = cv2.imread(str(sample_dir / "rgb.png"), cv2.IMREAD_COLOR)
            person = cv2.imread(str(sample_dir / "person_mask.png"), cv2.IMREAD_GRAYSCALE)
            target = cv2.imread(str(sample_dir / "object_mask.png"), cv2.IMREAD_GRAYSCALE)
            depth = np.load(sample_dir / "depth.npy")
            if rgb is None or person is None or target is None:
                raise ValueError("could not read RGB or masks")
            if rgb.shape[:2] != person.shape or person.shape != target.shape or target.shape != depth.shape:
                raise ValueError("RGB, mask, and depth shapes differ")
            if set(np.unique(person)) - {0, 255} or set(np.unique(target)) - {0, 255}:
                raise ValueError("masks are not binary 0/255 images")
            person_bool = person > 0
            target_bool = target > 0
            if not np.any(person_bool) or not np.any(target_bool):
                raise ValueError("empty person or object mask")
            if np.any(person_bool & target_bool):
                raise ValueError("person and object masks overlap")
            if not np.isfinite(depth[person_bool]).all() or not np.isfinite(depth[target_bool]).all():
                raise ValueError("masked depth contains non-finite values")

            camera = json.loads((sample_dir / "camera.json").read_text(encoding="utf-8"))
            scene = json.loads(scene_path.read_text(encoding="utf-8"))
            world_to_camera = np.asarray(camera["world_to_camera_opencv"], dtype=np.float64)
            camera_to_world = np.asarray(camera["camera_to_world_opencv"], dtype=np.float64)
            rotation = np.asarray(
                scene["camera_to_human_transform"]["rotation"], dtype=np.float64
            )
            if not np.allclose(world_to_camera @ camera_to_world, np.eye(4), atol=1e-6):
                raise ValueError("camera transforms are not inverses")
            if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
                raise ValueError("human rotation is not orthonormal")
            if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6):
                raise ValueError("human rotation determinant is not +1")

            relations[scene["relation"]] += 1
            objects[scene["object_name"]] += 1
            min_margin = min(min_margin, float(scene["classification_margin"]))
        except Exception as exc:
            failures.append(f"{sample_dir}: {exc}")

    expected_relations = {"left": 36, "right": 36, "front": 36, "back": 36}
    if dict(relations) != expected_relations:
        failures.append(f"Unexpected relation distribution: {dict(relations)}")
    if objects and set(objects.values()) != {16}:
        failures.append(f"Each object must have 16 samples: {dict(objects)}")

    return {
        "dataset_root": str(root.resolve()),
        "expected_samples": expected_samples,
        "total_samples": len(scene_paths),
        "valid_samples": len(scene_paths) - sum("Expected " not in item and "Found " not in item and "Unexpected " not in item and "Each object " not in item for item in failures),
        "invalid_samples": sum("Expected " not in item and "Found " not in item and "Unexpected " not in item and "Each object " not in item for item in failures),
        "relation_distribution": dict(sorted(relations.items())),
        "object_distribution": dict(sorted(objects.items())),
        "minimum_classification_margin": None if not scene_paths else min_margin,
        "raw_outputs": len(raw_outputs),
        "failures": failures,
        "passed": not failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--expected_samples", type=int, default=144)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate_dataset(args.dataset_root, args.expected_samples)
    report_path = args.report or args.dataset_root / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
