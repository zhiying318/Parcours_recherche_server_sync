"""Resumable staged batch runner for the Geometry Teacher pipeline."""

from __future__ import annotations

import argparse
import gc
import json
import os
import traceback
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from geometry_teacher.model_adapters import (
    GroundingDinoAdapter,
    Sam2Adapter,
    VggtAdapter,
    ViTPoseAdapter,
)
from geometry_teacher.pipeline import keypoints_to_vggt_pad, mask_to_vggt_pad
from geometry_teacher.reasoning import generate_reasoning_trace, validate_reasoning_trace
from geometry_teacher.solver import solve_human_object_geometry


REQUIRED_OUTPUTS = {
    "primitive.json",
    "trace.json",
    "validation.json",
    "person_mask.png",
    "object_mask.png",
    "geometry.npz",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Staged Geometry Teacher batch runner")
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--shard_index", type=int, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--keypoint_threshold", type=float, default=0.3)
    args = parser.parse_args()

    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    samples = discover_samples(args.dataset_root)
    samples = [sample for index, sample in enumerate(samples) if index % args.num_shards == args.shard_index]
    args.output_root.mkdir(parents=True, exist_ok=True)
    stage_root = args.output_root / "._stages" / f"shard_{args.shard_index}"
    stage_root.mkdir(parents=True, exist_ok=True)

    torch_module, memory_device, memory_fraction = configure_cuda(args.device)
    errors: list[dict] = []
    pending = [sample for sample in samples if not sample_complete(args.output_root, sample)]
    print(
        f"Shard {args.shard_index}/{args.num_shards}: {len(samples)} assigned, "
        f"{len(pending)} pending",
        flush=True,
    )

    run_detection_stage(pending, stage_root, args.device, errors)
    run_mask_stage(pending, stage_root, args.device, errors)
    run_pose_stage(pending, stage_root, args.device, errors)
    run_geometry_stage(pending, stage_root, args.device, errors)
    for sample in pending:
        try:
            finalize_sample(sample, stage_root, args.output_root, args.keypoint_threshold)
        except Exception as error:
            record_error(errors, sample, "finalize", error)

    report = build_report(args.output_root, samples, errors)
    report["shard_index"] = args.shard_index
    report["num_shards"] = args.num_shards
    report["assigned_samples"] = len(samples)
    if torch_module is not None:
        report["runtime"] = {
            "device": args.device,
            "cuda_memory_fraction_limit": memory_fraction,
            "peak_cuda_memory_allocated_mib": round(
                torch_module.cuda.max_memory_allocated(memory_device) / 1024**2
            ),
            "peak_cuda_memory_reserved_mib": round(
                torch_module.cuda.max_memory_reserved(memory_device) / 1024**2
            ),
        }
    report_path = args.output_root / f"batch_report_shard_{args.shard_index}.json"
    write_json(report_path, report)
    print(json.dumps(report, indent=2), flush=True)
    if report["failed_samples"]:
        raise SystemExit(1)


def configure_cuda(device: str):
    if not device.startswith("cuda"):
        return None, None, None
    import torch

    memory_device = 0 if device == "cuda" else device
    value = os.environ.get("GEOMETRY_CUDA_MEMORY_FRACTION")
    fraction = float(value) if value else None
    if fraction is not None:
        if not 0.0 < fraction <= 1.0:
            raise ValueError("GEOMETRY_CUDA_MEMORY_FRACTION must be in (0, 1]")
        torch.cuda.set_per_process_memory_fraction(fraction, device=memory_device)
    torch.cuda.reset_peak_memory_stats(memory_device)
    return torch, memory_device, fraction


def discover_samples(dataset_root: Path) -> list[dict]:
    samples = []
    for image_path in sorted(dataset_root.glob("*/*/rgb.png")):
        sample_dir = image_path.parent
        scene_gt = json.loads((sample_dir / "scene_gt.json").read_text(encoding="utf-8"))
        relation = scene_gt["relation"]
        object_name = scene_gt["object_name"]
        if relation not in {"left", "right", "front", "back"}:
            raise ValueError(f"Unknown relation {relation!r} in {sample_dir}")
        samples.append(
            {
                "image_path": image_path,
                "relative_image": str(image_path.relative_to(dataset_root)),
                "sample_id": f"{sample_dir.parent.name}/{sample_dir.name}",
                "relation": relation,
                "object_name": object_name,
            }
        )
    if not samples:
        raise ValueError(f"No samples found below {dataset_root}")
    return samples


def sample_paths(root: Path, sample: dict) -> Path:
    return root / sample["sample_id"]


def sample_complete(output_root: Path, sample: dict) -> bool:
    root = sample_paths(output_root, sample)
    if not root.is_dir() or not REQUIRED_OUTPUTS.issubset({item.name for item in root.iterdir()}):
        return False
    try:
        return bool(json.loads((root / "validation.json").read_text(encoding="utf-8"))["valid"])
    except (OSError, KeyError, json.JSONDecodeError):
        return False


def run_detection_stage(samples: list[dict], stage_root: Path, device: str, errors: list[dict]) -> None:
    todo = [sample for sample in samples if not (sample_paths(stage_root, sample) / "detections.json").is_file()]
    if not todo:
        return
    print(f"Detection stage: {len(todo)} samples", flush=True)
    model = GroundingDinoAdapter(device=device)
    try:
        for index, sample in enumerate(todo, 1):
            try:
                person = model.detect_one(sample["image_path"], "person")
                obj = model.detect_one(sample["image_path"], sample["object_name"])
                write_json(
                    sample_paths(stage_root, sample) / "detections.json",
                    {"person": json_detection(person), "object": json_detection(obj)},
                )
                print(f"  detection {index}/{len(todo)} {sample['sample_id']}", flush=True)
            except Exception as error:
                record_error(errors, sample, "detection", error)
    finally:
        model.close()


def run_mask_stage(samples: list[dict], stage_root: Path, device: str, errors: list[dict]) -> None:
    todo = [
        sample for sample in samples
        if (sample_paths(stage_root, sample) / "detections.json").is_file()
        and not (sample_paths(stage_root, sample) / "masks.npz").is_file()
    ]
    if not todo:
        return
    print(f"Segmentation stage: {len(todo)} samples", flush=True)
    model = Sam2Adapter(device=device)
    try:
        for index, sample in enumerate(todo, 1):
            try:
                detection = read_json(sample_paths(stage_root, sample) / "detections.json")
                person = model.segment_box(sample["image_path"], np.asarray(detection["person"]["box_xyxy"]))
                obj = model.segment_box(sample["image_path"], np.asarray(detection["object"]["box_xyxy"]))
                if np.any(person & obj):
                    raise ValueError("Selected person and object masks overlap")
                write_npz(sample_paths(stage_root, sample) / "masks.npz", person=person, object=obj)
                print(f"  segmentation {index}/{len(todo)} {sample['sample_id']}", flush=True)
            except Exception as error:
                record_error(errors, sample, "segmentation", error)
    finally:
        model.close()


def run_pose_stage(samples: list[dict], stage_root: Path, device: str, errors: list[dict]) -> None:
    todo = [
        sample for sample in samples
        if (sample_paths(stage_root, sample) / "detections.json").is_file()
        and not (sample_paths(stage_root, sample) / "keypoints.npy").is_file()
    ]
    if not todo:
        return
    print(f"Pose stage: {len(todo)} samples", flush=True)
    model = ViTPoseAdapter(device=device)
    try:
        for index, sample in enumerate(todo, 1):
            try:
                detection = read_json(sample_paths(stage_root, sample) / "detections.json")["person"]
                keypoints = model.predict_one(
                    sample["image_path"], np.asarray(detection["box_xyxy"]), detection["score"]
                )
                write_npy(sample_paths(stage_root, sample) / "keypoints.npy", keypoints)
                print(f"  pose {index}/{len(todo)} {sample['sample_id']}", flush=True)
            except Exception as error:
                record_error(errors, sample, "pose", error)
    finally:
        model.close()


def run_geometry_stage(samples: list[dict], stage_root: Path, device: str, errors: list[dict]) -> None:
    todo = [
        sample for sample in samples
        if not (sample_paths(stage_root, sample) / "geometry.npz").is_file()
    ]
    if not todo:
        return
    print(f"VGGT stage: {len(todo)} samples", flush=True)
    model = VggtAdapter(device=device)
    try:
        for index, sample in enumerate(todo, 1):
            try:
                geometry = model.predict(sample["image_path"])
                write_npz(sample_paths(stage_root, sample) / "geometry.npz", **geometry)
                print(f"  vggt {index}/{len(todo)} {sample['sample_id']}", flush=True)
            except Exception as error:
                record_error(errors, sample, "vggt", error)
    finally:
        model.close()


def finalize_sample(sample: dict, stage_root: Path, output_root: Path, threshold: float) -> None:
    if sample_complete(output_root, sample):
        return
    stage = sample_paths(stage_root, sample)
    needed = [stage / name for name in ("detections.json", "masks.npz", "keypoints.npy", "geometry.npz")]
    if not all(path.is_file() for path in needed):
        raise FileNotFoundError("One or more prerequisite stage files are missing")
    detections = read_json(stage / "detections.json")
    with np.load(stage / "masks.npz") as data:
        person_mask = data["person"].astype(bool)
        object_mask = data["object"].astype(bool)
    keypoints = np.load(stage / "keypoints.npy")
    with np.load(stage / "geometry.npz") as data:
        geometry = {key: data[key] for key in data.files}
    processed_hw = tuple(int(value) for value in geometry["processed_size_hw"])
    original_hw = tuple(int(value) for value in geometry["original_size_hw"])
    person_processed = mask_to_vggt_pad(person_mask, processed_hw)
    object_processed = mask_to_vggt_pad(object_mask, processed_hw)
    keypoints_processed = keypoints_to_vggt_pad(keypoints, original_hw, processed_hw)
    solution = solve_human_object_geometry(
        geometry["point_map_camera"], geometry["depth_confidence"], person_processed,
        object_processed, keypoints_processed, threshold,
    )
    primitive = build_primitive(sample["object_name"], detections, solution)
    trace = generate_reasoning_trace(primitive)
    validation = validate_reasoning_trace(primitive, trace, sample["relation"])
    output = sample_paths(output_root, sample)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "primitive.json", {"image": str(sample["image_path"]), "primitive": primitive})
    write_json(output / "trace.json", {"image": str(sample["image_path"]), **trace})
    cv2.imwrite(str(output / "person_mask.png"), person_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(output / "object_mask.png"), object_mask.astype(np.uint8) * 255)
    write_npz(
        output / "geometry.npz",
        depth=geometry["depth"], depth_confidence=geometry["depth_confidence"],
        point_map_camera=geometry["point_map_camera"],
        intrinsic_processed=geometry["intrinsic_processed"], keypoints=keypoints,
        person_mask_processed=person_processed, object_mask_processed=object_processed,
    )
    write_json(output / "validation.json", validation)
    if not validation["valid"]:
        raise RuntimeError("Generated trace failed validation")


def build_primitive(object_name: str, detections: dict, solution: dict) -> dict:
    human_center = solution["human_center_camera"]
    object_center = solution["object_center_camera"]
    right, up, forward = (solution[key] for key in ("right_axis_camera", "up_axis_camera", "forward_axis_camera"))
    transform = solution["camera_to_human"]
    object_human = solution["object_position_human"]
    relation = solution["relation"]
    return {
        "coordinate_convention": {
            "camera": "+x right, +y down, +z forward",
            "human": "+x right, +y up, +z back; front is -z",
        },
        "human_coordinate_frame": {
            "origin": human_center.tolist(), "right_axis": right.tolist(),
            "up_axis": up.tolist(), "forward_axis": forward.tolist(), "back_axis": (-forward).tolist(),
        },
        "camera_to_human_transform": {
            "rotation": transform[:3, :3].tolist(), "translation": transform[:3, 3].tolist(),
        },
        "object_relative_position": {
            "object": object_name, "camera_position": object_center.tolist(),
            "human_position": object_human.tolist(), "relation": relation,
        },
        "detections": detections,
    }


def build_report(output_root: Path, samples: list[dict], errors: list[dict]) -> dict:
    relation = defaultdict(lambda: {"total": 0, "valid": 0, "correct": 0})
    objects = defaultdict(lambda: {"total": 0, "valid": 0, "correct": 0})
    valid_count = correct_count = 0
    for sample in samples:
        relation[sample["relation"]]["total"] += 1
        objects[sample["object_name"]]["total"] += 1
        path = sample_paths(output_root, sample) / "validation.json"
        if not path.is_file():
            continue
        try:
            result = read_json(path)
        except Exception:
            continue
        valid = bool(result.get("valid"))
        correct = result.get("computed_answer") == sample["relation"]
        valid_count += int(valid)
        correct_count += int(correct)
        relation[sample["relation"]]["valid"] += int(valid)
        relation[sample["relation"]]["correct"] += int(correct)
        objects[sample["object_name"]]["valid"] += int(valid)
        objects[sample["object_name"]]["correct"] += int(correct)
    total = len(samples)
    return {
        "total_samples": total,
        "valid_samples": valid_count,
        "failed_samples": total - valid_count,
        "valid_ratio": valid_count / total if total else 0.0,
        "relation_accuracy": correct_count / total if total else 0.0,
        "by_relation": dict(sorted(relation.items())),
        "by_object": dict(sorted(objects.items())),
        "errors": errors,
    }


def record_error(errors: list[dict], sample: dict, stage: str, error: Exception) -> None:
    item = {
        "sample_id": sample["sample_id"], "stage": stage,
        "error_type": type(error).__name__, "message": str(error),
        "traceback": traceback.format_exc(),
    }
    errors.append(item)
    print(f"ERROR {stage} {sample['sample_id']}: {type(error).__name__}: {error}", flush=True)


def json_detection(detection: dict) -> dict:
    return {
        "box_xyxy": np.asarray(detection["box_xyxy"]).tolist(),
        "score": float(detection["score"]), "label": detection["label"],
    }


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def write_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, value)
    os.replace(temporary, path)


def write_npz(path: Path, **values) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **values)
    os.replace(temporary, path)


if __name__ == "__main__":
    main()
