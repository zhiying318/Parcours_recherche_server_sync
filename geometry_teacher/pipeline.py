"""Single-image geometry teacher orchestration."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from geometry_teacher.model_adapters import (
    GroundingDinoAdapter,
    Sam2Adapter,
    VggtAdapter,
    ViTPoseAdapter,
)
from geometry_teacher.solver import solve_human_object_geometry


def run_geometry_teacher(
    image_path: str | Path,
    object_name: str,
    device: str = "cuda",
    vitpose_model_id: str = "usyd-community/vitpose-base",
    keypoint_threshold: float = 0.3,
) -> tuple[dict, dict]:
    """Run all teacher models sequentially to minimize peak GPU memory."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    if not object_name.strip():
        raise ValueError("object_name must be non-empty")

    detector = GroundingDinoAdapter(device=device)
    try:
        person_detection = detector.detect_one(image_path, "person")
        object_detection = detector.detect_one(image_path, object_name)
    finally:
        detector.close()

    segmenter = Sam2Adapter(device=device)
    try:
        person_mask = segmenter.segment_box(image_path, person_detection["box_xyxy"])
        object_mask = segmenter.segment_box(image_path, object_detection["box_xyxy"])
    finally:
        segmenter.close()
    if np.any(person_mask & object_mask):
        raise ValueError("Selected person and object masks overlap")

    pose = ViTPoseAdapter(model_id=vitpose_model_id, device=device)
    try:
        keypoints = pose.predict_one(
            image_path, person_detection["box_xyxy"], person_detection["score"]
        )
    finally:
        pose.close()

    vggt = VggtAdapter(device=device)
    try:
        geometry = vggt.predict(image_path)
    finally:
        vggt.close()

    processed_hw = tuple(geometry["processed_size_hw"])
    person_mask_processed = mask_to_vggt_pad(person_mask, processed_hw)
    object_mask_processed = mask_to_vggt_pad(object_mask, processed_hw)
    keypoints_processed = keypoints_to_vggt_pad(
        keypoints, tuple(geometry["original_size_hw"]), processed_hw
    )
    point_map = geometry["point_map_camera"]
    confidence = geometry["depth_confidence"]
    solution = solve_human_object_geometry(
        point_map,
        confidence,
        person_mask_processed,
        object_mask_processed,
        keypoints_processed,
        keypoint_threshold,
    )
    human_center = solution["human_center_camera"]
    object_center = solution["object_center_camera"]
    right = solution["right_axis_camera"]
    up = solution["up_axis_camera"]
    forward = solution["forward_axis_camera"]
    camera_to_human = solution["camera_to_human"]
    object_human = solution["object_position_human"]
    relation = solution["relation"]

    primitive = {
        "coordinate_convention": {
            "camera": "+x right, +y down, +z forward",
            "human": "+x right, +y up, +z back; front is -z",
        },
        "human_coordinate_frame": {
            "origin": human_center.tolist(),
            "right_axis": right.tolist(),
            "up_axis": up.tolist(),
            "forward_axis": forward.tolist(),
            "back_axis": (-forward).tolist(),
        },
        "camera_to_human_transform": {
            "rotation": camera_to_human[:3, :3].tolist(),
            "translation": camera_to_human[:3, 3].tolist(),
        },
        "object_relative_position": {
            "object": object_name,
            "camera_position": object_center.tolist(),
            "human_position": object_human.tolist(),
            "relation": relation,
        },
        "detections": {
            "person": _json_detection(person_detection),
            "object": _json_detection(object_detection),
        },
    }
    artifacts = {
        "person_mask": person_mask,
        "object_mask": object_mask,
        "person_mask_processed": person_mask_processed,
        "object_mask_processed": object_mask_processed,
        "keypoints": keypoints,
        **geometry,
    }
    return primitive, artifacts


def mask_to_vggt_pad(mask: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    height, width = mask.shape
    target_h, target_w = target_hw
    if target_h != target_w:
        raise ValueError(f"VGGT pad target must be square, got {target_hw}")
    scale = target_w / max(height, width)
    resized_w = round(width * scale / 14) * 14
    resized_h = round(height * scale / 14) * 14
    resized = cv2.resize(mask.astype(np.uint8), (resized_w, resized_h), interpolation=cv2.INTER_NEAREST)
    output = np.zeros(target_hw, dtype=bool)
    top = (target_h - resized_h) // 2
    left = (target_w - resized_w) // 2
    output[top : top + resized_h, left : left + resized_w] = resized.astype(bool)
    return output


def keypoints_to_vggt_pad(
    keypoints: np.ndarray, original_hw: tuple[int, int], target_hw: tuple[int, int]
) -> np.ndarray:
    height, width = original_hw
    target_h, target_w = target_hw
    scale = target_w / max(height, width)
    resized_w = round(width * scale / 14) * 14
    resized_h = round(height * scale / 14) * 14
    output = keypoints.copy().astype(np.float64)
    output[:, 0] = output[:, 0] * (resized_w / width) + (target_w - resized_w) // 2
    output[:, 1] = output[:, 1] * (resized_h / height) + (target_h - resized_h) // 2
    return output


def _json_detection(detection: dict) -> dict:
    return {
        "box_xyxy": np.asarray(detection["box_xyxy"]).tolist(),
        "score": detection["score"],
        "label": detection["label"],
    }
