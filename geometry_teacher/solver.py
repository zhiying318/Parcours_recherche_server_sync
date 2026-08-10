"""Deterministic human-centric geometry solver."""

from __future__ import annotations

import numpy as np

from geometry_teacher.coordinates import (
    camera_to_human_transform,
    classify_horizontal_relation,
    transform_points,
)


COCO_LEFT_SHOULDER = 5
COCO_RIGHT_SHOULDER = 6
COCO_LEFT_HIP = 11
COCO_RIGHT_HIP = 12


def solve_human_object_geometry(
    point_map_camera: np.ndarray,
    point_confidence: np.ndarray,
    person_mask: np.ndarray,
    object_mask: np.ndarray,
    coco_keypoints: np.ndarray,
    keypoint_threshold: float = 0.3,
) -> dict:
    human_center = masked_point_center(point_map_camera, point_confidence, person_mask)
    object_center = masked_point_center(point_map_camera, point_confidence, object_mask)
    right, up, forward = human_axes_from_keypoints(
        point_map_camera,
        point_confidence,
        person_mask,
        coco_keypoints,
        keypoint_threshold,
    )
    transform = camera_to_human_transform(human_center, right, up, forward)
    object_human = transform_points(transform, object_center)
    return {
        "human_center_camera": human_center,
        "object_center_camera": object_center,
        "right_axis_camera": right,
        "up_axis_camera": up,
        "forward_axis_camera": forward,
        "camera_to_human": transform,
        "object_position_human": object_human,
        "relation": classify_horizontal_relation(object_human),
        "classification_margin": float(abs(abs(object_human[0]) - abs(object_human[2]))),
    }


def masked_point_center(
    point_map: np.ndarray, confidence: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    _validate_geometry_shapes(point_map, confidence, mask)
    valid = mask & np.isfinite(point_map).all(axis=-1) & np.isfinite(confidence)
    if not np.any(valid):
        raise ValueError("Mask contains no valid geometry points")
    threshold = np.median(confidence[valid])
    selected = valid & (confidence >= threshold)
    return np.median(point_map[selected], axis=0)


def human_axes_from_keypoints(
    point_map: np.ndarray,
    confidence: np.ndarray,
    person_mask: np.ndarray,
    keypoints: np.ndarray,
    keypoint_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _validate_geometry_shapes(point_map, confidence, person_mask)
    if keypoints.shape != (17, 3):
        raise ValueError(f"Expected COCO keypoints shape (17, 3), got {keypoints.shape}")
    indices = [COCO_LEFT_SHOULDER, COCO_RIGHT_SHOULDER, COCO_LEFT_HIP, COCO_RIGHT_HIP]
    if np.any(keypoints[indices, 2] < keypoint_threshold):
        raise ValueError("Required ViTPose shoulder/hip confidence is below threshold")
    joints = [
        sample_joint_3d(point_map, confidence, person_mask, keypoints[index, :2])
        for index in indices
    ]
    left_shoulder, right_shoulder, left_hip, right_hip = joints
    right = _normalize(right_shoulder - left_shoulder, "shoulder right axis")
    shoulder_midpoint = (left_shoulder + right_shoulder) / 2
    hip_midpoint = (left_hip + right_hip) / 2
    up_raw = shoulder_midpoint - hip_midpoint
    up = _normalize(up_raw - np.dot(up_raw, right) * right, "torso up axis")
    back = _normalize(np.cross(right, up), "back axis")
    return right, up, -back


def sample_joint_3d(
    point_map: np.ndarray,
    confidence: np.ndarray,
    person_mask: np.ndarray,
    point_xy: np.ndarray,
    radius: int = 2,
) -> np.ndarray:
    x, y = np.rint(point_xy).astype(int)
    height, width = confidence.shape
    if not (0 <= x < width and 0 <= y < height):
        raise ValueError(f"Keypoint ({x}, {y}) lies outside geometry map")
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    points = point_map[y0:y1, x0:x1].reshape(-1, 3)
    scores = confidence[y0:y1, x0:x1].reshape(-1)
    semantic = person_mask[y0:y1, x0:x1].reshape(-1)
    valid = semantic & np.isfinite(points).all(axis=1) & np.isfinite(scores)
    if not np.any(valid):
        raise ValueError(f"No valid person geometry near joint ({x}, {y})")
    threshold = np.median(scores[valid])
    return np.median(points[valid & (scores >= threshold)], axis=0)


def _normalize(vector: np.ndarray, name: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-8:
        raise ValueError(f"Degenerate {name}")
    return vector / norm


def _validate_geometry_shapes(
    point_map: np.ndarray, confidence: np.ndarray, mask: np.ndarray
) -> None:
    if point_map.ndim != 3 or point_map.shape[-1] != 3:
        raise ValueError(f"Expected point map shape (H, W, 3), got {point_map.shape}")
    if confidence.shape != point_map.shape[:2] or mask.shape != point_map.shape[:2]:
        raise ValueError("Point map, confidence, and mask spatial shapes must match")
    if mask.dtype != np.bool_:
        raise ValueError(f"Mask must have bool dtype, got {mask.dtype}")
