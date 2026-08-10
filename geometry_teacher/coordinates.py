"""Coordinate transforms for the geometry-teacher pipeline.

The camera frame follows OpenCV/VGGT: +x right, +y down, +z forward.
The human frame is right-handed: +x human-right, +y human-up, +z human-back.
Consequently, points in front of the human have a negative human-frame z.
"""

from __future__ import annotations

import numpy as np


BLENDER_CAMERA_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def world_to_opencv_camera(camera_to_world_blender: np.ndarray) -> np.ndarray:
    """Return a 4x4 world-to-camera transform in the OpenCV convention."""
    camera_to_world_blender = _matrix4(camera_to_world_blender)
    return BLENDER_CAMERA_TO_OPENCV @ np.linalg.inv(camera_to_world_blender)


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 rigid transform to one point or an array of 3D points."""
    transform = _matrix4(transform)
    points = np.asarray(points, dtype=np.float64)
    if points.shape[-1] != 3:
        raise ValueError(f"Expected points ending in dimension 3, got {points.shape}")
    homogeneous = np.concatenate(
        [points.reshape(-1, 3), np.ones((points.reshape(-1, 3).shape[0], 1))],
        axis=1,
    )
    transformed = homogeneous @ transform.T
    return transformed[:, :3].reshape(points.shape)


def camera_to_human_transform(
    human_origin_camera: np.ndarray,
    right_axis_camera: np.ndarray,
    up_axis_camera: np.ndarray,
    forward_axis_camera: np.ndarray,
) -> np.ndarray:
    """Build the camera-to-human rigid transform.

    ``forward_axis_camera`` describes the semantic forward direction. The
    right-handed human coordinate matrix uses its negation as the +z/back axis.
    Axes must already describe a proper orthonormal human frame.
    """
    origin = _vector3(human_origin_camera, "human_origin_camera")
    right = _unit3(right_axis_camera, "right_axis_camera")
    up = _unit3(up_axis_camera, "up_axis_camera")
    forward = _unit3(forward_axis_camera, "forward_axis_camera")
    rotation = np.stack([right, up, -forward], axis=0)
    assert_rotation_matrix(rotation)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ origin
    return transform


def classify_horizontal_relation(relative_position_human: np.ndarray) -> str:
    """Classify a human-frame displacement into left/right/front/back."""
    x_h, _, z_h = _vector3(relative_position_human, "relative_position_human")
    if abs(x_h) > abs(z_h):
        return "right" if x_h > 0 else "left"
    return "front" if z_h < 0 else "back"


def assert_rotation_matrix(rotation: np.ndarray, atol: float = 1e-6) -> None:
    """Raise ValueError unless ``rotation`` is a proper 3D rotation."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 rotation matrix, got {rotation.shape}")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=atol):
        raise ValueError("Rotation axes are not orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not np.isclose(determinant, 1.0, atol=atol):
        raise ValueError(f"Rotation determinant must be +1, got {determinant}")


def _matrix4(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 matrix, got {value.shape}")
    return value


def _vector3(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (3,):
        raise ValueError(f"Expected {name} to have shape (3,), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"Expected {name} to contain only finite values")
    return value


def _unit3(value: np.ndarray, name: str) -> np.ndarray:
    value = _vector3(value, name)
    norm = float(np.linalg.norm(value))
    if not np.isclose(norm, 1.0, atol=1e-6):
        raise ValueError(f"Expected {name} to be unit length, got norm {norm}")
    return value
