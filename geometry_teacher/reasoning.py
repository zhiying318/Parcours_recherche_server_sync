"""Deterministic reasoning-trace generation and validation."""

from __future__ import annotations

import numpy as np

from geometry_teacher.coordinates import (
    assert_rotation_matrix,
    classify_horizontal_relation,
)


RELATIONS = ("left", "right", "front", "back")


def generate_reasoning_trace(primitive: dict) -> dict:
    """Render a verifiable trace from a solved primitive without an LLM."""
    relative = primitive["object_relative_position"]
    object_name = relative["object"]
    position = _vector(relative["human_position"], "human_position")
    relation = classify_horizontal_relation(position)
    if relation != relative["relation"]:
        raise ValueError("Primitive relation disagrees with its human-frame position")

    x_h, y_h, z_h = position.tolist()
    dominant_axis = "x" if abs(x_h) > abs(z_h) else "z"
    return {
        "generator": "deterministic_geometry_template_v1",
        "reasoning_trace": [
            {
                "step": 1,
                "operation": "establish_human_frame",
                "text": "The human reference frame is represented in camera coordinates.",
            },
            {
                "step": 2,
                "operation": "camera_to_human",
                "human_position": [x_h, y_h, z_h],
                "text": (
                    f"The {object_name} center is transformed into the human frame at "
                    f"({x_h:.6f}, {y_h:.6f}, {z_h:.6f})."
                ),
            },
            {
                "step": 3,
                "operation": "dominant_horizontal_axis",
                "dominant_axis": dominant_axis,
                "relation": relation,
                "text": (
                    f"The dominant human-frame horizontal axis is {dominant_axis}; "
                    f"therefore the {object_name} is {relation}."
                ),
            },
            {
                "step": 4,
                "operation": "conclude",
                "answer": relation,
                "text": f"Therefore the answer is {relation}.",
            },
        ],
        "answer": relation,
    }


def validate_reasoning_trace(
    primitive: dict,
    trace: dict,
    expected_answer: str | None = None,
    atol: float = 1e-5,
) -> dict:
    """Validate transform, relation, trace conclusion, and optional ground truth."""
    frame = primitive["human_coordinate_frame"]
    transform = primitive["camera_to_human_transform"]
    relative = primitive["object_relative_position"]

    rotation = np.asarray(transform["rotation"], dtype=np.float64)
    translation = _vector(transform["translation"], "translation")
    origin = _vector(frame["origin"], "origin")
    object_camera = _vector(relative["camera_position"], "camera_position")
    object_human = _vector(relative["human_position"], "human_position")

    checks: dict[str, bool] = {}
    try:
        assert_rotation_matrix(rotation, atol=atol)
        checks["rotation_is_proper"] = True
    except ValueError:
        checks["rotation_is_proper"] = False
    checks["translation_matches_origin"] = bool(
        np.allclose(translation, -rotation @ origin, atol=atol)
    )
    recomputed_position = rotation @ object_camera + translation
    checks["transform_correct"] = bool(
        np.allclose(recomputed_position, object_human, atol=atol)
    )

    relation = classify_horizontal_relation(object_human)
    checks["primitive_relation_correct"] = relation == relative["relation"]
    trace_relation = trace["reasoning_trace"][2].get("relation")
    trace_answer = trace.get("answer")
    conclusion_answer = trace["reasoning_trace"][3].get("answer")
    checks["trace_relation_correct"] = trace_relation == relation
    checks["trace_conclusion_correct"] = trace_answer == relation == conclusion_answer
    checks["answer_matches_ground_truth"] = (
        True if expected_answer is None else trace_answer == expected_answer
    )
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "expected_answer": expected_answer,
        "computed_answer": relation,
    }


def _vector(value: object, name: str) -> np.ndarray:
    output = np.asarray(value, dtype=np.float64)
    if output.shape != (3,) or not np.all(np.isfinite(output)):
        raise ValueError(f"{name} must be a finite length-3 vector")
    return output
