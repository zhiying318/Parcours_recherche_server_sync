"""Canonical, auditable geometry extraction for the spatial SFT experiment.

This module deliberately has no dependency on the existing OPSD or
self-distillation implementation.  It reads the Blender export contract
directly and keeps source labels separate from the relation computed from the
exported evidence.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Iterable

RELATIONS = ("front", "behind", "left", "right")
SOURCE_RELATION_MAP = {
    "front": "front",
    "back": "behind",
    "behind": "behind",
    "left": "left",
    "right": "right",
    "infrontof": "front",
    "totheleft": "left",
    "totheright": "right",
}


class GeometryError(ValueError):
    """A source sample cannot safely become a supervision example."""


@dataclass(frozen=True)
class GeometryEvidence:
    person_position: tuple[float, float, float]
    object_position: tuple[float, float, float]
    person_forward: tuple[float, float, float]
    person_right: tuple[float, float, float]
    relative_vector: tuple[float, float, float]
    forward_projection: float
    right_projection: float
    computed_relation: str
    source_relation_raw: str
    source_relation_canonical: str
    source_relation_match: bool
    scale: float
    orthogonality_dot: float


@dataclass(frozen=True)
class QuantizedEvidence:
    person_position: tuple[float, float, float]
    object_position: tuple[float, float, float]
    person_forward: tuple[float, float, float]
    person_right: tuple[float, float, float]
    relative_vector: tuple[float, float, float]
    forward_projection: float
    right_projection: float
    computed_relation: str
    digits: int
    full_forward_projection: float
    full_right_projection: float
    normalized_margin: float
    classification_changed: bool
    degenerate: bool
    tie: bool


def _vec(value: Iterable[Any], name: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(x) for x in value)
    except (TypeError, ValueError) as exc:
        raise GeometryError(f"{name} is not a numeric vector") from exc
    if len(values) != 3 or not all(math.isfinite(x) for x in values):
        raise GeometryError(f"{name} must contain three finite values")
    return values  # type: ignore[return-value]


def _sub(a: Iterable[float], b: Iterable[float]) -> tuple[float, float, float]:
    return tuple(x - y for x, y in zip(a, b))  # type: ignore[return-value]


def _dot(a: Iterable[float], b: Iterable[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: Iterable[float]) -> float:
    return math.sqrt(_dot(a, a))


def _normalize(a: tuple[float, float, float], name: str) -> tuple[float, float, float]:
    length = _norm(a)
    if not math.isfinite(length) or length <= 1e-12:
        raise GeometryError(f"{name} has invalid or zero length")
    return tuple(x / length for x in a)  # type: ignore[return-value]


def canonical_source_relation(raw: Any) -> str:
    if not isinstance(raw, str) or raw not in SOURCE_RELATION_MAP:
        raise GeometryError(f"unsupported source relation: {raw!r}")
    return SOURCE_RELATION_MAP[raw]


def classify_projections(forward_projection: float, right_projection: float) -> str:
    """Apply the dataset rule: forward wins an exact magnitude tie."""
    if not (math.isfinite(forward_projection) and math.isfinite(right_projection)):
        raise GeometryError("non-finite projection")
    if abs(forward_projection) <= 1e-12 and abs(right_projection) <= 1e-12:
        raise GeometryError("zero displacement in both projected axes")
    if abs(forward_projection) >= abs(right_projection):
        return "front" if forward_projection > 0 else "behind"
    return "right" if right_projection > 0 else "left"


def verify_relation(computed_relation: str, source_relation: Any) -> dict[str, Any]:
    """Compare a computed relation with the independent source label."""
    canonical = canonical_source_relation(source_relation)
    return {
        "source_relation_raw": source_relation,
        "source_relation_canonical": canonical,
        "source_relation_match": computed_relation == canonical,
    }


def load_raw_sample(scene_path: str | Path) -> dict[str, Any]:
    path = Path(scene_path)
    with path.open(encoding="utf-8") as handle:
        scene = json.load(handle)
    required = (
        "human_visible_center_camera",
        "object_visible_center_camera",
        "human_frame_camera",
        "relation",
        "object_name",
    )
    missing = [key for key in required if key not in scene]
    if missing:
        raise GeometryError(f"{path}: missing fields {missing}")
    axes = scene["human_frame_camera"]
    for key in ("forward_axis", "right_axis"):
        if key not in axes:
            raise GeometryError(f"{path}: missing human_frame_camera.{key}")
    return scene


def extract_canonical_evidence(scene: dict[str, Any]) -> GeometryEvidence:
    person = _vec(scene["human_visible_center_camera"], "human_visible_center_camera")
    obj = _vec(scene["object_visible_center_camera"], "object_visible_center_camera")
    forward = _normalize(_vec(scene["human_frame_camera"]["forward_axis"], "forward_axis"), "forward_axis")
    right = _normalize(_vec(scene["human_frame_camera"]["right_axis"], "right_axis"), "right_axis")
    orthogonality = _dot(forward, right)
    if abs(orthogonality) > 1e-4:
        raise GeometryError(f"forward/right axes are not orthogonal: dot={orthogonality}")
    scale = person[2]
    if not math.isfinite(scale) or scale <= 0:
        raise GeometryError(f"person z scale must be positive, got {scale}")
    relative = _sub(obj, person)
    forward_projection = _dot(relative, forward)
    right_projection = _dot(relative, right)
    computed = classify_projections(forward_projection, right_projection)
    relation_check = verify_relation(computed, scene["relation"])
    return GeometryEvidence(
        person_position=person,
        object_position=obj,
        person_forward=forward,
        person_right=right,
        relative_vector=relative,
        forward_projection=forward_projection,
        right_projection=right_projection,
        computed_relation=computed,
        scale=scale,
        orthogonality_dot=orthogonality,
        **relation_check,
    )


def normalize_positions(evidence: GeometryEvidence) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    scale = evidence.scale
    return (
        tuple(x / scale for x in evidence.person_position),
        tuple(x / scale for x in evidence.object_position),
    )  # type: ignore[return-value]


def _round_vector(vector: Iterable[float], digits: int) -> tuple[float, float, float]:
    rounded = tuple(round(float(x), digits) for x in vector)
    return tuple(0.0 if x == 0.0 else x for x in rounded)  # type: ignore[return-value]


def quantize_and_verify_evidence(evidence: GeometryEvidence, digits=(3, 4, 5, 6)) -> QuantizedEvidence:
    """Quantize normalized positions and axes, recompute, and retry precision."""
    normalized_person, normalized_object = normalize_positions(evidence)
    full_f = evidence.forward_projection / evidence.scale
    full_r = evidence.right_projection / evidence.scale
    for precision in digits:
        person = _round_vector(normalized_person, precision)
        obj = _round_vector(normalized_object, precision)
        forward = _round_vector(evidence.person_forward, precision)
        right = _round_vector(evidence.person_right, precision)
        # The displayed displacement is itself a displayed quantity.  Round
        # it explicitly so audit JSON never leaks binary floating-point tails.
        relative = _round_vector(_sub(obj, person), precision)
        f_proj = round(_dot(relative, forward), precision)
        r_proj = round(_dot(relative, right), precision)
        try:
            relation = classify_projections(f_proj, r_proj)
        except GeometryError:
            relation = "invalid"
        changed = relation != evidence.computed_relation
        degenerate = relation == "invalid"
        if not changed and not degenerate:
            denominator = max(abs(full_f), abs(full_r), 1e-12)
            margin = abs(abs(full_f) - abs(full_r)) / denominator
            return QuantizedEvidence(
                person_position=person,
                object_position=obj,
                person_forward=forward,
                person_right=right,
                relative_vector=relative,
                forward_projection=f_proj,
                right_projection=r_proj,
                computed_relation=relation,
                digits=precision,
                full_forward_projection=full_f,
                full_right_projection=full_r,
                normalized_margin=margin,
                classification_changed=False,
                degenerate=False,
                tie=abs(abs(f_proj) - abs(r_proj)) <= 0.5 * 10 ** (-precision),
            )
    raise GeometryError(
        f"display quantization changed or degenerated relation after precisions {tuple(digits)}"
    )


def evidence_dict(evidence: GeometryEvidence | QuantizedEvidence) -> dict[str, Any]:
    return asdict(evidence)
