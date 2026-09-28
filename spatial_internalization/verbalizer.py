"""Deterministic short geometry reasoning verbalizer."""

from __future__ import annotations

from .geometry import QuantizedEvidence

RELATION_PHRASE = {
    "front": "in front of the person",
    "behind": "behind the person",
    "left": "to the person's left",
    "right": "to the person's right",
}


def vector_text(vector: tuple[float, float, float], digits: int) -> str:
    return "[" + ", ".join(f"{x:.{digits}f}" for x in vector) + "]"


def verbalize_evidence_template(object_name: str, evidence: QuantizedEvidence) -> str:
    f_abs, r_abs = abs(evidence.forward_projection), abs(evidence.right_projection)
    if abs(f_abs - r_abs) <= 0.5 * 10 ** (-evidence.digits):
        comparison = "The forward and right components have equal absolute magnitude."
    elif f_abs > r_abs:
        comparison = "The forward component has the larger absolute magnitude."
    else:
        comparison = "The right component has the larger absolute magnitude."
    dominant = "forward" if f_abs >= r_abs else "right"
    selected = evidence.forward_projection if dominant == "forward" else evidence.right_projection
    sign = "positive" if selected > 0 else "negative"
    phrase = RELATION_PHRASE[evidence.computed_relation]
    return (
        "Consider that the picture was taken from the origin [0.000, 0.000, 0.000] "
        "of a camera coordinate system, where +X points to the image's right, +Y "
        "points downward, and +Z points forward into the scene.\n\n"
        f"The person's estimated position is {vector_text(evidence.person_position, evidence.digits)}, "
        f"and the {object_name}'s estimated position is "
        f"{vector_text(evidence.object_position, evidence.digits)}.\n"
        f"The person-to-object displacement is therefore "
        f"{vector_text(evidence.relative_vector, evidence.digits)}.\n\n"
        f"The person's estimated forward axis is {vector_text(evidence.person_forward, evidence.digits)}, "
        f"and their right axis is {vector_text(evidence.person_right, evidence.digits)}.\n\n"
        f"The displacement projects approximately {evidence.forward_projection:.{evidence.digits}f} "
        f"onto the forward axis and {evidence.right_projection:.{evidence.digits}f} onto the right axis. "
        f"{comparison} The {dominant} component has the {sign} sign, which places the "
        f"{object_name} {phrase}."
    )
