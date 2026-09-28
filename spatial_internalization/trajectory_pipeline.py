"""Data and deterministic geometry logic for the second internalization version.

The first version in this directory creates supervision directly from Blender
geometry.  This module keeps that geometry implementation as the source of
truth, but joins it with one audited test07 trajectory CSV and exposes the
intermediate evidence representation used by the local consolidator and
verbalizer.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable

from .data_pipeline import (
    make_label_preserving_noisy,
    scene_group_id,
)
from .geometry import (
    GeometryError,
    GeometryEvidence,
    QuantizedEvidence,
    evidence_dict,
    extract_canonical_evidence,
    load_raw_sample,
    quantize_and_verify_evidence,
)
from .verbalizer import RELATION_PHRASE, verbalize_evidence_template


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CSV = (
    REPO_ROOT
    / "comfort_addionalprompt_tests/test07_camera_geometry_before_question/"
    / "results_preciseprompt/mcq_long_qwen3_5vl_thinking.csv"
)
DEFAULT_SOURCE_ROOT = REPO_ROOT / "COMFORT/data/comfort_human_car_geometry_gt/comfort_human_car"
DEFAULT_FIXED_SPLIT_MANIFEST = REPO_ROOT / "spatial_internalization/data/manifest.json"

HISTORICAL_TEACHER_RECORD = (
    "Qwen3.5-VL thinking; the source CSV does not record a verifiable checkpoint "
    "size, so the size is intentionally not inferred"
)

_VECTOR_RE = r"\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\]"
_PROMPT_GEOMETRY_RE = re.compile(
    r"located at (?P<person>\[[^]]+\]) and is looking in the direction of "
    r"(?P<forward>\[[^]]+\]).*?is located at (?P<object>\[[^]]+\])",
    flags=re.DOTALL,
)


def _vector(text: str) -> tuple[float, float, float]:
    match = re.fullmatch(_VECTOR_RE, text.strip())
    if not match:
        raise ValueError(f"not a 3-vector: {text!r}")
    return tuple(float(value) for value in match.groups())  # type: ignore[return-value]


def _same_vector(left: Iterable[float], right: Iterable[float], tolerance: float = 1e-6) -> bool:
    return all(abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right))


def _rounded(vector: Iterable[float], digits: int = 3) -> tuple[float, float, float]:
    return tuple(round(float(value), digits) for value in vector)  # type: ignore[return-value]


def parse_prompt_geometry(prompt: str) -> dict[str, tuple[float, float, float]]:
    """Extract and validate the three GT vectors embedded in test07's prompt."""
    match = _PROMPT_GEOMETRY_RE.search(prompt)
    if not match:
        raise ValueError("test07 prompt does not contain the expected three vectors")
    return {name: _vector(value) for name, value in match.groupdict().items()}


def trajectory_id(image_path: str) -> str:
    digest = hashlib.sha256(image_path.encode("utf-8")).hexdigest()[:16]
    return f"test07_qwen35_thinking::{digest}"


def _source_image_map(source_root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    result = {}
    for scene_path in sorted(source_root.glob("*/*/scene_gt.json")):
        image_path = scene_path.parent / "0.png"
        if not image_path.exists():
            raise GeometryError(f"missing source image: {image_path}")
        result[image_path.relative_to(REPO_ROOT).as_posix()] = (
            scene_path,
            load_raw_sample(scene_path),
        )
    return result


def _row_rejection_reason(
    row: dict[str, str],
    scene: dict[str, Any],
    clean: GeometryEvidence,
) -> str | None:
    if not row["model_answer"].strip():
        return "empty_trajectory"
    if "</think>" not in row["model_answer"]:
        return "trajectory_not_closed"
    if row["pred_letter"].strip() != row["correct_letter"].strip():
        return "answer_incorrect_or_unparsed"
    if row["correct_relation"].strip() != clean.computed_relation:
        return "csv_relation_disagrees_with_clean_gt"
    try:
        prompt_vectors = parse_prompt_geometry(row["mcq_prompt"])
    except ValueError as exc:
        return f"prompt_geometry_unparseable:{exc}"
    expected = {
        "person": _rounded(scene["human_visible_center_camera"]),
        "forward": _rounded(scene["human_frame_camera"]["forward_axis"]),
        "object": _rounded(scene["object_visible_center_camera"]),
    }
    if any(not _same_vector(prompt_vectors[name], expected[name]) for name in expected):
        return "csv_prompt_geometry_disagrees_with_blender_gt"
    if row["correct_letter"].strip() not in "ABCD":
        return "invalid_answer_letter"
    return None


def load_qualified_trajectories(
    source_csv: Path,
    source_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load only complete, answer-correct, geometry-checked source trajectories."""
    scenes = _source_image_map(source_root)
    required = {
        "image_path",
        "second_object",
        "mcq_prompt",
        "correct_relation",
        "correct_letter",
        "model_answer",
        "pred_letter",
    }
    qualified = []
    rejected = []
    with source_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or not required.issubset(rows[0]):
        missing = sorted(required - set(rows[0] if rows else ()))
        raise ValueError(f"source CSV is missing required columns: {missing}")

    seen = set()
    for index, row in enumerate(rows):
        image_path = row["image_path"].removeprefix("./")
        if image_path in seen:
            rejected.append({"source_row": index, "image_path": image_path, "reason": "duplicate_image"})
            continue
        seen.add(image_path)
        if image_path not in scenes:
            rejected.append({"source_row": index, "image_path": image_path, "reason": "image_not_in_blender_source"})
            continue
        scene_path, scene = scenes[image_path]
        try:
            clean = extract_canonical_evidence(scene)
        except GeometryError as exc:
            rejected.append({"source_row": index, "image_path": image_path, "reason": f"invalid_clean_gt:{exc}"})
            continue
        reason = _row_rejection_reason(row, scene, clean)
        if reason:
            rejected.append({"source_row": index, "image_path": image_path, "reason": reason})
            continue
        display = quantize_and_verify_evidence(clean)
        qualified.append({
            "source_row": index,
            "trajectory_id": trajectory_id(image_path),
            "image_path": image_path,
            "scene_path": scene_path,
            "scene": scene,
            "scene_group_id": scene_group_id(scene_path),
            "camera_view": re.search(r"__cam_([^/]+)$", scene_path.parent.name).group(1),
            "row": row,
            "clean_raw": clean,
            "clean_display": display,
            "raw_trajectory_chars": len(row["model_answer"]),
        })
    if len({item["image_path"] for item in qualified}) != len(qualified):
        raise ValueError("qualified source contains duplicate images")
    audit = {
        "source_csv": source_csv.relative_to(REPO_ROOT).as_posix()
        if source_csv.is_relative_to(REPO_ROOT)
        else str(source_csv),
        "historical_teacher_model": HISTORICAL_TEACHER_RECORD,
        "raw_row_count": len(rows),
        "qualified_row_count": len(qualified),
        "rejected_row_count": len(rejected),
        "rejected_by_reason": _count_reasons(rejected),
        "trajectory_filter": {
            "requires_nonempty_model_answer": True,
            "requires_think_end_marker": True,
            "requires_predicted_letter_equal_csv_answer": True,
            "requires_csv_relation_equal_blender_geometry": True,
            "requires_prompt_geometry_equal_blender_geometry": True,
        },
    }
    return qualified, rejected, audit


def _count_reasons(records: Iterable[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        reason = str(record["reason"]).split(":", 1)[0]
        counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items()))


def load_fixed_split_assignments(manifest_path: Path) -> dict[str, str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assignments = manifest.get("group_assignments")
    if not isinstance(assignments, dict):
        raise ValueError(f"fixed split manifest has no group_assignments: {manifest_path}")
    if set(assignments.values()) != {"train", "validation", "test"}:
        raise ValueError("fixed split must contain train, validation, and test")
    return {str(key): str(value) for key, value in assignments.items()}


def attach_fixed_splits(
    trajectories: list[dict[str, Any]], assignments: dict[str, str]
) -> None:
    for item in trajectories:
        try:
            item["split"] = assignments[item["scene_group_id"]]
        except KeyError as exc:
            raise ValueError(
                f"qualified trajectory group is absent from the fixed split: {item['scene_group_id']}"
            ) from exc


def first_version_noise_seeds(
    source_root: Path,
    seed: int,
    assignments: dict[str, str],
    position_sigma: float,
    angle_sigma_degrees: float,
) -> dict[tuple[str, int], int]:
    """Reproduce the first version's exact per-record seed expression.

    The old implementation used ``seed + augmentation_index * 1000003 +
    len(records['train'])``.  The length is calculated over all valid source
    samples, not only the trajectory-filtered subset, so the mapping is built
    against the same Blender source order before filtering.
    """
    del position_sigma, angle_sigma_degrees  # part of the recorded contract
    samples = []
    for scene_path in sorted(source_root.glob("*/*/scene_gt.json")):
        image_path = scene_path.parent / "0.png"
        if image_path.exists():
            scene = load_raw_sample(scene_path)
            clean = extract_canonical_evidence(scene)
            quantize_and_verify_evidence(clean)
            samples.append((image_path.relative_to(REPO_ROOT).as_posix(), scene_path))
    result: dict[tuple[str, int], int] = {}
    train_record_count = 0
    for image_path, scene_path in samples:
        split = assignments[scene_group_id(scene_path)]
        count = 9 if split == "train" else 1
        for index in range(count):
            if split == "train" and index > 0:
                result[(image_path, index)] = seed + index * 1000003 + train_record_count
            if split == "train":
                # The first version appends answer_only and
                # geometry_reasoning records to the shared list for every
                # augmentation, so its len(records["train"]) advances by 2.
                train_record_count += 2
    return result


def geometry_for_augmentation(
    item: dict[str, Any],
    augmentation_index: int,
    noise_seed: int | None,
    position_sigma: float,
    angle_sigma_degrees: float,
) -> tuple[QuantizedEvidence, int]:
    if augmentation_index == 0:
        return item["clean_display"], 0
    if noise_seed is None:
        raise ValueError("no noise seed for a non-clean augmentation")
    noisy, attempts = make_label_preserving_noisy(
        item["clean_raw"],
        noise_seed,
        position_sigma,
        angle_sigma_degrees,
    )
    return quantize_and_verify_evidence(noisy), attempts


def _display_vector(vector: Iterable[float], digits: int = 3) -> str:
    return "[" + ", ".join(f"{float(value):.{digits}f}" for value in vector) + "]"


def canonical_geometry_evidence(
    object_name: str,
    display: QuantizedEvidence,
    provenance: str,
) -> list[dict[str, Any]]:
    """Create auditable evidence nodes whose dependencies are deterministic."""
    approximate = "estimated" if provenance != "clean_blender_gt" else "estimated"
    relation = RELATION_PHRASE[display.computed_relation]
    f_abs, r_abs = abs(display.forward_projection), abs(display.right_projection)
    dominant = "forward" if f_abs >= r_abs else "right"
    comparison = (
        "The forward component has the larger absolute magnitude."
        if f_abs > r_abs
        else "The right component has the larger absolute magnitude."
        if r_abs > f_abs
        else "The forward and right components have equal absolute magnitude."
    )
    sign = "positive" if (display.forward_projection if dominant == "forward" else display.right_projection) > 0 else "negative"
    return [
        {
            "id": "G_COORDINATE_SYSTEM",
            "text": "The camera coordinate system has +X toward image right, +Y downward, and +Z forward into the scene.",
            "source": provenance,
            "depends_on": [],
            "fields": ["camera_coordinate_system"],
        },
        {
            "id": "G_PERSON_POSITION",
            "text": f"The person's {approximate} position is {_display_vector(display.person_position)}.",
            "source": provenance,
            "depends_on": ["G_COORDINATE_SYSTEM"],
            "fields": ["person_position"],
        },
        {
            "id": "G_OBJECT_POSITION",
            "text": f"The {object_name}'s {approximate} position is {_display_vector(display.object_position)}.",
            "source": provenance,
            "depends_on": ["G_COORDINATE_SYSTEM"],
            "fields": ["object_position"],
        },
        {
            "id": "G_PERSON_FORWARD_AXIS",
            "text": f"The person's {approximate} forward axis is {_display_vector(display.person_forward)}.",
            "source": provenance,
            "depends_on": ["G_COORDINATE_SYSTEM"],
            "fields": ["person_forward"],
        },
        {
            "id": "G_PERSON_RIGHT_AXIS",
            "text": f"The person's {approximate} right axis is {_display_vector(display.person_right)}.",
            "source": provenance,
            "depends_on": ["G_COORDINATE_SYSTEM"],
            "fields": ["person_right"],
        },
        {
            "id": "G_DISPLACEMENT",
            "text": f"The displacement from the person to the {object_name} is {_display_vector(display.relative_vector)}.",
            "source": "derived_from_clean_gt" if provenance == "clean_blender_gt" else "derived_from_perturbed_geometry",
            "depends_on": ["G_PERSON_POSITION", "G_OBJECT_POSITION"],
            "fields": ["relative_vector"],
        },
        {
            "id": "G_FORWARD_PROJECTION",
            "text": f"The displacement projects approximately {display.forward_projection:.3f} onto the person's forward axis.",
            "source": "derived_from_clean_gt" if provenance == "clean_blender_gt" else "derived_from_perturbed_geometry",
            "depends_on": ["G_DISPLACEMENT", "G_PERSON_FORWARD_AXIS"],
            "fields": ["forward_projection"],
        },
        {
            "id": "G_RIGHT_PROJECTION",
            "text": f"The displacement projects approximately {display.right_projection:.3f} onto the person's right axis.",
            "source": "derived_from_clean_gt" if provenance == "clean_blender_gt" else "derived_from_perturbed_geometry",
            "depends_on": ["G_DISPLACEMENT", "G_PERSON_RIGHT_AXIS"],
            "fields": ["right_projection"],
        },
        {
            "id": "G_DOMINANT_COMPONENT",
            "text": comparison,
            "source": "derived_from_clean_gt" if provenance == "clean_blender_gt" else "derived_from_perturbed_geometry",
            "depends_on": ["G_FORWARD_PROJECTION", "G_RIGHT_PROJECTION"],
            "fields": ["dominant_component"],
        },
        {
            "id": "G_SPATIAL_CONCLUSION",
            "text": f"The dominant {dominant} component has the {sign} sign, placing the {object_name} {relation}.",
            "source": "derived_from_clean_gt" if provenance == "clean_blender_gt" else "derived_from_perturbed_geometry",
            "depends_on": ["G_DOMINANT_COMPONENT", "G_FORWARD_PROJECTION", "G_RIGHT_PROJECTION"],
            "fields": ["computed_relation"],
        },
    ]


def _looks_geometry_dependent(text: str) -> bool:
    lowered = text.lower()
    return bool(
        re.search(r"\[[^]]+\]|[-+]?\d+\.\d+", text)
        or any(word in lowered for word in ("coordinate", "projection", "axis", "forward", "behind", "left", "right", "displacement"))
    )


def _is_answer_option_analysis(text: str) -> bool:
    lowered = text.lower()
    return bool(
        re.search(r"\b(?:option|choice|answer)\s*[abcd]\b", lowered)
        or re.search(r"\b(?:correct answer|choose|select)\b", lowered)
    )


def _sanitize_llm_units(parsed: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_units = parsed.get("evidence", [])
    if not isinstance(raw_units, list):
        raise ValueError("consolidator JSON field 'evidence' must be a list")
    source_ids = {str(unit.get("id")) for unit in raw_units if isinstance(unit, dict)}
    units = []
    dropped = []
    for index, raw in enumerate(raw_units, start=1):
        if not isinstance(raw, dict) or not isinstance(raw.get("text"), str) or not raw["text"].strip():
            dropped.append({"claim_id": f"COT_{index:03d}", "reason": "malformed_consolidator_claim"})
            continue
        if _is_answer_option_analysis(raw["text"]):
            dropped.append({"claim_id": f"COT_{index:03d}", "reason": "answer_option_analysis"})
            continue
        original_id = str(raw.get("id", f"COT_{index:03d}"))
        depends = raw.get("depends_on", [])
        if not isinstance(depends, list):
            dropped.append({"claim_id": f"COT_{index:03d}", "reason": "depends_on_not_a_list"})
            continue
        units.append({
            "id": f"COT_{index:03d}",
            "text": raw["text"].strip(),
            "source": "trajectory_cot",
            "depends_on": [str(value) for value in depends if str(value) in source_ids or str(value).startswith("G_")],
            "fields": [str(value) for value in raw.get("fields", [])] if isinstance(raw.get("fields", []), list) else [],
            "original_id": original_id,
        })
    # The local model can reference its own IDs before they are normalized.
    id_map = {unit["original_id"]: unit["id"] for unit in units}
    for unit in units:
        unit["depends_on"] = [id_map.get(value, value) for value in unit["depends_on"]]
        unit.pop("original_id", None)
    return units, dropped


def consolidate_record(
    item: dict[str, Any],
    parsed_output: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    cot_units, dropped = _sanitize_llm_units(parsed_output)
    canonical = canonical_geometry_evidence(
        item["scene"]["object_name"], item["clean_display"], "clean_blender_gt"
    )
    return {
        "parent_trajectory_id": item["trajectory_id"],
        "image_path": item["image_path"],
        "scene_group_id": item["scene_group_id"],
        "camera_view": item["camera_view"],
        "object_name": item["scene"]["object_name"],
        "split": item["split"],
        "correct_relation": item["clean_display"].computed_relation,
        "clean_gt_raw": evidence_dict(item["clean_raw"]),
        "clean_gt_display": evidence_dict(item["clean_display"]),
        "units": canonical + cot_units,
        "consolidator_dropped_claims": dropped,
        "raw_trajectory_chars": item["raw_trajectory_chars"],
        "generation": generation,
        "geometry_provenance": "clean Blender GT; trajectory evidence consolidated from the retained CSV CoT",
    }


def adapt_evidence(
    consolidated: dict[str, Any],
    display: QuantizedEvidence,
    object_name: str,
    augmentation_index: int,
) -> dict[str, Any]:
    """Replace geometry evidence and mark stale CoT claims for rewriting."""
    provenance = "clean_blender_gt" if augmentation_index == 0 else "trajectory-derived, geometry-perturbed supervision"
    canonical = canonical_geometry_evidence(object_name, display, provenance)
    canonical_ids = {unit["id"] for unit in canonical}
    changed_ids = set() if augmentation_index == 0 else canonical_ids - {"G_COORDINATE_SYSTEM"}
    recomputed_ids = set() if augmentation_index == 0 else {
        "G_DISPLACEMENT",
        "G_FORWARD_PROJECTION",
        "G_RIGHT_PROJECTION",
        "G_DOMINANT_COMPONENT",
        "G_SPATIAL_CONCLUSION",
    }
    clean_units = {unit["id"]: unit for unit in consolidated["units"]}
    dropped = list(consolidated.get("consolidator_dropped_claims", []))
    known_ids = set(clean_units) | canonical_ids
    cot_units = []
    rewrite_ids = set()
    for unit in consolidated["units"]:
        if unit["id"] in canonical_ids:
            continue
        depends = set(unit.get("depends_on", []))
        if changed_ids and not depends and _looks_geometry_dependent(unit["text"]):
            rewrite_ids.add(unit["id"])
        unknown = sorted(depends - known_ids)
        if unknown:
            dropped.append({"claim_id": unit["id"], "reason": f"unknown_dependencies:{unknown}"})
            continue
        if depends & changed_ids:
            rewrite_ids.add(unit["id"])
        cot_units.append(unit)

    # Rewrite transitive dependants in the same single request.
    changed = True
    while changed:
        changed = False
        for unit in cot_units:
            if unit["id"] not in rewrite_ids and set(unit.get("depends_on", [])) & rewrite_ids:
                rewrite_ids.add(unit["id"])
                changed = True
    return {
        "units": canonical + cot_units,
        "changed_evidence_ids": sorted(changed_ids),
        "recomputed_evidence_ids": sorted(recomputed_ids),
        "claims_to_rewrite": sorted(rewrite_ids),
        "rewritten_claim_ids": [],
        "dropped_claims": dropped,
        "geometry_provenance": provenance,
        "clean_geometry_reference": consolidated["clean_gt_display"],
        "adapted_geometry": evidence_dict(display),
    }


def apply_rewritten_claims(
    adapted: dict[str, Any], parsed_output: dict[str, Any]
) -> dict[str, Any]:
    """Replace only claim text; keep the audited IDs and dependency graph."""
    expected = set(adapted.get("claims_to_rewrite", []))
    raw_claims = parsed_output.get("rewritten_claims")
    if not isinstance(raw_claims, list):
        raise ValueError("rewrite JSON field 'rewritten_claims' must be a list")
    replacements = {}
    for claim in raw_claims:
        if not isinstance(claim, dict):
            raise ValueError("rewritten claim must be an object")
        claim_id = str(claim.get("id", ""))
        text = claim.get("text")
        if claim_id in replacements or not isinstance(text, str) or not text.strip():
            raise ValueError("rewritten claims need unique IDs and non-empty text")
        replacements[claim_id] = text.strip()
    if set(replacements) != expected:
        raise ValueError(
            f"rewritten claim IDs {sorted(replacements)} do not match expected {sorted(expected)}"
        )
    units = []
    for unit in adapted["units"]:
        updated = dict(unit)
        if unit["id"] in replacements:
            updated["text"] = replacements[unit["id"]]
            updated["source"] = "trajectory_cot_geometry_rewritten"
        units.append(updated)
    result = dict(adapted)
    result["units"] = units
    result["rewritten_claim_ids"] = sorted(expected)
    return result


def assistant_targets(
    object_name: str,
    display: QuantizedEvidence,
    reasoning_chain: str,
    answer_letter: str,
) -> dict[str, str]:
    template = verbalize_evidence_template(object_name, display)
    return {
        "answer_only": answer_letter,
        "template_reasoning": f"{template}\n{answer_letter}",
        "trajectory_reasoning": f"{reasoning_chain.strip()}\n{answer_letter}",
    }


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
