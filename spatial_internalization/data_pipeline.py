"""Independent source-data discovery, splitting, augmentation, and JSONL IO."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any

from .geometry import (
    GeometryError,
    GeometryEvidence,
    QuantizedEvidence,
    SOURCE_RELATION_MAP,
    classify_projections,
    extract_canonical_evidence,
    evidence_dict,
    load_raw_sample,
    quantize_and_verify_evidence,
)
from .verbalizer import verbalize_evidence_template

RELATIONS = ("front", "behind", "left", "right")
OPTION_TEXT = {
    "front": "in front of them",
    "behind": "behind them",
    "left": "on their left",
    "right": "on their right",
}
CAMERA_NAMES = ("back", "front", "left", "right")


def stable_image_seed(image_path: str, seed: int) -> int:
    digest = hashlib.sha256(image_path.encode("utf-8")).digest()
    return seed + int.from_bytes(digest[:8], "big", signed=False)


def _camera_name(path: Path) -> str:
    match = re.search(r"__cam_([^/]+)$", path.parent.name)
    return match.group(1) if match else "unknown"


def scene_group_id(scene_path: Path) -> str:
    """Group four camera views of one object/relation configuration together."""
    stem = re.sub(r"__cam_[^/]+$", "", scene_path.parent.name)
    return f"{scene_path.parent.parent.name}/{stem}"


def discover_source_samples(repo_root: Path, source_root: Path) -> list[dict[str, Any]]:
    samples = []
    for scene_path in sorted(source_root.glob("*/*/scene_gt.json")):
        scene = load_raw_sample(scene_path)
        image_path = scene_path.parent / "0.png"
        if not image_path.exists():
            raise GeometryError(f"missing source image: {image_path}")
        samples.append({
            "scene_path": scene_path,
            "scene": scene,
            "image_path": image_path,
            "image_relpath": image_path.relative_to(repo_root).as_posix(),
            "group_id": scene_group_id(scene_path),
            "camera_view": _camera_name(scene_path),
        })
    return samples


def stratified_group_split(samples: list[dict[str, Any]], seed: int, ratios=(0.8, 0.1, 0.1)) -> dict[str, str]:
    """Split groups without crossing object/relation scenes; stratify by source class."""
    groups: dict[str, dict[str, Any]] = {}
    for sample in samples:
        group = groups.setdefault(sample["group_id"], {"relation": sample["scene"]["relation"], "samples": []})
        group["samples"].append(sample)
    by_relation: dict[str, list[str]] = {}
    for group_id, group in groups.items():
        by_relation.setdefault(group["relation"], []).append(group_id)
    assignments: dict[str, str] = {}
    rng = random.Random(seed)
    for relation, group_ids in sorted(by_relation.items()):
        group_ids = sorted(group_ids)
        rng.shuffle(group_ids)
        n = len(group_ids)
        n_train = max(1, int(round(n * ratios[0])))
        n_val = max(1, int(round(n * ratios[1])))
        if n_train + n_val >= n:
            n_train, n_val = n - 2, 1
        for group_id in group_ids[:n_train]:
            assignments[group_id] = "train"
        for group_id in group_ids[n_train:n_train + n_val]:
            assignments[group_id] = "validation"
        for group_id in group_ids[n_train + n_val:]:
            assignments[group_id] = "test"
    return assignments


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _rotate(v, axis, angle):
    c, s = math.cos(angle), math.sin(angle)
    cross = _cross(axis, v)
    dot = sum(x * y for x, y in zip(axis, v))
    return tuple(v[i] * c + cross[i] * s + axis[i] * dot * (1 - c) for i in range(3))


def _normalize(v):
    length = math.sqrt(sum(x * x for x in v))
    return tuple(x / length for x in v)


def perturb_geometry(evidence: GeometryEvidence, position_sigma: float, angle_sigma_degrees: float, rng: random.Random) -> dict[str, tuple[float, float, float]]:
    """Independent reproduction of the old Gaussian position/yaw perturbation.

    The implementation is local to this experiment.  It rejects a draw later
    if the recomputed relation no longer equals the clean source relation.
    """
    rotation_axis = _normalize(_cross(evidence.person_right, evidence.person_forward))
    angle = math.radians(rng.gauss(0.0, angle_sigma_degrees))
    forward = _normalize(_rotate(evidence.person_forward, rotation_axis, angle))
    rotated_right = _rotate(evidence.person_right, rotation_axis, angle)
    projection = sum(x * y for x, y in zip(rotated_right, forward))
    right = _normalize(tuple(rotated_right[i] - projection * forward[i] for i in range(3)))
    person = tuple(x + rng.gauss(0.0, position_sigma) for x in evidence.person_position)
    obj = tuple(x + rng.gauss(0.0, position_sigma) for x in evidence.object_position)
    return {"person_position": person, "object_position": obj, "person_forward": forward, "person_right": right}


def _evidence_from_geometry(geometry: dict[str, tuple[float, float, float]], clean: GeometryEvidence) -> GeometryEvidence:
    person, obj = geometry["person_position"], geometry["object_position"]
    relative = tuple(obj[i] - person[i] for i in range(3))
    f = sum(relative[i] * geometry["person_forward"][i] for i in range(3))
    r = sum(relative[i] * geometry["person_right"][i] for i in range(3))
    return GeometryEvidence(
        person_position=person,
        object_position=obj,
        person_forward=geometry["person_forward"],
        person_right=geometry["person_right"],
        relative_vector=relative,
        forward_projection=f,
        right_projection=r,
        computed_relation=classify_projections(f, r),
        source_relation_raw=clean.source_relation_raw,
        source_relation_canonical=clean.source_relation_canonical,
        source_relation_match=True,
        scale=person[2],
        orthogonality_dot=sum(x * y for x, y in zip(geometry["person_forward"], geometry["person_right"])),
    )


def make_label_preserving_noisy(clean: GeometryEvidence, seed: int, position_sigma: float, angle_sigma_degrees: float, max_attempts: int = 1000) -> tuple[GeometryEvidence, int]:
    for attempt in range(1, max_attempts + 1):
        candidate = _evidence_from_geometry(
            perturb_geometry(clean, position_sigma, angle_sigma_degrees, random.Random(seed + attempt)), clean
        )
        if candidate.scale > 0 and candidate.computed_relation == clean.computed_relation:
            return candidate, attempt
    raise GeometryError(f"could not draw label-preserving noise in {max_attempts} attempts")


def format_user_question(question: str, choices: dict[str, str]) -> str:
    """Serialize the one user prompt seen by the model."""
    return question + "\nChoose ONE option and respond with ONLY the letter.\n" + "\n".join(
        f"{letter}. {choices[letter]}" for letter in "ABCD"
    )


def make_options(object_name: str, relation: str, image_path: str, seed: int) -> tuple[str, str, dict[str, str]]:
    keys = list(RELATIONS)
    random.Random(stable_image_seed(image_path, seed)).shuffle(keys)
    letters = "ABCD"
    option_map = {
        letter: f"From the person's perspective, the {object_name} is {OPTION_TEXT[key]}."
        for letter, key in zip(letters, keys)
    }
    correct_letter = letters[keys.index(relation)]
    question = f"Where is the {object_name} in the perspective of the person?"
    return question, correct_letter, option_map


def _record_id(sample: dict[str, Any], augmentation_index: int) -> str:
    return f"{sample['group_id'].replace('/', '__')}__{sample['camera_view']}__aug{augmentation_index:02d}"


def build_sft_example(sample: dict[str, Any], split: str, source_evidence: GeometryEvidence, display: QuantizedEvidence, seed: int, augmentation_index: int, variant: str) -> dict[str, Any]:
    question, letter, options = make_options(
        sample["scene"]["object_name"], source_evidence.computed_relation,
        sample["image_relpath"], seed + augmentation_index * 10007,
    )
    reasoning = verbalize_evidence_template(sample["scene"]["object_name"], display)
    target = letter if variant == "answer_only" else f"{reasoning}\n{letter}"
    return {
        "id": _record_id(sample, augmentation_index),
        "split": split,
        "variant": variant,
        "object_name": sample["scene"]["object_name"],
        "image_path": sample["image_relpath"],
        "scene_gt_path": sample["scene_path"].relative_to(sample["repo_root"]).as_posix(),
        "scene_group_id": sample["group_id"],
        "camera_view": sample["camera_view"],
        "question": question,
        "choices": options,
        "relation": source_evidence.computed_relation,
        "answer_letter": letter,
        "assistant_target": target,
        "is_noisy": augmentation_index > 0,
        "augmentation_index": augmentation_index,
        "geometry": evidence_dict(display),
        "audit": {
            "source_geometry": evidence_dict(source_evidence),
            "display_geometry": evidence_dict(display),
        },
    }


def object_fold_split(samples: list[dict[str, Any]], seed: int, fold: int) -> tuple[dict[str, str], dict[str, Any]]:
    """Leave one object out for test and the next object out for validation."""
    objects = sorted({sample["scene"]["object_name"] for sample in samples})
    if len(objects) < 3:
        raise ValueError("object cross-validation requires at least three objects")
    if not 0 <= fold < len(objects):
        raise ValueError(f"fold must be between 0 and {len(objects) - 1}")
    random.Random(seed).shuffle(objects)
    object_assignments = {name: "train" for name in objects}
    object_assignments[objects[fold]] = "test"
    object_assignments[objects[(fold + 1) % len(objects)]] = "validation"
    assignments = {}
    for sample in samples:
        split = object_assignments[sample["scene"]["object_name"]]
        previous = assignments.setdefault(sample["group_id"], split)
        if previous != split:
            raise ValueError("one source group contains objects assigned to different splits")
    return assignments, {
        "fold": fold,
        "num_folds": len(objects),
        "object_order": objects,
        "object_assignments": object_assignments,
        "objects_by_split": {split: [name for name in objects if object_assignments[name] == split]
                             for split in ("train", "validation", "test")},
    }


def build_datasets(repo_root: Path, source_root: Path, seed: int = 20260919, augmentations: int = 8, position_sigma: float = 0.05, angle_sigma_degrees: float = 2.0, object_fold: int | None = None) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    samples = discover_source_samples(repo_root, source_root)
    for sample in samples:
        sample["repo_root"] = repo_root
    fold_metadata = None
    if object_fold is None:
        assignments = stratified_group_split(samples, seed)
    else:
        assignments, fold_metadata = object_fold_split(samples, seed, object_fold)
    records = {"train": [], "validation": [], "test": []}
    audit = {"valid": [], "excluded": [], "boundary": []}
    clean_by_sample = {}
    for sample in samples:
        try:
            clean = extract_canonical_evidence(sample["scene"])
            if not clean.source_relation_match:
                raise GeometryError("computed relation disagrees with source relation")
            display = quantize_and_verify_evidence(clean)
            clean_by_sample[sample["image_relpath"]] = (clean, display)
            audit["valid"].append({"image_path": sample["image_relpath"], "scene_group_id": sample["group_id"], "source": evidence_dict(clean), "display": evidence_dict(display)})
            if display.digits > 3 or display.normalized_margin < 0.05:
                audit["boundary"].append({"image_path": sample["image_relpath"], "scene_group_id": sample["group_id"], "digits": display.digits, "normalized_margin": display.normalized_margin})
        except GeometryError as exc:
            audit["excluded"].append({"image_path": sample["image_relpath"], "scene_group_id": sample["group_id"], "reason": str(exc)})
    for sample in samples:
        key = sample["image_relpath"]
        if key not in clean_by_sample:
            continue
        clean, display = clean_by_sample[key]
        split = assignments[sample["group_id"]]
        count = augmentations + 1 if split == "train" else 1
        for index in range(count):
            current, current_display = clean, display
            attempts = 0
            if index:
                current, attempts = make_label_preserving_noisy(clean, seed + index * 1000003 + len(records["train"]), position_sigma, angle_sigma_degrees)
                try:
                    current_display = quantize_and_verify_evidence(current)
                except GeometryError as exc:
                    audit["excluded"].append({"image_path": key, "reason": f"noisy display: {exc}", "augmentation_index": index})
                    continue
            for variant in ("answer_only", "geometry_reasoning"):
                record = build_sft_example(sample, split, current, current_display, seed, index, variant)
                record["audit"]["clean_source_geometry"] = evidence_dict(clean)
                record["audit"]["noise_attempts"] = attempts
                record["audit"]["noise_parameters"] = {"position_sigma": position_sigma if index else 0.0, "angle_sigma_degrees": angle_sigma_degrees if index else 0.0}
                records[split].append(record)
                records[split][-1]["noise_attempts"] = attempts
    # Make train/validation/test exactly shared by ID across A/B, while each
    # variant remains a separate JSONL file.
    by_variant = {variant: {split: [] for split in records} for variant in ("answer_only", "geometry_reasoning")}
    for split, values in records.items():
        for value in values:
            by_variant[value["variant"]][split].append(value)
    manifest = {
        "experiment": "spatial_internalization",
        "source_root": source_root.relative_to(repo_root).as_posix(),
        "source_contract": "COMFORT scene_gt.json contract v1.0; visible-center medians from depth/masks",
        "source_relation_field": "relation",
        "source_relation_normalization": SOURCE_RELATION_MAP,
        "seed": seed,
        "split_unit": "object_relation_view_group (4 camera views kept together; no independent scene_id in source)",
        "split_ratios": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "group_counts": {split: len({r["scene_group_id"] for r in values if r["variant"] == "answer_only"}) for split, values in records.items()},
        "source_sample_count": len(samples),
        "valid_sample_count": len(audit["valid"]),
        "excluded_sample_count": len(audit["excluded"]),
        "augmentation_count_train": augmentations,
        "noise": {"position_sigma": position_sigma, "angle_sigma_degrees": angle_sigma_degrees, "label_preserving_rejection": True},
        "record_counts": {variant: {split: len(values) for split, values in splits.items()} for variant, splits in by_variant.items()},
        "answer_format": "final line is one of A/B/C/D; option text matches baseline long MCQ",
        "reasoning_format": "plain assistant target; no explicit <think> wrapper because Qwen3.5 enable_thinking=False is used consistently",
    }
    if fold_metadata is not None:
        manifest.update(fold_metadata)
        manifest["split_unit"] = "object (all relations, camera views and augmentations kept together)"
        manifest["split_ratios"] = {
            split: len(names) / fold_metadata["num_folds"]
            for split, names in fold_metadata["objects_by_split"].items()
        }
    return by_variant, {"manifest": manifest, "audit": audit, "groups": assignments}


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
