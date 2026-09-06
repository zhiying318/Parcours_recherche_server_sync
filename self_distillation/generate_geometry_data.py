#!/usr/bin/env python3
"""Build clean/noisy geometry-privileged records from successful test07 runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST07_ROOT = REPO_ROOT / "comfort_addionalprompt_tests/test07_camera_geometry_before_question"
DEFAULT_SOURCE_ROOT = REPO_ROOT / "COMFORT/data/comfort_human_car_geometry_gt/comfort_human_car"
DEFAULT_PROMPT_INFO = TEST07_ROOT / "data/prompt_info.json"
DEFAULT_RESULTS_CSV = TEST07_ROOT / "results_preciseprompt/mcq_long_qwen3_5vl_thinking.csv"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "data"
RELATION_TEXT = {
    "front": "in front of the person", "behind": "behind the person",
    "left": "to the left of the person", "right": "to the right of the person",
}
OPTION_TEXT = {
    "front": "in front of them",
    "behind": "behind them",
    "left": "on their left",
    "right": "on their right",
}
RELATIONS = tuple(OPTION_TEXT)


def dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def add(left, right):
    return [a + b for a, b in zip(left, right)]


def subtract(left, right):
    return [a - b for a, b in zip(left, right)]


def scale(vector, factor):
    return [value * factor for value in vector]


def normalize(vector):
    return scale(vector, 1.0 / math.sqrt(dot(vector, vector)))


def cross(left, right):
    return [left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0]]


def rotate(vector, axis, angle):
    cosine, sine = math.cos(angle), math.sin(angle)
    return add(add(scale(vector, cosine), scale(cross(axis, vector), sine)),
               scale(axis, dot(axis, vector) * (1.0 - cosine)))


def perturb_axes(right, forward, angle_sigma_degrees, rng):
    rotation_axis = normalize(cross(right, forward))
    angle = math.radians(rng.gauss(0.0, angle_sigma_degrees))
    noisy_forward = normalize(rotate(forward, rotation_axis, angle))
    rotated_right = rotate(right, rotation_axis, angle)
    noisy_right = normalize(subtract(
        rotated_right, scale(noisy_forward, dot(rotated_right, noisy_forward))))
    return noisy_right, noisy_forward


def classify_geometry(geometry):
    relative = subtract(geometry["object_position"], geometry["person_position"])
    forward_projection = dot(relative, geometry["person_forward"])
    right_projection = dot(relative, geometry["person_right"])
    if abs(forward_projection) >= abs(right_projection):
        relation = "front" if forward_projection > 0.0 else "behind"
    else:
        relation = "right" if right_projection > 0.0 else "left"
    return relation, relative, forward_projection, right_projection


def split_object_names(object_names, seed, train_fraction, val_fraction):
    shuffled = sorted(object_names)
    random.Random(seed).shuffle(shuffled)
    train_end = int(len(shuffled) * train_fraction)
    val_end = train_end + int(len(shuffled) * val_fraction)
    return {"train": set(shuffled[:train_end]),
            "val": set(shuffled[train_end:val_end]),
            "test": set(shuffled[val_end:])}


def clean_geometry(scene):
    return {
        "person_position": list(scene["human_visible_center_camera"]),
        "object_position": list(scene["object_visible_center_camera"]),
        "person_right": normalize(list(scene["human_frame_camera"]["right_axis"])),
        "person_forward": normalize(list(scene["human_frame_camera"]["forward_axis"])),
    }


def round_geometry(geometry, digits=3):
    """Quantize noisy geometry before it is used for labels or teacher text."""
    def rounded(value):
        result = round(value, digits)
        return 0.0 if result == 0.0 else result

    return {
        name: [rounded(value) for value in vector]
        for name, vector in geometry.items()
    }


def perturb_geometry(geometry, position_sigma, angle_sigma_degrees, rng):
    noisy_right, noisy_forward = perturb_axes(
        geometry["person_right"], geometry["person_forward"], angle_sigma_degrees, rng)
    noisy_geometry = {
        "person_position": [v + rng.gauss(0.0, position_sigma)
                            for v in geometry["person_position"]],
        "object_position": [v + rng.gauss(0.0, position_sigma)
                            for v in geometry["object_position"]],
        "person_right": noisy_right,
        "person_forward": noisy_forward,
    }
    # The teacher receives coordinates at millimetre precision, so all derived
    # labels and reasoning must use the same quantized values.
    return round_geometry(noisy_geometry)


def sample_label_preserving_geometry(
    geometry, position_sigma, angle_sigma_degrees, rng
):
    clean_relation = classify_geometry(geometry)[0]
    attempts = 0
    while True:
        attempts += 1
        candidate = perturb_geometry(
            geometry, position_sigma, angle_sigma_degrees, rng)
        if classify_geometry(candidate)[0] == clean_relation:
            return candidate, attempts


def vector_text(vector, digits=3):
    return "[" + ", ".join(f"{value:.{digits}f}" for value in vector) + "]"


def make_teacher_geometry(object_name, geometry):
    return (
        "Consider that the picture was taken from the origin [0.000, 0.000, 0.000] "
        "of a camera coordinate system, where +X points to the image's right, +Y "
        "points downward, and +Z points forward into the scene. The person in the "
        f"image is located at {vector_text(geometry['person_position'])} and is looking "
        f"in the direction of {vector_text(geometry['person_forward'])} (unit vector). "
        f"The {object_name} is located at {vector_text(geometry['object_position'])}."
    )


def make_shuffled_problem(object_name, relation, augmentation_index, sample_seed):
    """Create a balanced, deterministic MCQ for a noisy training record.

    Eight noisy augmentations place the correct semantic relation twice in each
    letter position.  The distractor order is independently shuffled, so the
    resulting option order is still varied within that balanced assignment.
    """
    if augmentation_index < 1:
        raise ValueError("only noisy augmentations have shuffled options")
    positions = list(range(4)) * 2
    random.Random(sample_seed - augmentation_index).shuffle(positions)
    correct_index = positions[(augmentation_index - 1) % len(positions)]

    distractors = [candidate for candidate in RELATIONS if candidate != relation]
    random.Random(sample_seed).shuffle(distractors)
    options = list(distractors)
    options.insert(correct_index, relation)
    answer_letter = chr(ord("A") + correct_index)
    question = (
        f"Where is the {object_name} in the perspective of the person?\n"
        "Choose ONE option and respond with ONLY the letter."
    )
    choices = "\n".join(
        f"{chr(ord('A') + index)}. From the person's perspective, the {object_name} "
        f"is {OPTION_TEXT[option]}."
        for index, option in enumerate(options)
    )
    return question + "\n" + choices, answer_letter


def read_successful_rows(results_csv):
    with results_csv.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    return [row for row in rows
            if row["pred_letter"].strip() == row["correct_letter"].strip()]


def make_record(row, scene_path, scene, clean_teacher_geometry, split,
                augmentation_index, sample_seed, position_sigma,
                angle_sigma_degrees):
    source_geometry = clean_geometry(scene)
    is_noisy = augmentation_index > 0
    geometry, noise_attempts = source_geometry, 0
    if is_noisy:
        geometry, noise_attempts = sample_label_preserving_geometry(
            source_geometry, position_sigma, angle_sigma_degrees,
            random.Random(sample_seed))
    relation = classify_geometry(geometry)[0]
    student_problem = row["mcq_prompt"][len(clean_teacher_geometry) + 1:]
    answer_letter = row["correct_letter"].strip()
    if is_noisy:
        student_problem, answer_letter = make_shuffled_problem(
            scene["object_name"], relation, augmentation_index, sample_seed)
    teacher_geometry = (make_teacher_geometry(scene["object_name"], geometry)
                        if is_noisy else clean_teacher_geometry)
    image_path = row["image_path"].removeprefix("./")
    record_id = f"{scene_path.parent.name}__aug{augmentation_index:02d}"
    return {
        "id": record_id,
        "split": split,
        "object_name": scene["object_name"],
        "source_scene_gt": scene_path.relative_to(REPO_ROOT).as_posix(),
        "image_path": image_path,
        "is_noisy": is_noisy,
        "reference_source": "noisy_geometry" if is_noisy else "test07_clean_geometry",
        "seed": sample_seed,
        "noise_attempts": noise_attempts,
        "sigma": {"position": position_sigma if is_noisy else 0.0,
                  "angle_degrees": angle_sigma_degrees if is_noisy else 0.0},
        "clean_geometry": source_geometry,
        "teacher_geometry_values": geometry,
        "relation": relation,
        "correct_letter": answer_letter,
        "problem": student_problem,
        "teacher_geometry": teacher_geometry,
        "messages": [
            {"role": "user", "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": student_problem}]},
        ],
    }


def generate_records(source_root, prompt_info_path, results_csv, seed,
                     train_augmentations, position_sigma, angle_sigma_degrees,
                     train_fraction, val_fraction):
    prompt_info = json.loads(prompt_info_path.read_text(encoding="utf-8"))
    rows = read_successful_rows(results_csv)
    object_splits = split_object_names(
        sorted({row["second_object"] for row in rows}), seed,
        train_fraction, val_fraction)
    scenes = {}
    for scene_path in sorted(source_root.glob("*/*/scene_gt.json")):
        image_path = (scene_path.parent / "0.png").relative_to(REPO_ROOT).as_posix()
        scenes[image_path] = (scene_path,
                              json.loads(scene_path.read_text(encoding="utf-8")))

    records = {"train": [], "val": [], "test": []}
    sample_number = 0
    for row in rows:
        image_path = row["image_path"].removeprefix("./")
        clean_teacher_geometry = prompt_info[image_path]
        if not row["mcq_prompt"].startswith(clean_teacher_geometry + "\n"):
            raise ValueError(f"test07 prefix mismatch: {image_path}")
        scene_path, scene = scenes[image_path]
        split = next(name for name, objects in object_splits.items()
                     if row["second_object"] in objects)
        augmentation_count = train_augmentations if split == "train" else 0
        for augmentation_index in range(augmentation_count + 1):
            records[split].append(make_record(
                row, scene_path, scene, clean_teacher_geometry, split,
                augmentation_index, seed + sample_number, position_sigma,
                angle_sigma_degrees))
            sample_number += 1
    return records, object_splits


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--prompt-info", type=Path, default=DEFAULT_PROMPT_INFO)
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--train-augmentations", type=int, default=8)
    parser.add_argument("--position-sigma", type=float, default=0.05)
    parser.add_argument("--angle-sigma-degrees", type=float, default=2.0)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    return parser.parse_args()


def main():
    args = parse_args()
    records, object_splits = generate_records(
        args.source_root, args.prompt_info, args.results_csv, args.seed,
        args.train_augmentations, args.position_sigma, args.angle_sigma_degrees,
        args.train_fraction, args.val_fraction)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for split, split_records in records.items():
        write_jsonl(args.output_root / f"{split}.jsonl", split_records)
    manifest = {
        "source_results": args.results_csv.relative_to(REPO_ROOT).as_posix(),
        "source_rows": 144,
        "correct_source_rows": sum(not r["is_noisy"]
                                   for values in records.values() for r in values),
        "seed": args.seed,
        "train_augmentations": args.train_augmentations,
        "position_sigma": args.position_sigma,
        "angle_sigma_degrees": args.angle_sigma_degrees,
        "object_splits": {split: sorted(names)
                          for split, names in object_splits.items()},
        "record_counts": {split: len(values) for split, values in records.items()},
        "train_clean_count": sum(not r["is_noisy"] for r in records["train"]),
        "train_noisy_count": sum(r["is_noisy"] for r in records["train"]),
    }
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
