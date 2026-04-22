"""
Generate image_quads.json for test03_pov_four evaluation.

3-choice MCQ: correct answer is always cam_pov_front (person faces forward).
Options:
  1. cam_pov_front              (correct)
  2. cam_pov_{third_obj_pos}    (where third object is)
  3. cam_pov_{correct_relation} (where ref/second object is)

Edge cases (both resolved by picking first unused direction from fallback order
[cam_pov_back, cam_pov_left, cam_pov_right]):
  - correct_relation == "infrontof": option3 would duplicate option1 → use fallback
  - third_pos == "infrontof":        option2 would duplicate option1 → use fallback

Run:
    python comfort_eval_tests/test03_pov_four/generate_quads.py \
        --data_root COMFORT/data/comfort_human_car_pov_two \
        --output comfort_eval_tests/test03_pov_four/image_quads.json
"""

import os
import json
import argparse

EXTERNAL_CAMS = ["cam_back", "cam_front", "cam_left", "cam_right"]

CORRECT_CAM = "cam_pov_front"  # always

RELATION_TO_POV_CAM = {
    "infrontof": "cam_pov_front",
    "behind":    "cam_pov_back",
    "totheleft": "cam_pov_left",
    "totheright":"cam_pov_right",
}

# Fixed fallback order (non-front, alphabetical) for conflict resolution
FALLBACK_DISTRACTORS = ["cam_pov_back", "cam_pov_left", "cam_pov_right"]

RELATION_MAP = {
    "infrontof": "front",
    "totheleft": "left",
    "totheright": "right",
    "behind": "behind",
}


def resolve_three_options(relation_dir, third_pos):
    """
    Returns (cam_dist1, cam_dist2) — the two distractor cam names.
    cam_dist1 corresponds to third object position,
    cam_dist2 corresponds to ref object position.
    Both are guaranteed to differ from CORRECT_CAM and from each other.
    """
    cam_secondobj = RELATION_TO_POV_CAM[relation_dir]
    cam_thirdobj  = RELATION_TO_POV_CAM[third_pos]

    # Fix cam_secondobj if it duplicates the correct cam
    # (happens when correct_relation == "infrontof")
    if cam_secondobj == CORRECT_CAM:
        cam_secondobj = next(c for c in FALLBACK_DISTRACTORS if c != cam_thirdobj)

    # Fix cam_thirdobj if it duplicates the correct cam
    # (happens when third_pos == "infrontof")
    if cam_thirdobj == CORRECT_CAM:
        cam_thirdobj = next(c for c in FALLBACK_DISTRACTORS if c != cam_secondobj)

    return cam_thirdobj, cam_secondobj


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True,
                        help="Path to comfort_human_car_pov_two root")
    parser.add_argument("--output", required=True,
                        help="Output JSON path")
    return parser.parse_args()


def main():
    args = parse_args()
    quads = []

    for relation_dir in sorted(os.listdir(args.data_root)):
        rel_path = os.path.join(args.data_root, relation_dir)
        if not os.path.isdir(rel_path):
            continue

        relation_key = RELATION_MAP.get(relation_dir, relation_dir)

        # Group folders by (ref_obj, relation, third_obj, third_pos) scene key
        # Folder: {obj}__{relation}__{cam}__third_{third_obj}_at_{third_pos}
        scenes = {}
        for folder in sorted(os.listdir(rel_path)):
            if not os.path.isdir(os.path.join(rel_path, folder)):
                continue
            parts = folder.split("__")
            if len(parts) != 4:
                continue
            obj_name, rel, cam_name, third_part = parts
            if not third_part.startswith("third_"):
                continue
            after_third = third_part[len("third_"):]
            at_idx = after_third.rfind("_at_")
            if at_idx == -1:
                continue
            third_obj = after_third[:at_idx]
            third_pos = after_third[at_idx + 4:]

            scene_key = (obj_name, rel, third_obj, third_pos)
            scenes.setdefault(scene_key, {})[cam_name] = os.path.join(
                args.data_root, relation_dir, folder, "0.png"
            )

        for (obj_name, rel, third_obj, third_pos), cam_map in scenes.items():
            cam_dist1, cam_dist2 = resolve_three_options(relation_dir, third_pos)

            pov_correct = cam_map.get(CORRECT_CAM)
            pov_dist1   = cam_map.get(cam_dist1)
            pov_dist2   = cam_map.get(cam_dist2)

            if any(v is None for v in [pov_correct, pov_dist1, pov_dist2]):
                missing = [c for c, v in [(CORRECT_CAM, pov_correct),
                                           (cam_dist1, pov_dist1),
                                           (cam_dist2, pov_dist2)] if v is None]
                print(f"WARNING: missing {missing} for {obj_name}__{rel}__"
                      f"third_{third_obj}_at_{third_pos}, skipping")
                continue

            for ext_cam in EXTERNAL_CAMS:
                ext_img = cam_map.get(ext_cam)
                if ext_img is None:
                    print(f"WARNING: missing {ext_cam} for "
                          f"{obj_name}__{rel}__third_{third_obj}_at_{third_pos}, skipping")
                    continue

                quads.append({
                    "external_img":          ext_img,
                    "pov_correct":           pov_correct,
                    "pov_dist1":             pov_dist1,
                    "pov_dist2":             pov_dist2,
                    "cam_correct":           CORRECT_CAM,
                    "cam_dist1":             cam_dist1,
                    "cam_dist2":             cam_dist2,
                    "cam_view_external":     ext_cam,
                    "second_object":         obj_name,
                    "correct_relation":      relation_key,
                    "third_object":          third_obj,
                    "third_object_relation": RELATION_MAP.get(third_pos, third_pos),
                })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(quads, f, indent=2, ensure_ascii=False)

    print(f"Written {len(quads)} quads to {args.output}")


if __name__ == "__main__":
    main()
