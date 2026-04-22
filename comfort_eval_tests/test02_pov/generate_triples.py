"""
Generate image_triples.json for test02_pov evaluation.

Run AFTER generating the comfort_human_car_pov dataset:
    python comfort_eval_tests/test02_pov/generate_triples.py \
        --data_root COMFORT/data/comfort_human_car_pov \
        --output comfort_eval_tests/test02_pov/image_triples.json

Each triple:
    external_img       : one of the 4 external camera images (cam_back/front/left/right)
    correct_pov_img    : always cam_pov_front (person always faces forward)
    distractor_pov_img : cam_pov_back if object is in front; else POV facing toward object
"""

import os
import json
import argparse

EXTERNAL_CAMS = ["cam_back", "cam_front", "cam_left", "cam_right"]

# Correct POV = always cam_pov_front (person faces -Y regardless of object position)
CORRECT_CAM = "cam_pov_front"

# Distractor:
#   infrontof → cam_pov_back (opposite of front, since object is already in front)
#   behind    → cam_pov_back (facing toward the object, which is behind)
#   totheleft → cam_pov_left (facing toward the object, which is to the left)
#   totheright→ cam_pov_right (facing toward the object, which is to the right)
RELATION_TO_DISTRACT_CAM = {
    "infrontof": "cam_pov_back",
    "behind":    "cam_pov_back",
    "totheleft": "cam_pov_left",
    "totheright":"cam_pov_right",
}

RELATION_MAP = {
    "infrontof": "front",
    "totheleft": "left",
    "totheright": "right",
    "behind": "behind",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True,
                        help="Path to comfort_human_car_pov data root, e.g. COMFORT/data/comfort_human_car_pov")
    parser.add_argument("--output", required=True,
                        help="Output JSON path, e.g. comfort_eval_tests/test02_pov/image_triples.json")
    return parser.parse_args()


def main():
    args = parse_args()
    triples = []

    for relation_dir in sorted(os.listdir(args.data_root)):
        rel_path = os.path.join(args.data_root, relation_dir)
        if not os.path.isdir(rel_path):
            continue

        relation_key = RELATION_MAP.get(relation_dir, relation_dir)

        # Group scene folders by (object, relation) — one shared_yaw per group
        scenes = {}  # key: (obj_name, relation_dir)
        for folder in sorted(os.listdir(rel_path)):
            parts = folder.split("__")
            if len(parts) != 3:
                continue
            obj_name, rel, cam_name = parts
            key = (obj_name, rel)
            scenes.setdefault(key, {})[cam_name] = os.path.join(
                args.data_root, relation_dir, folder, "0.png"
            )

        for (obj_name, rel), cam_map in scenes.items():
            correct_cam   = CORRECT_CAM
            distract_cam  = RELATION_TO_DISTRACT_CAM.get(relation_dir)
            correct_img   = cam_map.get(correct_cam)
            distractor_img = cam_map.get(distract_cam)
            if correct_img is None or distractor_img is None:
                print(f"WARNING: missing POV images for {obj_name}__{rel} "
                      f"(correct={correct_cam}, distract={distract_cam}), skipping")
                continue

            for ext_cam in EXTERNAL_CAMS:
                ext_img = cam_map.get(ext_cam)
                if ext_img is None:
                    print(f"WARNING: missing {ext_cam} for {obj_name}__{rel}, skipping")
                    continue

                triples.append({
                    "external_img": ext_img,
                    "correct_pov_img": correct_img,
                    "distractor_pov_img": distractor_img,
                    "cam_view_external": ext_cam,
                    "cam_view_correct": correct_cam,
                    "cam_view_distractor": distract_cam,
                    "second_object": obj_name,
                    "correct_relation": relation_key,
                })

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(triples, f, indent=2, ensure_ascii=False)

    print(f"Written {len(triples)} triples to {args.output}")


if __name__ == "__main__":
    main()
