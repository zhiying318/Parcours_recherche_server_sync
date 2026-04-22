"""
Generate comfort_image_pairs.json from COMFORT/data/comfort_human_car.

For each (object, relation) combination, collect all 4 camera-view images and
produce all C(4,2) = 6 ordered pairs. Output is saved at comfort_image_pairs.json
in the project root.

Run from /home/zzou:
    python comfort_eval_output/get_comfort_pair_paths_json.py
"""

import os
import json
from itertools import combinations
from collections import defaultdict

COMFORT_ROOT = "COMFORT/data/comfort_human_car"
OUTPUT_JSON = "comfort_image_pairs.json"


def main():
    # Group image paths by (relation, object)
    groups = defaultdict(dict)  # groups[(relation, object)][cam_view] = rel_path

    for relation in sorted(os.listdir(COMFORT_ROOT)):
        rel_dir = os.path.join(COMFORT_ROOT, relation)
        if not os.path.isdir(rel_dir):
            continue
        for scene in sorted(os.listdir(rel_dir)):
            scene_dir = os.path.join(rel_dir, scene)
            img_path = os.path.join(scene_dir, "0.png")
            if not os.path.isfile(img_path):
                continue
            # scene name: {object}__{relation}__{cam_view}
            parts = scene.split("__")
            if len(parts) != 3:
                continue
            obj, _, cam_view = parts
            groups[(relation, obj)][cam_view] = img_path

    pairs = []
    for (relation, obj), views in sorted(groups.items()):
        cam_views = sorted(views.keys())
        for v1, v2 in combinations(cam_views, 2):
            pairs.append({
                "img1": views[v1],
                "img2": views[v2],
                "cam_view_1": v1,
                "cam_view_2": v2,
            })

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(pairs)} pairs to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
