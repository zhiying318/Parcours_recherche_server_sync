#!/usr/bin/env python3
"""Build matched single-image prompt-ablation data from Blender geometry GT."""

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SUITE_ROOT = Path(__file__).resolve().parent
DATASET_ROOT = REPO_ROOT / "COMFORT/data/comfort_human_car_geometry_gt/comfort_human_car"

EXPERIMENTS = (
    "test00_baseline",
    "test01_camera_side",
    "test02_camera_coordinate_system",
    "test03_object_camera_xyz",
    "test04_person_object_camera_xyz",
    "test05_person_forward_axis",
    "test06_camera_coordinate_system_person_forward_axis",
)

SIDE_TEXT = {
    "cam_front": "in front of",
    "cam_back": "behind",
    "cam_left": "to the left of",
    "cam_right": "to the right of",
}


def _vec(values):
    return "[" + ", ".join(f"{float(value):.3f}" for value in values) + "]"


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main():
    sample_dirs = sorted(path.parent for path in DATASET_ROOT.glob("*/*/scene_gt.json"))
    if not sample_dirs:
        raise FileNotFoundError(f"No geometry-GT samples found below {DATASET_ROOT}")

    image_paths = []
    prompt_maps = {name: {} for name in EXPERIMENTS if name != "test00_baseline"}

    for sample_dir in sample_dirs:
        image = sample_dir / "0.png"
        if not image.is_file():
            raise FileNotFoundError(image)
        relative_image = image.relative_to(REPO_ROOT).as_posix()
        config = json.loads((sample_dir / "config.json").read_text(encoding="utf-8"))
        scene = json.loads((sample_dir / "scene_gt.json").read_text(encoding="utf-8"))
        image_paths.append(relative_image)

        view_name = config["view_name"]
        obj = scene["object_visible_center_camera"]
        human = scene["human_visible_center_camera"]
        convention = (
            "In the camera coordinate system, +X points to the image's right, "
            "+Y points downward, and +Z points forward into the scene."
        )
        camera_side = (
            "The viewpoint from which this image was captured is located "
            f"{SIDE_TEXT[view_name]} the depicted person."
        )
        prompt_maps["test01_camera_side"][relative_image] = camera_side
        prompt_maps["test02_camera_coordinate_system"][relative_image] = (
            f"{camera_side} {convention}"
        )
        prompt_maps["test03_object_camera_xyz"][relative_image] = (
            f"{camera_side} {convention} "
            f"The coordinates of the {scene['object_name']} from the camera's perspective are {_vec(obj)}."
        )
        prompt_maps["test04_person_object_camera_xyz"][relative_image] = (
            f"{camera_side} {convention} "
            f"The coordinates of the {scene['object_name']} from the camera's perspective are {_vec(obj)}. "
            f"The coordinates of the person from the camera's perspective are {_vec(human)}."
        )
        axes = scene["human_frame_camera"]
        prompt_maps["test05_person_forward_axis"][relative_image] = (
            f"{camera_side} {convention} "
            f"The coordinates of the {scene['object_name']} from the camera's perspective are {_vec(obj)}. "
            f"The coordinates of the person from the camera's perspective are {_vec(human)}. "
            f"In the camera coordinate system, the person's forward direction is {_vec(axes['forward_axis'])}."
        )
        prompt_maps["test06_camera_coordinate_system_person_forward_axis"][relative_image] = (
            f"{convention} "
            f"In the camera coordinate system, the person's forward direction is {_vec(axes['forward_axis'])}."
        )

    if len(image_paths) != 144:
        raise ValueError(f"Expected 144 matched samples, found {len(image_paths)}")

    for experiment in EXPERIMENTS:
        directory = SUITE_ROOT / experiment
        _write_json(directory / "data/image_paths.json", image_paths)
        if experiment != "test00_baseline":
            prompt_map = prompt_maps[experiment]
            if set(prompt_map) != set(image_paths):
                raise ValueError(f"Prompt map mismatch for {experiment}")
            _write_json(directory / "data/prompt_info.json", prompt_map)
        (directory / "results").mkdir(parents=True, exist_ok=True)

    print(f"Generated {len(image_paths)} matched samples for {len(EXPERIMENTS)} experiments.")


if __name__ == "__main__":
    main()
