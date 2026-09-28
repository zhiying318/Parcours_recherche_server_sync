import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from spatial_internalization.data_pipeline import build_datasets, object_fold_split


def samples():
    result = []
    positions = {"front": [0, 0, 12], "back": [0, 0, 8],
                 "left": [-2, 0, 10], "right": [2, 0, 10]}
    for index in range(9):
        for relation, position in positions.items():
            group = f"{relation}/object{index}__{relation}"
            for view in ("front", "back", "left", "right"):
                path = Path("/repo/source") / f"{group}__cam_{view}"
                result.append({
                    "group_id": group, "camera_view": view,
                    "image_path": path / "0.png", "scene_path": path / "scene_gt.json",
                    "image_relpath": (path / "0.png").relative_to("/repo").as_posix(),
                    "scene": {"object_name": f"object{index}", "relation": relation,
                              "human_visible_center_camera": [0, 0, 10],
                              "object_visible_center_camera": position,
                              "human_frame_camera": {"forward_axis": [0, 0, 1], "right_axis": [1, 0, 0]}},
                })
    return result


class ObjectFoldTests(unittest.TestCase):
    def test_all_folds_cover_each_test_image_once_and_hold_objects_apart(self):
        source = samples()
        tested = Counter()
        validated = Counter()
        for fold in range(9):
            assignments, metadata = object_fold_split(source, 17, fold)
            self.assertEqual((assignments, metadata), object_fold_split(list(reversed(source)), 17, fold))
            split_objects = {split: set(names) for split, names in metadata["objects_by_split"].items()}
            self.assertEqual([len(split_objects[s]) for s in ("train", "validation", "test")], [7, 1, 1])
            self.assertFalse(split_objects["train"] & split_objects["test"])
            self.assertFalse(split_objects["train"] & split_objects["validation"])
            self.assertFalse(split_objects["validation"] & split_objects["test"])
            for sample in source:
                split = assignments[sample["group_id"]]
                if split == "test":
                    tested[sample["image_relpath"]] += 1
                if split == "validation":
                    validated[sample["image_relpath"]] += 1
        self.assertEqual(len(tested), 144)
        self.assertEqual(set(tested.values()), {1})
        self.assertEqual(tested, validated)

    def test_augmentation_and_shared_variant_splits(self):
        with patch("spatial_internalization.data_pipeline.discover_source_samples", return_value=samples()):
            datasets, metadata = build_datasets(Path("/repo"), Path("/repo/source"), object_fold=0)
        for split, expected in (("train", 1008), ("validation", 16), ("test", 16)):
            a, b = [datasets[v][split] for v in ("answer_only", "geometry_reasoning")]
            self.assertEqual(len(a), expected)
            self.assertEqual([r["id"] for r in a], [r["id"] for r in b])
            self.assertEqual({r["object_name"] for r in a}, set(metadata["manifest"]["objects_by_split"][split]))
            self.assertEqual(sum(r["is_noisy"] for r in a), 896 if split == "train" else 0)
            self.assertTrue(all(r["relation"] == r["audit"]["clean_source_geometry"]["computed_relation"] for r in a))

    def test_invalid_fold(self):
        for fold in (-1, 9):
            with self.assertRaises(ValueError):
                object_fold_split(samples(), 17, fold)


if __name__ == "__main__":
    unittest.main()
