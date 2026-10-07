import unittest

from spatial_internalization.core.geometry import GeometryError, extract_canonical_evidence, quantize_and_verify_evidence
from spatial_internalization.stage1_fixed_template.dataset_pipeline import scene_group_id, stratified_group_split


def scene(person, obj, forward, right, relation="front"):
    return {
        "object_name": "basketball",
        "relation": relation,
        "human_visible_center_camera": person,
        "object_visible_center_camera": obj,
        "human_frame_camera": {"forward_axis": forward, "right_axis": right},
    }


class GeometryTests(unittest.TestCase):
    def test_four_directions_and_forward_tie_priority(self):
        cases = [
            ([0, 0, 1], [0, 0, 2], [0, 0, 1], [1, 0, 0], "front"),
            ([0, 0, 1], [0, 0, 0], [0, 0, 1], [1, 0, 0], "behind"),
            ([0, 0, 1], [-2, 0, 1], [0, 0, 1], [1, 0, 0], "left"),
            ([0, 0, 1], [2, 0, 1], [0, 0, 1], [1, 0, 0], "right"),
            ([0, 0, 1], [1, 0, 2], [0, 0, 1], [1, 0, 0], "front"),
        ]
        for person, obj, forward, right, expected in cases:
            evidence = extract_canonical_evidence(scene(person, obj, forward, right, {"behind": "back"}.get(expected, expected)))
            self.assertEqual(evidence.computed_relation, expected)
            self.assertTrue(evidence.source_relation_match)

    def test_common_positive_scale_does_not_change_relation(self):
        base = scene([0.4, -0.2, 2.0], [1.2, -0.2, 2.1], [0, 0, 1], [1, 0, 0], "right")
        scaled = scene([2.0, -1.0, 10.0], [6.0, -1.0, 10.5], [0, 0, 1], [1, 0, 0], "right")
        self.assertEqual(extract_canonical_evidence(base).computed_relation, extract_canonical_evidence(scaled).computed_relation)

    def test_right_axis_sign(self):
        e = extract_canonical_evidence(scene([0, 0, 1], [2, 0, 1], [0, 0, 1], [1, 0, 0], "right"))
        self.assertGreater(e.right_projection, 0)
        e = extract_canonical_evidence(scene([0, 0, 1], [-2, 0, 1], [0, 0, 1], [1, 0, 0], "left"))
        self.assertLess(e.right_projection, 0)

    def test_rounding_flip_retries_higher_precision(self):
        e = extract_canonical_evidence(scene(
            [0.5230099230252523, -0.24960406449905848, 1.0],
            [0.516360321773858, -0.2527935807902563, 1.0069922005313423],
            [1, 0, 0], [0, 0, 1], "right"))
        q = quantize_and_verify_evidence(e)
        self.assertEqual(q.computed_relation, "right")
        self.assertEqual(q.digits, 4)

    def test_bad_gt_and_degenerate_input_are_not_silently_fixed(self):
        mismatch = extract_canonical_evidence(scene([0, 0, 1], [0, 0, 2], [0, 0, 1], [1, 0, 0], "back"))
        self.assertFalse(mismatch.source_relation_match)
        with self.assertRaises(GeometryError):
            extract_canonical_evidence(scene([0, 0, 1], [0, 0, 2], [0, 0, 1], [0, 0, 0], "front"))
        with self.assertRaises(GeometryError):
            extract_canonical_evidence(scene([0, 0, 0], [0, 0, 2], [0, 0, 1], [1, 0, 0], "front"))

    def test_scene_group_split_has_no_group_leakage(self):
        samples = []
        for relation in ("front", "back", "left", "right"):
            for index in range(9):
                group = f"{relation}/object__{relation}__{index}"
                for view in range(4):
                    samples.append({"group_id": group, "scene": {"relation": relation}, "image_relpath": f"{group}/{view}.png"})
        split = stratified_group_split(samples, seed=7)
        self.assertEqual(len(split), 36)
        self.assertEqual(set(split.values()), {"train", "validation", "test"})
        self.assertEqual(sum(value == "validation" for value in split.values()), 4)
        self.assertEqual(sum(value == "test" for value in split.values()), 4)
