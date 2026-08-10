import unittest

import numpy as np

from geometry_teacher.reasoning import generate_reasoning_trace, validate_reasoning_trace


class ReasoningTraceTest(unittest.TestCase):
    def setUp(self):
        self.primitive = {
            "human_coordinate_frame": {
                "origin": [1.0, 2.0, 3.0],
                "right_axis": [1.0, 0.0, 0.0],
                "up_axis": [0.0, 1.0, 0.0],
                "forward_axis": [0.0, 0.0, -1.0],
                "back_axis": [0.0, 0.0, 1.0],
            },
            "camera_to_human_transform": {
                "rotation": np.eye(3).tolist(),
                "translation": [-1.0, -2.0, -3.0],
            },
            "object_relative_position": {
                "object": "car",
                "camera_position": [3.0, 2.0, 2.5],
                "human_position": [2.0, 0.0, -0.5],
                "relation": "right",
            },
        }

    def test_generated_trace_passes_all_checks(self):
        trace = generate_reasoning_trace(self.primitive)
        result = validate_reasoning_trace(self.primitive, trace, expected_answer="right")
        self.assertTrue(result["valid"])
        self.assertTrue(all(result["checks"].values()))

    def test_wrong_ground_truth_fails_answer_check(self):
        trace = generate_reasoning_trace(self.primitive)
        result = validate_reasoning_trace(self.primitive, trace, expected_answer="left")
        self.assertFalse(result["valid"])
        self.assertFalse(result["checks"]["answer_matches_ground_truth"])

    def test_inconsistent_transform_fails(self):
        trace = generate_reasoning_trace(self.primitive)
        self.primitive["camera_to_human_transform"]["translation"][0] = 0.0
        result = validate_reasoning_trace(self.primitive, trace)
        self.assertFalse(result["valid"])
        self.assertFalse(result["checks"]["transform_correct"])


if __name__ == "__main__":
    unittest.main()
