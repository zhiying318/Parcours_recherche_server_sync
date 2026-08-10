import unittest

import numpy as np

from geometry_teacher.solver import solve_human_object_geometry


class GeometrySolverTest(unittest.TestCase):
    def setUp(self) -> None:
        height = width = 21
        v, u = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        self.points = np.stack(
            [(u - 10) / 10, (v - 10) / 10, np.ones((height, width))], axis=-1
        )
        self.confidence = np.ones((height, width))
        self.person = np.zeros((height, width), dtype=bool)
        self.person[3:18, 7:14] = True
        self.keypoints = np.zeros((17, 3))
        self.keypoints[[5, 6, 11, 12], 2] = 1.0
        self.keypoints[5, :2] = [8, 6]
        self.keypoints[6, :2] = [12, 6]
        self.keypoints[11, :2] = [9, 14]
        self.keypoints[12, :2] = [11, 14]

    def test_solver_returns_proper_frame(self) -> None:
        target = np.zeros_like(self.person)
        target[8:13, 16:20] = True
        result = solve_human_object_geometry(
            self.points, self.confidence, self.person, target, self.keypoints
        )
        rotation = result["camera_to_human"][:3, :3]
        np.testing.assert_allclose(rotation @ rotation.T, np.eye(3), atol=1e-7)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0)
        self.assertEqual(result["relation"], "right")

    def test_low_keypoint_confidence_is_rejected(self) -> None:
        target = np.zeros_like(self.person)
        target[8:13, 16:20] = True
        self.keypoints[5, 2] = 0.1
        with self.assertRaisesRegex(ValueError, "confidence"):
            solve_human_object_geometry(
                self.points, self.confidence, self.person, target, self.keypoints
            )

    def test_non_boolean_mask_is_rejected(self) -> None:
        target = np.zeros_like(self.person, dtype=np.uint8)
        target[8:13, 16:20] = 1
        with self.assertRaisesRegex(ValueError, "bool dtype"):
            solve_human_object_geometry(
                self.points, self.confidence, self.person, target, self.keypoints
            )


if __name__ == "__main__":
    unittest.main()
