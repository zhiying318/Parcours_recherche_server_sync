import unittest

import numpy as np

from geometry_teacher.coordinates import (
    assert_rotation_matrix,
    camera_to_human_transform,
    classify_horizontal_relation,
    transform_points,
    world_to_opencv_camera,
)


class CoordinateContractTest(unittest.TestCase):

  def test_blender_camera_axes_convert_to_opencv(self) -> None:
    transform = world_to_opencv_camera(np.eye(4))
    points_world = np.array(
        [
            [1.0, 0.0, 0.0],   # Blender camera right
            [0.0, 1.0, 0.0],   # Blender camera up
            [0.0, 0.0, -1.0],  # Blender camera forward
        ]
    )
    expected_opencv = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    np.testing.assert_allclose(transform_points(transform, points_world), expected_opencv)


  def test_camera_world_round_trip(self) -> None:
    camera_to_world_blender = np.array(
        [
            [0.0, -1.0, 0.0, 3.0],
            [1.0, 0.0, 0.0, -2.0],
            [0.0, 0.0, 1.0, 5.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    world_to_camera = world_to_opencv_camera(camera_to_world_blender)
    camera_to_world = np.linalg.inv(world_to_camera)
    points = np.array([[0.2, -1.0, 3.0], [4.0, 2.0, -0.5]])
    reconstructed = transform_points(camera_to_world, transform_points(world_to_camera, points))
    np.testing.assert_allclose(reconstructed, points, atol=1e-12)


  def test_comfort_axes_form_proper_human_rotation_and_labels(self) -> None:
    world_to_camera = world_to_opencv_camera(np.eye(4))
    human_origin_camera = transform_points(world_to_camera, np.zeros(3))

    right_camera = transform_points(world_to_camera, np.array([-1.0, 0.0, 0.0])) - human_origin_camera
    up_camera = transform_points(world_to_camera, np.array([0.0, 0.0, 1.0])) - human_origin_camera
    forward_camera = transform_points(world_to_camera, np.array([0.0, -1.0, 0.0])) - human_origin_camera
    camera_to_human = camera_to_human_transform(
        human_origin_camera, right_camera, up_camera, forward_camera
    )
    assert_rotation_matrix(camera_to_human[:3, :3])

    world_offsets = {
        "left": np.array([2.5, 0.0, 0.0]),
        "right": np.array([-2.5, 0.0, 0.0]),
        "front": np.array([0.0, -2.5, 0.0]),
        "back": np.array([0.0, 2.5, 0.0]),
    }
    for expected, point_world in world_offsets.items():
        point_camera = transform_points(world_to_camera, point_world)
        point_human = transform_points(camera_to_human, point_camera)
        self.assertEqual(classify_horizontal_relation(point_human), expected)


  def test_relation_uses_dominant_human_horizontal_axis_and_tie_is_depth(self) -> None:
    self.assertEqual(classify_horizontal_relation(np.array([3.0, 99.0, -2.0])), "right")
    self.assertEqual(classify_horizontal_relation(np.array([-3.0, -99.0, 2.0])), "left")
    self.assertEqual(classify_horizontal_relation(np.array([1.0, 0.0, -2.0])), "front")
    self.assertEqual(classify_horizontal_relation(np.array([1.0, 0.0, 2.0])), "back")
    self.assertEqual(classify_horizontal_relation(np.array([2.0, 0.0, -2.0])), "front")


  def test_left_handed_axes_are_rejected_as_rotation(self) -> None:
    left_handed = np.stack(
        [
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, -1.0, 0.0]),
        ]
    )
    with self.assertRaisesRegex(ValueError, "determinant"):
      assert_rotation_matrix(left_handed)


if __name__ == "__main__":
  unittest.main()
