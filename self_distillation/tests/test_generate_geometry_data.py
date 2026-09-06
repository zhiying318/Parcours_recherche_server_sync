import math
import random
import unittest
from collections import Counter

from self_distillation.generate_geometry_data import (
    DEFAULT_PROMPT_INFO,
    DEFAULT_RESULTS_CSV,
    DEFAULT_SOURCE_ROOT,
    OPTION_TEXT,
    classify_geometry,
    generate_records,
    make_shuffled_problem,
    perturb_axes,
    sample_label_preserving_geometry,
)
from self_distillation.vlm_opsd.collator import VLMOPSDCollator


class RecordingProcessor:
    def __init__(self):
        self.tokenizer = type("Tokenizer", (), {})()
        self.calls = []

    def apply_chat_template(self, conversations, **kwargs):
        self.calls.append((conversations, kwargs))
        return {"input_ids": len(self.calls)}


class GeometryDataTest(unittest.TestCase):
    def test_collator_hides_geometry_from_student(self):
        processor = RecordingProcessor()
        collator = VLMOPSDCollator(processor, DEFAULT_SOURCE_ROOT)
        feature = {
            "id": "sample",
            "image_path": "image.png",
            "problem": "Where is the ball?\nA. left\nB. right",
            "teacher_geometry": "person=[1, 2, 3], forward=[0, 0, 1]",
        }
        collator([feature])
        student_text = processor.calls[0][0][0][0]["content"][1]["text"]
        teacher_text = processor.calls[1][0][0][0]["content"][1]["text"]
        self.assertEqual(student_text, feature["problem"])
        self.assertNotIn("person=[", student_text)
        self.assertTrue(teacher_text.startswith(feature["problem"] + "\n\n"))
        self.assertIn("=== Reference Solution Begin ===", teacher_text)
        self.assertIn(feature["teacher_geometry"], teacher_text)
        self.assertIn("Answer:", teacher_text)
        self.assertFalse(processor.calls[0][1]["enable_thinking"])

    def test_perturbed_frame_is_unit_and_orthogonal(self):
        right, forward = perturb_axes(
            [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], 4.0, random.Random(17)
        )
        self.assertAlmostEqual(sum(value * value for value in right), 1.0, places=12)
        self.assertAlmostEqual(sum(value * value for value in forward), 1.0, places=12)
        self.assertAlmostEqual(sum(a * b for a, b in zip(right, forward)), 0.0, places=12)

    def test_classification_uses_dominant_person_axis(self):
        base = {
            "person_position": [0.0, 0.0, 0.0],
            "person_right": [1.0, 0.0, 0.0],
            "person_forward": [0.0, 0.0, 1.0],
        }
        expected = {
            "front": [0.1, 0.0, 2.0], "behind": [0.1, 0.0, -2.0],
            "left": [-2.0, 0.0, 0.1], "right": [2.0, 0.0, 0.1],
        }
        for relation, position in expected.items():
            geometry = {**base, "object_position": position}
            self.assertEqual(classify_geometry(geometry)[0], relation)

    def test_rejection_sampling_preserves_the_clean_label(self):
        geometry = {
            "person_position": [0.0, 0.0, 0.0], "object_position": [0.0, 0.0, 1.0],
            "person_right": [1.0, 0.0, 0.0], "person_forward": [0.0, 0.0, 1.0],
        }
        noisy, attempts = sample_label_preserving_geometry(
            geometry, 2.0, 0.0, random.Random(2)
        )
        self.assertGreater(attempts, 1)
        self.assertEqual(classify_geometry(noisy)[0], "front")

    def test_noisy_option_shuffle_is_balanced_and_label_consistent(self):
        relation = "right"
        shuffled = [
            make_shuffled_problem("ball", relation, augmentation_index, 1000 + augmentation_index)
            for augmentation_index in range(1, 9)
        ]
        self.assertEqual(
            Counter(answer_letter for _, answer_letter in shuffled),
            Counter({"A": 2, "B": 2, "C": 2, "D": 2}),
        )
        for problem, answer_letter in shuffled:
            self.assertIn(
                f"{answer_letter}. From the person's perspective, the ball is "
                f"{OPTION_TEXT[relation]}.", problem,
            )

    def test_only_correct_source_rows_and_train_only_augmentation(self):
        args = (DEFAULT_SOURCE_ROOT, DEFAULT_PROMPT_INFO, DEFAULT_RESULTS_CSV,
                20260827, 2, 0.1, 3.0, 0.70, 0.15)
        first, first_splits = generate_records(*args)
        second, second_splits = generate_records(*args)
        self.assertEqual(first, second)
        self.assertEqual(first_splits, second_splits)
        self.assertTrue(first_splits["train"].isdisjoint(first_splits["val"]))
        self.assertTrue(first_splits["train"].isdisjoint(first_splits["test"]))
        self.assertTrue(first_splits["val"].isdisjoint(first_splits["test"]))
        self.assertEqual(len(first["train"]), 74 * 3)
        self.assertEqual(len(first["val"]), 15)
        self.assertEqual(len(first["test"]), 24)
        self.assertTrue(any(record["is_noisy"] for record in first["train"]))
        self.assertFalse(any(record["is_noisy"] for record in first["val"]))
        self.assertFalse(any(record["is_noisy"] for record in first["test"]))
        for record in first["train"] + first["val"] + first["test"]:
            clean_relation = classify_geometry(record["clean_geometry"])[0]
            self.assertEqual(record["relation"], clean_relation)
            expected_minimum_attempts = 1 if record["is_noisy"] else 0
            self.assertGreaterEqual(record["noise_attempts"], expected_minimum_attempts)
            forward = record["teacher_geometry_values"]["person_forward"]
            forward_norm = math.sqrt(sum(v * v for v in forward))
            if record["is_noisy"]:
                self.assertAlmostEqual(forward_norm, 1.0, delta=0.001)
            else:
                self.assertAlmostEqual(forward_norm, 1.0, places=12)
            if record["is_noisy"]:
                for vector in record["teacher_geometry_values"].values():
                    self.assertTrue(all(value == round(value, 3) for value in vector))
            self.assertNotIn("solution", record)
            self.assertEqual(
                record["reference_source"],
                "noisy_geometry" if record["is_noisy"] else "test07_clean_geometry",
            )
            self.assertTrue(record["problem"].startswith("Where is the "))
            self.assertNotIn("camera coordinate system", record["problem"])
            self.assertIn("camera coordinate system", record["teacher_geometry"])
            self.assertIn("image", record["messages"][0]["content"][0])

if __name__ == "__main__":
    unittest.main()
