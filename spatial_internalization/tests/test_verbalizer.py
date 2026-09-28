import unittest

from spatial_internalization.geometry import QuantizedEvidence
from spatial_internalization.verbalizer import verbalize_evidence_template


class VerbalizerTests(unittest.TestCase):
    def test_camera_origin_sentence_is_exact(self):
        # The generated-data integration test below checks the actual targets;
        # this unit test keeps the requested fixed opening from drifting.
        prefix = (
            "Consider that the picture was taken from the origin [0.000, 0.000, 0.000] "
            "of a camera coordinate system, where +X points to the image's right, +Y "
            "points downward, and +Z points forward into the scene."
        )
        evidence = QuantizedEvidence(
            person_position=(0.0, 0.0, 1.0), object_position=(0.0, 0.0, 2.0),
            person_forward=(0.0, 0.0, 1.0), person_right=(1.0, 0.0, 0.0),
            relative_vector=(0.0, 0.0, 1.0), forward_projection=1.0,
            right_projection=0.0, computed_relation="front", digits=3,
            full_forward_projection=1.0, full_right_projection=0.0,
            normalized_margin=1.0, classification_changed=False,
            degenerate=False, tie=False,
        )
        self.assertTrue(verbalize_evidence_template("ball", evidence).startswith(prefix))
