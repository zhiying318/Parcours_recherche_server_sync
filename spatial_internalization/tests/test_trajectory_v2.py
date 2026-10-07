import json
from pathlib import Path
import unittest

from spatial_internalization.core.offline_qwen import model_metadata, resolve_local_model_path
from spatial_internalization.stage2_trajectory_guided.trajectory_pipeline import (
    DEFAULT_FIXED_SPLIT_MANIFEST,
    DEFAULT_SOURCE_CSV,
    DEFAULT_SOURCE_ROOT,
    adapt_evidence,
    apply_rewritten_claims,
    canonical_geometry_evidence,
    evidence_dict,
    first_version_noise_seeds,
    geometry_for_augmentation,
    load_fixed_split_assignments,
    load_qualified_trajectories,
)


class TrajectoryV2Tests(unittest.TestCase):
    def test_source_filter_and_fixed_split_are_deterministic(self):
        qualified, rejected, _ = load_qualified_trajectories(DEFAULT_SOURCE_CSV, DEFAULT_SOURCE_ROOT)
        self.assertEqual(len(qualified), 113)
        self.assertEqual(len(rejected), 31)
        assignments = load_fixed_split_assignments(DEFAULT_FIXED_SPLIT_MANIFEST)
        split_counts = {name: 0 for name in ("train", "validation", "test")}
        for item in qualified:
            split_counts[assignments[item["scene_group_id"]]] += 1
        self.assertEqual(split_counts, {"train": 87, "validation": 14, "test": 12})

    def test_noise_reuses_first_version_values(self):
        qualified, _, _ = load_qualified_trajectories(DEFAULT_SOURCE_CSV, DEFAULT_SOURCE_ROOT)
        assignments = load_fixed_split_assignments(DEFAULT_FIXED_SPLIT_MANIFEST)
        seeds = first_version_noise_seeds(DEFAULT_SOURCE_ROOT, 20260919, assignments, 0.05, 2.0)
        item = next(item for item in qualified if item["image_path"].endswith("behind/basketball__behind__cam_back/0.png"))
        display, attempts = geometry_for_augmentation(item, 1, seeds[(item["image_path"], 1)], 0.05, 2.0)
        self.assertEqual(display.object_position, (0.01, 0.025, 0.775))
        self.assertEqual(display.forward_projection, -0.255)
        self.assertEqual(attempts, 1)

    def test_adaptation_rewrites_claims_that_depend_on_changed_geometry(self):
        qualified, _, _ = load_qualified_trajectories(DEFAULT_SOURCE_CSV, DEFAULT_SOURCE_ROOT)
        item = qualified[0]
        clean_units = canonical_geometry_evidence(
            item["scene"]["object_name"], item["clean_display"], "clean_blender_gt"
        )
        clean_units.append({
            "id": "COT_001",
            "text": "The old displacement places the object on the right.",
            "source": "trajectory_cot",
            "depends_on": ["G_DISPLACEMENT", "G_RIGHT_PROJECTION"],
            "fields": ["spatial_claim"],
        })
        clean_units.append({
            "id": "COT_002",
            "text": "That projection determines the final spatial relation.",
            "source": "trajectory_cot",
            "depends_on": ["COT_001"],
            "fields": ["spatial_conclusion"],
        })
        consolidated = {
            "units": clean_units,
            "clean_gt_display": evidence_dict(item["clean_display"]),
            "clean_gt_raw": evidence_dict(item["clean_raw"]),
            "consolidator_dropped_claims": [],
        }
        assignments = load_fixed_split_assignments(DEFAULT_FIXED_SPLIT_MANIFEST)
        seeds = first_version_noise_seeds(DEFAULT_SOURCE_ROOT, 20260919, assignments, 0.05, 2.0)
        display, _ = geometry_for_augmentation(item, 1, seeds[(item["image_path"], 1)], 0.05, 2.0)
        adapted = adapt_evidence(consolidated, display, item["scene"]["object_name"], 1)
        self.assertIn("COT_001", {unit["id"] for unit in adapted["units"]})
        self.assertEqual(adapted["claims_to_rewrite"], ["COT_001", "COT_002"])
        adapted = apply_rewritten_claims(adapted, {
            "rewritten_claims": [
                {
                    "id": "COT_001",
                    "text": "The updated projections place the object behind the person.",
                },
                {
                    "id": "COT_002",
                    "text": "That projection determines the updated spatial relation.",
                },
            ]
        })
        rewritten = next(unit for unit in adapted["units"] if unit["id"] == "COT_001")
        self.assertEqual(rewritten["source"], "trajectory_cot_geometry_rewritten")
        self.assertEqual(adapted["rewritten_claim_ids"], ["COT_001", "COT_002"])
        self.assertIn("G_DISPLACEMENT", adapted["changed_evidence_ids"])
        self.assertIn("G_FORWARD_PROJECTION", adapted["recomputed_evidence_ids"])

    def test_local_model_resolution_records_snapshot_and_context(self):
        path = resolve_local_model_path()
        metadata = model_metadata(path)
        self.assertEqual(metadata["model_type"], "qwen3_5")
        self.assertEqual(metadata["context_length"], 262144)
        self.assertTrue((path / "model.safetensors.index.json").is_file())


if __name__ == "__main__":
    unittest.main()
