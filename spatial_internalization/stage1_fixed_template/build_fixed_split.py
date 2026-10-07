#!/usr/bin/env python3
"""Generate clean/audited A/B capability-internalization SFT data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .dataset_pipeline import build_datasets, write_jsonl

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "COMFORT/data/comfort_human_car_geometry_gt/comfort_human_car"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "spatial_internalization/datasets/stage1_fixed_split"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--augmentations", type=int, default=8)
    parser.add_argument("--position-sigma", type=float, default=0.05)
    parser.add_argument("--angle-sigma-degrees", type=float, default=2.0)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    # Keep repository-relative source paths as COMFORT/... in JSONL.  The
    # directory is a symlink in this checkout, but it still resolves to the
    # original Blender export tree at runtime.
    source_root = args.source_root
    output_root = args.output_root.resolve()
    datasets, metadata = build_datasets(
        REPO_ROOT,
        source_root,
        seed=args.seed,
        augmentations=args.augmentations,
        position_sigma=args.position_sigma,
        angle_sigma_degrees=args.angle_sigma_degrees,
    )
    for variant, splits in datasets.items():
        variant_root = output_root / variant
        for split, records in splits.items():
            write_jsonl(variant_root / f"{split}.jsonl", records)
    manifest = metadata["manifest"]
    manifest["output_root"] = (
        output_root.relative_to(REPO_ROOT).as_posix()
        if output_root.is_relative_to(REPO_ROOT)
        else str(output_root)
    )
    manifest["group_assignments"] = metadata["groups"]
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_root / "audit.json").write_text(json.dumps(metadata["audit"], indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
