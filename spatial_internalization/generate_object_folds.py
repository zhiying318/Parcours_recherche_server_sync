#!/usr/bin/env python3
"""Generate leave-one-object-out folds using the existing coordinate augmentation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .data_pipeline import build_datasets, discover_source_samples, object_fold_split, write_jsonl
from .generate_data import REPO_ROOT, DEFAULT_SOURCE_ROOT


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=REPO_ROOT / "spatial_internalization/data_object_cv")
    parser.add_argument("--seed", type=int, default=20260919)
    parser.add_argument("--fold", default="all", help="all (default), or a zero-based fold number")
    parser.add_argument("--augmentations", type=int, default=8)
    parser.add_argument("--position-sigma", type=float, default=0.05)
    parser.add_argument("--angle-sigma-degrees", type=float, default=2.0)
    args = parser.parse_args(argv)
    if args.augmentations < 0 or args.position_sigma < 0 or args.angle_sigma_degrees < 0:
        parser.error("augmentation count and noise scales must be non-negative")
    # Preserve the lexical COMFORT path if it is a symlink.
    source_root = args.source_root.absolute()
    samples = discover_source_samples(REPO_ROOT, source_root)
    _, plan = object_fold_split(samples, args.seed, 0)
    try:
        folds = list(range(plan["num_folds"])) if args.fold == "all" else [int(args.fold)]
        for fold in folds:
            object_fold_split(samples, args.seed, fold)
    except ValueError as exc:
        parser.error(str(exc))
    output_root = args.output_root.resolve()
    # Refuse accidental replacement of already generated folds.
    for fold in folds:
        if (output_root / f"fold_{fold}").exists():
            parser.error(f"{output_root / f'fold_{fold}'} already exists; choose a new --output-root")
    for fold in folds:
        datasets, metadata = build_datasets(
            REPO_ROOT, source_root, seed=args.seed, augmentations=args.augmentations,
            position_sigma=args.position_sigma, angle_sigma_degrees=args.angle_sigma_degrees,
            object_fold=fold,
        )
        fold_root = output_root / f"fold_{fold}"
        for variant, splits in datasets.items():
            for split, records in splits.items():
                write_jsonl(fold_root / variant / f"{split}.jsonl", records)
        manifest = metadata["manifest"]
        manifest["output_root"] = str(fold_root)
        manifest["group_assignments"] = metadata["groups"]
        for name, value in (("manifest", manifest), ("audit", metadata["audit"])):
            (fold_root / f"{name}.json").write_text(
                json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
            )
        print(json.dumps({"fold": fold, "data_root": str(fold_root),
                          "objects": manifest["objects_by_split"],
                          "records": manifest["record_counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
