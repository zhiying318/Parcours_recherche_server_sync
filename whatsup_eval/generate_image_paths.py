#!/usr/bin/env python3
"""Build the image list used by the WhatsUp MCQ evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--valid_json",
        default="../valide_image_paths.json",
        help="JSON file produced by the validation pipeline.",
    )
    parser.add_argument(
        "--out_json",
        default="image_paths.json",
        help="Output JSON list for the WhatsUp evaluator.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.valid_json)
    if not src.exists():
        raise FileNotFoundError(src)

    with src.open("r", encoding="utf-8") as f:
        image_paths = json.load(f)

    if not isinstance(image_paths, list):
        raise ValueError(f"Expected a JSON list in {src}")

    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(image_paths, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(image_paths)} image paths to {out}")


if __name__ == "__main__":
    main()
