"""Small inspection helper for the deterministic scene-group split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path, default=Path(__file__).parent / "data/manifest.json", nargs="?")
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    print(json.dumps({"group_counts": manifest["group_counts"], "group_assignments": manifest["group_assignments"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

