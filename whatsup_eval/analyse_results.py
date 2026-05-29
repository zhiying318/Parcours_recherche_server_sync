#!/usr/bin/env python3
"""Summarize WhatsUp MCQ result CSV files."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_files", nargs="+")
    return parser.parse_args()


def summarize(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)
    correct = sum(str(row.get("correct", "")).lower() == "true" for row in rows)
    print(f"{path}\t{correct}/{total}\t{(correct / total * 100 if total else 0):.1f}%")

    for field in ["correct_relation", "face_view"]:
        buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for row in rows:
            key = row.get(field, "")
            buckets[key][0] += 1
            buckets[key][1] += str(row.get("correct", "")).lower() == "true"
        for key in sorted(buckets):
            n, ok = buckets[key]
            print(f"  {field}={key}\t{ok}/{n}\t{(ok / n * 100 if n else 0):.1f}%")


def main() -> None:
    args = parse_args()
    for csv_file in args.csv_files:
        summarize(Path(csv_file))


if __name__ == "__main__":
    main()
