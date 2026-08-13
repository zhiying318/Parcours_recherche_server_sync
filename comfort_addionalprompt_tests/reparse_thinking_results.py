#!/usr/bin/env python3
"""Re-extract terminal MCQ choices from saved thinking-model responses."""

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from spatial_eval.prompts.MCQ import _normalize_choice_thinking


def reparse(path: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "model_answer" not in reader.fieldnames:
            raise ValueError(f"{path} is not a compatible result CSV")
        if "pred_letter" not in reader.fieldnames:
            raise ValueError(f"{path} has no pred_letter column")
        rows = list(reader)
        fieldnames = reader.fieldnames

    changed = 0
    incomplete = 0
    for row in rows:
        prediction = _normalize_choice_thinking(row["model_answer"])
        changed += prediction != row["pred_letter"]
        incomplete += prediction == ""
        row["pred_letter"] = prediction

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    print(
        f"{path}: rows={len(rows)}, changed={changed}, "
        f"incomplete_without_terminal_answer={incomplete}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.csv:
        reparse(path)


if __name__ == "__main__":
    main()
