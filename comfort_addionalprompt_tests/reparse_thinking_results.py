#!/usr/bin/env python3
"""Re-extract terminal MCQ choices from saved thinking-model responses."""

import argparse
import csv
import io
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from spatial_eval.prompts.MCQ import _normalize_choice_thinking


def reparse(path: Path) -> None:
    original = path.read_bytes()
    text = original.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    if reader.fieldnames is None or "model_answer" not in reader.fieldnames:
        raise ValueError(f"{path} is not a compatible result CSV")
    if reader.fieldnames[-1] != "pred_letter":
        raise ValueError(f"{path} must have pred_letter as its final column")
    rows = list(reader)
    raw_records = _split_csv_records(original)
    if len(raw_records) != len(rows) + 1:
        raise ValueError(
            f"Parsed {len(rows)} data rows but found {len(raw_records) - 1} raw records"
        )

    changed = 0
    incomplete = 0
    output = [raw_records[0]]
    for row, record in zip(rows, raw_records[1:]):
        prediction = _normalize_choice_thinking(row["model_answer"])
        changed += prediction != row["pred_letter"]
        incomplete += prediction == ""
        body, ending = _separate_record_ending(record)
        delimiter = body.rfind(b",")
        if delimiter < 0:
            raise ValueError("CSV data record contains no field delimiter")
        existing = body[delimiter + 1 :].decode("utf-8")
        if existing != row["pred_letter"]:
            raise ValueError("Raw final field disagrees with parsed pred_letter")
        output.append(body[: delimiter + 1] + prediction.encode("ascii") + ending)

    updated = b"".join(output)
    path.write_bytes(updated)

    print(
        f"{path}: rows={len(rows)}, changed={changed}, "
        f"incomplete_without_terminal_answer={incomplete}"
    )


def _split_csv_records(data: bytes) -> list[bytes]:
    """Split CSV records while retaining every original byte and line ending."""
    records = []
    start = 0
    quoted = False
    index = 0
    while index < len(data):
        byte = data[index]
        if byte == ord('"'):
            if quoted and index + 1 < len(data) and data[index + 1] == ord('"'):
                index += 2
                continue
            quoted = not quoted
        elif byte == ord("\n") and not quoted:
            records.append(data[start : index + 1])
            start = index + 1
        index += 1
    if quoted:
        raise ValueError("CSV ends inside a quoted field")
    if start < len(data):
        records.append(data[start:])
    return records


def _separate_record_ending(record: bytes) -> tuple[bytes, bytes]:
    if record.endswith(b"\r\n"):
        return record[:-2], b"\r\n"
    if record.endswith(b"\n"):
        return record[:-1], b"\n"
    return record, b""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.csv:
        reparse(path)


if __name__ == "__main__":
    main()
