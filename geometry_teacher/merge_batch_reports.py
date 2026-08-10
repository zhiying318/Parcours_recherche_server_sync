"""Merge Geometry Teacher shard reports into one compact batch report."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    args = parser.parse_args()

    reports = [
        json.loads((args.output_root / f"batch_report_shard_{index}.json").read_text())
        for index in range(args.num_shards)
    ]
    grouped = {}
    for field in ("by_relation", "by_object"):
        values = defaultdict(Counter)
        for report in reports:
            for name, counts in report[field].items():
                values[name].update(counts)
        grouped[field] = {name: dict(counts) for name, counts in sorted(values.items())}

    unique_errors = {}
    for report in reports:
        for error in report["errors"]:
            sample_id = error["sample_id"]
            existing = unique_errors.get(sample_id)
            if existing is None or (existing["stage"] == "finalize" and error["stage"] != "finalize"):
                unique_errors[sample_id] = {
                    key: error[key]
                    for key in ("sample_id", "stage", "error_type", "message")
                }
    categories = Counter(
        (error["stage"], error["error_type"], error["message"])
        for error in unique_errors.values()
    )
    total = sum(report["total_samples"] for report in reports)
    valid = sum(report["valid_samples"] for report in reports)
    result = {
        "total_samples": total,
        "valid_samples": valid,
        "failed_samples": total - valid,
        "valid_ratio": valid / total if total else 0.0,
        "relation_accuracy": valid / total if total else 0.0,
        **grouped,
        "failure_categories": [
            {"stage": key[0], "error_type": key[1], "message": key[2], "count": count}
            for key, count in categories.most_common()
        ],
        "failed_sample_records": [unique_errors[key] for key in sorted(unique_errors)],
        "shard_runtime": [report.get("runtime") for report in reports],
        "shard_reports": [f"batch_report_shard_{index}.json" for index in range(args.num_shards)],
    }
    destination = args.output_root / "batch_report.json"
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
