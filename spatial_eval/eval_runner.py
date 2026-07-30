# spatial_eval/eval_runner.py
import os
import csv
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Protocol
from .utils import parse_relation_from_basename, parse_relation_for_COMFORT
from .backends.base import VLMBackend

class Asker(Protocol):
    def evaluate_one(self, backend: VLMBackend, img_path: str, second_object: str, correct_relation: str) -> Dict[str, Any]:
        ...

class PairAsker(Protocol):
    def evaluate_one(self, backend: VLMBackend, img_paths: List[str], second_object: str, correct_relation: str) -> Dict[str, Any]:
        ...

SINGLE_HEADER = [
    "image_path", "second_object", "mcq_prompt", "correct_relation",
    "correct_letter", "opposite_relation", "opposite_letter",
    "model_answer", "pred_letter",
]

PAIR_HEADER = [
    "image_path_1", "image_path_2", "cam_view_1", "cam_view_2",
    "second_object", "mcq_prompt", "correct_relation", "correct_letter",
    "opposite_relation", "opposite_letter", "model_answer", "pred_letter",
]


def _completed_rows(output_csv, header, key_columns):
    if not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0:
        return set()
    with open(output_csv, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        if next(reader, None) != header:
            raise ValueError(
                f"Cannot resume {output_csv}: CSV header is not compatible."
            )
        return {tuple(row[:key_columns]) for row in reader if len(row) >= key_columns}


def _record_error(output_csv, sample, exc):
    error_path = os.path.splitext(output_csv)[0] + "_errors.jsonl"
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "sample": sample,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    with open(error_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()

def run_eval(
    backend: VLMBackend,
    image_paths: List[str],
    output_csv: str,
    asker: Asker,
    resume: bool = False,
):
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    completed = _completed_rows(output_csv, SINGLE_HEADER, 1) if resume else set()
    mode = "a" if resume and os.path.exists(output_csv) and os.path.getsize(output_csv) else "w"
    failures = 0

    with open(output_csv, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if mode == "w":
            writer.writerow(SINGLE_HEADER)
            f.flush()

        for source_path in image_paths:
            output_path = f"./{source_path}" if "COMFORT" in source_path else source_path
            if (output_path,) in completed:
                continue
            try:
                if "COMFORT" in source_path:
                    second_object, correct_relation = parse_relation_for_COMFORT(source_path)
                    correct_relation = {
                        "infrontof": "front",
                        "totheleft": "left",
                        "totheright": "right",
                    }.get(correct_relation, correct_relation)
                else:
                    base = os.path.splitext(os.path.basename(source_path))[0]
                    second_object, correct_relation = parse_relation_from_basename(base)

                payload = asker.evaluate_one(
                    backend=backend,
                    img_path=output_path,
                    second_object=second_object,
                    correct_relation=correct_relation,
                )
                writer.writerow([
                    output_path, second_object,
                    payload.get("mcq_prompt", ""),
                    payload.get("correct_relation", ""),
                    payload.get("correct_letter", ""),
                    payload.get("opposite_relation", ""),
                    payload.get("opposite_letter", ""),
                    payload.get("raw_answer", ""),
                    payload.get("pred_letter", ""),
                ])
                f.flush()
            except Exception as exc:
                failures += 1
                _record_error(output_csv, {"image_path": output_path}, exc)
                print(f"Failed: {output_path}: {type(exc).__name__}: {exc}")

    if failures:
        raise RuntimeError(
            f"{failures} sample(s) failed; rerun with --resume. "
            f"Details: {os.path.splitext(output_csv)[0]}_errors.jsonl"
        )


def run_eval_pov(
    backend,
    image_triples: List[Dict[str, Any]],
    output_csv: str,
    asker,
):
    """
    image_triples: list of dicts with keys:
        external_img, correct_pov_img, distractor_pov_img,
        second_object, correct_relation, cam_view_external
    """
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "external_img", "correct_pov_img", "distractor_pov_img",
            "cam_view_external",
            "second_object", "correct_relation",
            "mcq_prompt",
            "img_option_A", "img_option_B",
            "correct_letter", "pred_letter", "correct",
            "raw_answer",
        ])

        for entry in image_triples:
            external_img = f"./{entry['external_img']}"
            correct_pov_img = f"./{entry['correct_pov_img']}"
            distractor_pov_img = f"./{entry['distractor_pov_img']}"

            payload = asker.evaluate_one(
                backend=backend,
                external_img=external_img,
                correct_pov_img=correct_pov_img,
                distractor_pov_img=distractor_pov_img,
            )

            writer.writerow([
                external_img, correct_pov_img, distractor_pov_img,
                entry.get("cam_view_external", ""),
                entry.get("second_object", ""),
                entry.get("correct_relation", ""),
                payload.get("mcq_prompt", ""),
                payload.get("img_option_A", ""),
                payload.get("img_option_B", ""),
                payload.get("correct_letter", ""),
                payload.get("pred_letter", ""),
                payload.get("correct", ""),
                payload.get("raw_answer", ""),
            ])


def run_eval_pov_four(
    backend,
    image_quads: List[Dict[str, Any]],
    output_csv: str,
    asker,
):
    """
    image_quads: list of dicts with keys:
        external_img,
        pov_correct, cam_correct,
        pov_dist1, cam_dist1,
        pov_dist2, cam_dist2,
        correct_relation, cam_view_external,
        second_object, third_object, third_object_relation
    """
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "external_img",
            "pov_correct", "cam_correct",
            "pov_dist1", "cam_dist1",
            "pov_dist2", "cam_dist2",
            "cam_view_external",
            "second_object", "correct_relation",
            "third_object", "third_object_relation",
            "correct_letter", "cam_order",
            "pred_letter", "correct",
            "mcq_prompt", "raw_answer",
        ])

        for entry in image_quads:
            external_img = f"./{entry['external_img']}"
            pov_imgs = {
                entry["cam_correct"]: f"./{entry['pov_correct']}",
                entry["cam_dist1"]:   f"./{entry['pov_dist1']}",
                entry["cam_dist2"]:   f"./{entry['pov_dist2']}",
            }

            payload = asker.evaluate_one(
                backend=backend,
                external_img=external_img,
                pov_imgs=pov_imgs,
                correct_cam=entry["cam_correct"],
            )

            writer.writerow([
                external_img,
                f"./{entry['pov_correct']}", entry["cam_correct"],
                f"./{entry['pov_dist1']}",  entry["cam_dist1"],
                f"./{entry['pov_dist2']}",  entry["cam_dist2"],
                entry.get("cam_view_external", ""),
                entry.get("second_object", ""),
                entry.get("correct_relation", ""),
                entry.get("third_object", ""),
                entry.get("third_object_relation", ""),
                payload.get("correct_letter", ""),
                str(payload.get("cam_order", "")),
                payload.get("pred_letter", ""),
                payload.get("correct", ""),
                payload.get("mcq_prompt", ""),
                payload.get("raw_answer", ""),
            ])


def run_eval_pair(
    backend,
    image_pairs: List[Dict[str, Any]],
    output_csv: str,
    asker,
    resume: bool = False,
):
    """
    image_pairs: list of dicts with keys img1, img2, cam_view_1, cam_view_2.
    Each pair shares the same scene (same object + relation), different camera angles.
    """
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    completed = _completed_rows(output_csv, PAIR_HEADER, 2) if resume else set()
    mode = "a" if resume and os.path.exists(output_csv) and os.path.getsize(output_csv) else "w"
    failures = 0

    with open(output_csv, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if mode == "w":
            writer.writerow(PAIR_HEADER)
            f.flush()

        for entry in image_pairs:
            source_img1 = entry["img1"]
            source_img2 = entry["img2"]
            img1 = f"./{source_img1}"
            img2 = f"./{source_img2}"
            if (img1, img2) in completed:
                continue
            cam_view_1 = entry.get("cam_view_1", "")
            cam_view_2 = entry.get("cam_view_2", "")

            second_object, correct_relation = parse_relation_for_COMFORT(source_img1)
            if correct_relation == "infrontof":
                correct_relation = "front"
            elif correct_relation == "totheleft":
                correct_relation = "left"
            elif correct_relation == "totheright":
                correct_relation = "right"

            try:
                payload = asker.evaluate_one(
                    backend=backend,
                    img_paths=[img1, img2],
                    second_object=second_object,
                    correct_relation=correct_relation,
                )
                writer.writerow([
                    img1, img2, cam_view_1, cam_view_2, second_object,
                    payload.get("mcq_prompt", ""),
                    payload.get("correct_relation", ""),
                    payload.get("correct_letter", ""),
                    payload.get("opposite_relation", ""),
                    payload.get("opposite_letter", ""),
                    payload.get("raw_answer", ""),
                    payload.get("pred_letter", ""),
                ])
                f.flush()
            except Exception as exc:
                failures += 1
                _record_error(
                    output_csv,
                    {"image_path_1": img1, "image_path_2": img2},
                    exc,
                )
                print(f"Failed: {img1}, {img2}: {type(exc).__name__}: {exc}")

    if failures:
        raise RuntimeError(
            f"{failures} pair(s) failed; rerun with --resume. "
            f"Details: {os.path.splitext(output_csv)[0]}_errors.jsonl"
        )
