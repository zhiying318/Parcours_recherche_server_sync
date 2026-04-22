# spatial_eval/eval_runner.py
import os
import csv
from typing import List, Dict, Any, Protocol
from .utils import parse_relation_from_basename, parse_relation_for_COMFORT
from .backends.base import VLMBackend

class Asker(Protocol):
    def evaluate_one(self, backend: VLMBackend, img_path: str, second_object: str, correct_relation: str) -> Dict[str, Any]:
        ...

class PairAsker(Protocol):
    def evaluate_one(self, backend: VLMBackend, img_paths: List[str], second_object: str, correct_relation: str) -> Dict[str, Any]:
        ...

def run_eval(
    backend: VLMBackend,
    image_paths: List[str],
    output_csv: str,
    asker: Asker,
):
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # writer.writerow([
        #     "image_path",
        #     "second_object",
        #     "correct_relation",
        #     "payload_json",
        # ])

        # for img_path in image_paths:
        #     base = os.path.splitext(os.path.basename(img_path))[0]
        #     second_object, correct_relation = parse_relation_from_basename(base)

        #     payload = asker.evaluate_one(
        #         backend=backend,
        #         img_path=img_path,
        #         second_object=second_object,
        #         correct_relation=correct_relation,
        #     )

        #     writer.writerow([
        #         img_path,
        #         second_object,
        #         correct_relation,
        #         __import__("json").dumps(payload, ensure_ascii=False),
        #     ])
        writer.writerow([
            "image_path",
            "second_object",
            "mcq_prompt",
            "correct_relation",
            "correct_letter",
            "opposite_relation",
            "opposite_letter",
            "model_answer",  
            "pred_letter",
        ])

        for img_path in image_paths:
            if 'COMFORT' not in img_path:
                base = os.path.splitext(os.path.basename(img_path))[0]
                second_object, correct_relation = parse_relation_from_basename(base)

                payload = asker.evaluate_one(
                    backend=backend,
                    img_path=img_path,
                    second_object=second_object,
                    correct_relation=correct_relation,
                )

                writer.writerow([
                    img_path,
                    second_object,
                    payload.get("mcq_prompt", ""),
                    payload.get("correct_relation", ""),
                    payload.get("correct_letter", ""),
                    payload.get("opposite_relation", ""),
                    payload.get("opposite_letter", ""),
                    payload.get("raw_answer", ""),  
                    payload.get("pred_letter", ""),
                ])
            else:
                # For generated dataset using COMFORT framework, they are named as 0.png, 1.png, ... without relation info in filename.
                second_object, correct_relation = parse_relation_for_COMFORT(img_path)

                if correct_relation == "infrontof":
                    correct_relation = "front"
                elif correct_relation == "totheleft":
                    correct_relation = "left"
                elif correct_relation == "totheright":
                    correct_relation = "right"

                img_path = f"./{img_path}"  # change relative path to absolute path, in case COMFORT images are stored outside current working dir

                payload = asker.evaluate_one(
                    backend=backend,
                    img_path=img_path,
                    second_object=second_object,
                    correct_relation=correct_relation,
                )

                writer.writerow([
                    img_path,
                    second_object,
                    payload.get("mcq_prompt", ""),
                    payload.get("correct_relation", ""),
                    payload.get("correct_letter", ""),
                    payload.get("opposite_relation", ""),
                    payload.get("opposite_letter", ""),
                    payload.get("raw_answer", ""),
                    payload.get("pred_letter", ""),
                ])


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
):
    """
    image_pairs: list of dicts with keys img1, img2, cam_view_1, cam_view_2.
    Each pair shares the same scene (same object + relation), different camera angles.
    """
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path_1", "image_path_2",
            "cam_view_1", "cam_view_2",
            "second_object",
            "mcq_prompt",
            "correct_relation", "correct_letter",
            "opposite_relation", "opposite_letter",
            "model_answer", "pred_letter",
        ])

        for entry in image_pairs:
            img1 = entry["img1"]
            img2 = entry["img2"]
            cam_view_1 = entry.get("cam_view_1", "")
            cam_view_2 = entry.get("cam_view_2", "")

            # Both images come from the same COMFORT scene; parse from img1
            second_object, correct_relation = parse_relation_for_COMFORT(img1)
            if correct_relation == "infrontof":
                correct_relation = "front"
            elif correct_relation == "totheleft":
                correct_relation = "left"
            elif correct_relation == "totheright":
                correct_relation = "right"

            img1 = f"./{img1}"
            img2 = f"./{img2}"

            payload = asker.evaluate_one(
                backend=backend,
                img_paths=[img1, img2],
                second_object=second_object,
                correct_relation=correct_relation,
            )

            writer.writerow([
                img1, img2,
                cam_view_1, cam_view_2,
                second_object,
                payload.get("mcq_prompt", ""),
                payload.get("correct_relation", ""),
                payload.get("correct_letter", ""),
                payload.get("opposite_relation", ""),
                payload.get("opposite_letter", ""),
                payload.get("raw_answer", ""),
                payload.get("pred_letter", ""),
            ])