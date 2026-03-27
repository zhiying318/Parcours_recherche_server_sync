# spatial_eval/eval_runner.py
import os
import csv
from typing import List, Dict, Any, Protocol
from .utils import parse_relation_from_basename, parse_relation_for_COMFORT
from .backends.base import VLMBackend

class Asker(Protocol):
    def evaluate_one(self, backend: VLMBackend, img_path: str, second_object: str, correct_relation: str) -> Dict[str, Any]:
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