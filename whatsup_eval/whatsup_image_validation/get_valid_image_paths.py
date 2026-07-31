#!/usr/bin/env python3
"""Validate WhatsUp edited images and build the canonical valid-image list.

This is the reproducible script form of ``validation_by_mLLM.ipynb``. It
intentionally preserves the selection rule that produced the existing 563
images:

1. Keep a first-round image when at least one of the three object checks is
   answered ``yes``. ``only_one_human`` is recorded but is not a filter.
2. Keep a second-round image only when both human-completeness checks are
   answered ``yes``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VALIDATION_DIR = Path(__file__).resolve().parent
DEFAULT_FOLDERS = (
    "Qwen_Image_Edit_2509_v0",
    "Qwen_Image_Edit_2509_v0_armchair",
    "Qwen_Image_Edit_2509_v0_table",
)
FIRST_ROUND_FIELDS = (
    "only_one_human",
    "object_next_to_human",
    "object_in_image",
    "object_recognisable",
)
OBJECT_FILTER_FIELDS = (
    "object_next_to_human",
    "object_in_image",
    "object_recognisable",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the two-round Qwen3-VL validation for WhatsUp images."
    )
    parser.add_argument("--model_id", default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=PROJECT_ROOT / "Version0_dataset",
    )
    parser.add_argument(
        "--folders",
        nargs="+",
        default=list(DEFAULT_FOLDERS),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=VALIDATION_DIR,
    )
    return parser.parse_args()


def collect_image_paths(dataset_dir: Path, folders: list[str]) -> list[Path]:
    image_paths: list[Path] = []
    for folder in folders:
        folder_path = dataset_dir / folder
        if not folder_path.is_dir():
            raise FileNotFoundError(f"Dataset folder not found: {folder_path}")
        image_paths.extend(
            path
            for path in sorted(folder_path.iterdir())
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )
    return image_paths


def repo_relative(path: Path) -> str:
    try:
        relative = path.resolve().relative_to(PROJECT_ROOT.resolve())
    except ValueError:
        return str(path)
    return f"./{relative.as_posix()}"


class QwenValidator:
    def __init__(self, model_id: str):
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            dtype="auto",
            device_map="auto",
        )
        self.model.eval()

    def ask_yes_no(self, image_path: Path, question: str) -> str:
        prompt = f"{question} Answer with exactly one word: yes or no."
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": f"file://{image_path.resolve()}",
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        images, videos = process_vision_info(messages, image_patch_size=16)
        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            do_resize=False,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
            )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs["input_ids"], output)
        ]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response.strip().lower()


def run_first_round(
    validator: QwenValidator,
    image_paths: list[Path],
    output_csv: Path,
) -> list[dict[str, str]]:
    questions = (
        ("only_one_human", "Is there only 1 human in this image?"),
        ("object_next_to_human", "Is the {second_object} next to the human?"),
        ("object_in_image", "Is the {second_object} in the image?"),
        (
            "object_recognisable",
            "Is the {second_object} clearly recognisable?",
        ),
    )
    rows: list[dict[str, str]] = []
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_path", *FIRST_ROUND_FIELDS],
        )
        writer.writeheader()
        handle.flush()

        for image_path in image_paths:
            second_object = image_path.name.split("_", 1)[0]
            row = {"image_path": repo_relative(image_path)}
            for field, template in questions:
                row[field] = validator.ask_yes_no(
                    image_path,
                    template.format(second_object=second_object),
                )
            writer.writerow(row)
            handle.flush()
            rows.append(row)
    return rows


def passes_first_round(row: dict[str, str]) -> bool:
    # Preserve the notebook rule: reject only if all three object checks fail.
    return any(row[field] == "yes" for field in OBJECT_FILTER_FIELDS)


def run_second_round(
    validator: QwenValidator,
    first_round_rows: list[dict[str, str]],
    path_lookup: dict[str, Path],
    output_csv: Path,
) -> list[str]:
    questions = (
        (
            "human_figure_complete",
            "Is the human figure complete in the image?",
        ),
        (
            "human_body_cut_off",
            "Is there any margin between the top of the head and the upper "
            "border of the image?",
        ),
    )
    valid_paths: list[str] = []
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "image_path",
                "human_figure_complete",
                "human_body_cut_off",
            ],
        )
        writer.writeheader()
        handle.flush()

        for first_row in first_round_rows:
            if not passes_first_round(first_row):
                continue
            stored_path = first_row["image_path"]
            image_path = path_lookup[stored_path]
            row = {"image_path": stored_path}
            for field, question in questions:
                row[field] = validator.ask_yes_no(image_path, question)
            writer.writerow(row)
            handle.flush()

            if all(row[field] == "yes" for field, _ in questions):
                valid_paths.append(stored_path)
    return valid_paths


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = collect_image_paths(args.dataset_dir, args.folders)
    print(f"Found {len(image_paths)} candidate images.")

    validator = QwenValidator(args.model_id)
    first_csv = args.output_dir / "validation_qwen3vl.csv"
    second_csv = args.output_dir / "validation_qwen3vl_2nd.csv"
    valid_json = args.output_dir / "valide_image_paths.json"

    first_rows = run_first_round(validator, image_paths, first_csv)
    path_lookup = {repo_relative(path): path for path in image_paths}
    valid_paths = run_second_round(
        validator,
        first_rows,
        path_lookup,
        second_csv,
    )
    valid_json.write_text(
        json.dumps(valid_paths, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )

    print(f"First-round candidates: {sum(map(passes_first_round, first_rows))}")
    print(f"Valid images: {len(valid_paths)}")
    print(f"Saved: {valid_json}")


if __name__ == "__main__":
    main()
