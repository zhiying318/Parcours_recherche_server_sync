#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import csv
import argparse
from dataclasses import dataclass
from typing import Tuple, List, Optional

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoModelForVision2Seq,
    AutoModelForImageTextToText,
)
from qwen_vl_utils import process_vision_info

# ----------------------------
# Utils
# ----------------------------

def get_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float16

def yn(x: str) -> str:
    x = (x or "").strip().lower()
    x = re.split(r"\s+", x)[0].strip(".,;:!()[]{}<>\"'`")
    if x in ("yes", "y", "true", "1"):
        return "yes"
    if x in ("no", "n", "false", "0"):
        return "no"
    return x  # fallback

def parse_relation_from_basename(base: str) -> Tuple[str, str]:
    """
    base example (no extension):
      book_right_of_chair_FACE-CAMERA
      beer-bottle_left_of_chair_FACE-LEFT
    returns: (second_object, correct_relation)
    """
    core = base.split("_FACE-", 1)[0]
    second_object = core.split("_")[0]
    direction1 = core.split("_")[1]
    direction2 = base.split("_FACE-", 1)[1]

    correct_relation = None

    if direction2 == "CAMERA":
        if direction1 == "right":
            correct_relation = "left"
        elif direction1 == "left":
            correct_relation = "right"
    elif direction2 == "LEFT":
        if direction1 == "right":
            correct_relation = "behind"
        elif direction1 == "left":
            correct_relation = "front"
    elif direction2 == "RIGHT":
        if direction1 == "right":
            correct_relation = "front"
        elif direction1 == "left":
            correct_relation = "behind"

    if correct_relation is None:
        raise ValueError(f"Unrecognized pattern for base='{base}'")

    return second_object, correct_relation

# ----------------------------
# Unified model interface
# ----------------------------

class VLMBackend:
    def ask(self, image_path: str, prompt: str, max_new_tokens: int) -> str:
        raise NotImplementedError

@dataclass
class QwenBackend(VLMBackend):
    model_id: str
    device_map: str = "auto"
    dtype: torch.dtype = torch.float16

    def __post_init__(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            device_map=self.device_map,
        )
        self.model.eval()

    @torch.inference_mode()
    def ask(self, image_path: str, prompt: str, max_new_tokens: int = 64) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=int(max_new_tokens))
        gen = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(gen, skip_special_tokens=True).strip()

@dataclass
class InternVLBackend(VLMBackend):
    model_id: str
    device_map: str = "auto"
    dtype: torch.dtype = torch.float16

    def __post_init__(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            device_map=self.device_map,
        )
        self.model.eval()

    @torch.inference_mode()
    def ask(self, image_path: str, prompt: str, max_new_tokens: int = 64) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]

        inputs_raw = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

        # move to device safely
        inputs = {}
        for k, v in inputs_raw.items():
            inputs[k] = v.to(self.model.device) if torch.is_tensor(v) else v

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )

        input_len = inputs["input_ids"].shape[-1]
        return self.processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

@dataclass
class Qwen3Backend(VLMBackend):
    model_id: str
    device_map: str = "auto"
    dtype: str | torch.dtype = "auto"   # 按你成功代码写 "auto"

    def __post_init__(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=self.dtype,
            device_map=self.device_map,
            attn_implementation="eager",
        )
        self.model.eval()

    @torch.inference_mode()
    def ask(self, image_path: str, prompt: str, max_new_tokens: int = 64) -> str:
        # 这里保持和你成功代码一致：用 file:// 形式
        img_uri = "file://" + os.path.abspath(image_path)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_uri},
                {"type": "text", "text": prompt},
            ],
        }]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        print(inputs.keys())
        images, videos = process_vision_info(messages, image_patch_size=16)

        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            do_resize=False,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
        )

        # 去掉输入部分，只保留新生成
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], output)
        ]
        resp = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return resp.strip()

@dataclass
class Gemma3Backend(VLMBackend):
    model_id: str
    device_map: str = "auto"
    dtype: str | torch.dtype = "auto"

    def __post_init__(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map=self.device_map,
        )
        self.model.eval()

    @torch.inference_mode()
    def ask(self, image_path: str, prompt: str, max_new_tokens: int = 64) -> str:
        # img_uri = "file://" + os.path.abspath(image_path)
        image = Image.open(image_path).convert("RGB")
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
        )

        generated_ids_trimmed = output[:, inputs["input_ids"].shape[-1]:]

        resp = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return resp.strip()

def build_backend(name: str, model_id: str, dtype: torch.dtype, device_map: str) -> VLMBackend:
    name = name.lower()
    if name in ("qwen2", "qwen2.5", "qwen2.5vl"):
        return QwenBackend(model_id=model_id, dtype=dtype, device_map=device_map)
    if name in ("qwen3vl", "qwen3"):
        return Qwen3Backend(model_id=model_id, dtype=dtype, device_map=device_map)
    if name in ("internvl", "internvl3.5", "intern"):
        return InternVLBackend(model_id=model_id, dtype=dtype, device_map=device_map)
    if name in ("gemma3", "gemma", "gemma-vl"):
        return Gemma3Backend(model_id=model_id, dtype=dtype, device_map=device_map)

    raise ValueError(f"Unknown backend '{name}'. Use qwen or internvl.")

# ----------------------------
# Eval loop
# ----------------------------

def run_eval(
    backend: VLMBackend,
    image_paths: List[str],
    output_csv: str,
    max_new_tokens_where: int = 64,
    max_new_tokens_yn: int = 8,
):
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_path",
            "second_object",
            "where_answer",
            "front_answer",
            "back_answer",
            "left_answer",
            "right_answer",
            "correct_answer",
        ])

        for img_path in image_paths:
            base = os.path.splitext(os.path.basename(img_path))[0]
            second_object, correct_relation = parse_relation_from_basename(base)

            where_prompt = f"Where is the {second_object} in the view of the human in the image?"
            front_prompt = f"Is the {second_object} in the front of the human? Answer only yes or no."
            back_prompt  = f"Is the {second_object} in the back of the human? Answer only yes or no."
            left_prompt  = f"Is the {second_object} on the left of the human? Answer only yes or no."
            right_prompt = f"Is the {second_object} on the right of the human? Answer only yes or no."

            where_ans = backend.ask(img_path, where_prompt, max_new_tokens_where)
            front_ans = yn(backend.ask(img_path, front_prompt, max_new_tokens_yn))
            back_ans  = yn(backend.ask(img_path, back_prompt,  max_new_tokens_yn))
            left_ans  = yn(backend.ask(img_path, left_prompt,  max_new_tokens_yn))
            right_ans = yn(backend.ask(img_path, right_prompt, max_new_tokens_yn))

            writer.writerow([
                img_path,
                second_object,
                where_ans,
                front_ans,
                back_ans,
                left_ans,
                right_ans,
                correct_relation,
            ])

# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Spatial QA evaluation for VLMs (Qwen / InternVL / Qwen3VL).")
    parser.add_argument("--backend", required=True, choices=["qwen", "internvl", "qwen3vl", "gemma3"], help="Which adapter to use.")
    parser.add_argument("--model_id", required=True, help="HuggingFace model id.")
    parser.add_argument("--image_json", required=True, help="JSON file containing a list of image paths.")
    parser.add_argument("--out_csv", required=True, help="Output CSV path.")
    parser.add_argument("--cuda_visible_devices", default=None, help="Set CUDA_VISIBLE_DEVICES before loading.")
    parser.add_argument("--device_map", default="auto", help="HF device_map, default=auto.")
    parser.add_argument("--max_new_tokens_where", type=int, default=64)
    parser.add_argument("--max_new_tokens_yn", type=int, default=8)

    args = parser.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)

    dtype = get_dtype()

    with open(args.image_json, "r") as f:
        image_paths = json.load(f)

    backend = build_backend(args.backend, args.model_id, dtype=dtype, device_map=args.device_map)

    run_eval(
        backend=backend,
        image_paths=image_paths,
        output_csv=args.out_csv,
        max_new_tokens_where=args.max_new_tokens_where,
        max_new_tokens_yn=args.max_new_tokens_yn,
    )

    print(f"Saved: {args.out_csv}")

if __name__ == "__main__":
    main()