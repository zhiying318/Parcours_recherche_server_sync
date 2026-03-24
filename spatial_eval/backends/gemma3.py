# spatial_eval/backends/gemma3.py

from dataclasses import dataclass
import os
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

from .base import VLMBackend

@dataclass
class Gemma3Backend(VLMBackend):
    model_id: str
    dtype: str | torch.dtype = torch.bfloat16
    device_map: str = "auto"

    def __post_init__(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            device_map=self.device_map,
        )
        self.model.eval()

    @torch.inference_mode()
    def ask(self, image_path: str, prompt: str, max_new_tokens: int = 512) -> str:
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
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        output = self.model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
        )

        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], output)
        ]

        resp = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return resp.strip()