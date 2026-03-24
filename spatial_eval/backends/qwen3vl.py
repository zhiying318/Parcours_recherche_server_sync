# spatial_eval/backends/qwen3vl.py
from dataclasses import dataclass
import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from .base import VLMBackend

@dataclass
class Qwen3VLBackend(VLMBackend):
    model_id: str
    device_map: str = "auto"
    dtype: str | torch.dtype = "auto"
    attn_implementation: str = "eager"

    def __post_init__(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=self.dtype,
            device_map=self.device_map,
            attn_implementation=self.attn_implementation,
        )
        self.model.eval()

    @torch.inference_mode()
    def ask(self, image_path: str, prompt: str, max_new_tokens: int = 512) -> str:
        img_uri = "file://" + os.path.abspath(image_path)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img_uri},
                {"type": "text", "text": prompt},
            ],
        }]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = process_vision_info(messages, image_patch_size=16)

        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            do_resize=False,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        # print(inputs.keys())
        output = self.model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False)

        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], output)]
        resp = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return resp.strip()