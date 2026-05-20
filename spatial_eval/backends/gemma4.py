# spatial_eval/backends/gemma4.py

from dataclasses import dataclass
import gc

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .base import VLMBackend


@dataclass
class Gemma4Backend(VLMBackend):
    model_id: str
    dtype: str | torch.dtype = torch.bfloat16
    device_map: str = "auto"
    attn_implementation: str = "sdpa"
    enable_thinking: bool = False

    def __post_init__(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            padding_side="left",
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=self.dtype,
            device_map=self.device_map,
            attn_implementation=self.attn_implementation,
        )
        self.model.eval()

    def _run_messages(self, messages, max_new_tokens):
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=self.enable_thinking,
        ).to(self.model.device)

        generate_kwargs = {
            "max_new_tokens": int(max_new_tokens),
        }
        if self.enable_thinking:
            generate_kwargs.update({
                "do_sample": True,
                "temperature": 0.6,
            })
        else:
            generate_kwargs["do_sample"] = False

        output = self.model.generate(
            **inputs,
            **generate_kwargs,
        )
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], output)
        ]
        response = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        del inputs, output, trimmed
        gc.collect()
        torch.cuda.empty_cache()

        return response

    @torch.inference_mode()
    def ask(self, image_path: str, prompt: str, max_new_tokens: int = 512) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": Image.open(image_path).convert("RGB")},
            {"type": "text", "text": prompt},
        ]}]
        return self._run_messages(messages, max_new_tokens)

    @torch.inference_mode()
    def ask_multi(self, image_paths, prompt: str, max_new_tokens: int = 512) -> str:
        content = [
            {"type": "image", "image": Image.open(p).convert("RGB")}
            for p in image_paths
        ]
        content.append({"type": "text", "text": prompt})
        return self._run_messages([{"role": "user", "content": content}], max_new_tokens)
