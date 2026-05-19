from dataclasses import dataclass
import os
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

from .base import VLMBackend


@dataclass
class Qwen3ThinkingBackend(VLMBackend):
    model_id: str
    device_map: str = "auto"
    dtype: str | torch.dtype = "auto"
    enable_thinking: bool = True

    def __post_init__(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        self.model.eval()

    def _build_inputs(self, messages):
        return self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.model.device)

    def _decode_thinking(self, inputs, outputs):
        gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        return self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    @torch.inference_mode()
    def ask(self, image_path: str, prompt: str, max_new_tokens: int = 40960) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": Image.open(image_path).convert("RGB")},
            {"type": "text", "text": prompt},
        ]}]
        inputs = self._build_inputs(messages)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            repetition_penalty=1.0,
            pad_token_id=self.processor.tokenizer.eos_token_id, # for thinking models to avoid the "Setting `pad_token_id` to `eos_token_id`:248044 for open-end generation."
        )
        # see raw output for debugging
        gen_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        raw_response = self.processor.decode(
            gen_ids,
            skip_special_tokens=False,
        )
        # print("RAW:", repr(raw_response))

        response = self.processor.decode(
            outputs[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        )

        return response.strip()

    @torch.inference_mode()
    def ask_multi(self, image_paths, prompt: str, max_new_tokens: int = 40960) -> str:
        content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in image_paths]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self._build_inputs(messages)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            repetition_penalty=1.0,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        return self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
