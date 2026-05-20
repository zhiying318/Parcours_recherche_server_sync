# spatial_eval/backends/qwen3vl.py
from dataclasses import dataclass
import gc
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
    enable_thinking: bool = False

    def __post_init__(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=self.dtype,
            device_map=self.device_map,
            attn_implementation=self.attn_implementation,
        )
        self.model.eval()

    def _build_messages(self, image_paths, prompt):
        content = [{"type": "image", "image": "file://" + os.path.abspath(p)} for p in image_paths]
        content.append({"type": "text", "text": prompt})
        return [{"role": "user", "content": content}]

    def _run_messages(self, messages, max_new_tokens):
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if self.enable_thinking:
            template_kwargs["enable_thinking"] = True
        text = self.processor.apply_chat_template(messages, **template_kwargs)
        images, videos = process_vision_info(messages, image_patch_size=16)
        inputs = self.processor(
            text=[text], images=images, videos=videos,
            do_resize=False, padding=True, return_tensors="pt",
        ).to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=int(max_new_tokens), do_sample=False)
        trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], output)]
        response = self.processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()

        del inputs, output, trimmed
        gc.collect()
        torch.cuda.empty_cache()

        return response

    @torch.inference_mode()
    def ask(self, image_path: str, prompt: str, max_new_tokens: int = 512) -> str:
        return self._run_messages(self._build_messages([image_path], prompt), max_new_tokens)

    @torch.inference_mode()
    def ask_multi(self, image_paths, prompt: str, max_new_tokens: int = 512) -> str:
        return self._run_messages(self._build_messages(image_paths, prompt), max_new_tokens)
