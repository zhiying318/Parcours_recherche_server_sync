# spatial_eval/backends/internvl.py
from dataclasses import dataclass
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from .base import VLMBackend

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

    def _run_messages(self, messages, max_new_tokens):
        inputs_raw = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        )
        inputs = {k: (v.to(self.model.device) if torch.is_tensor(v) else v) for k, v in inputs_raw.items()}
        output_ids = self.model.generate(
            **inputs, max_new_tokens=int(max_new_tokens), do_sample=False,
            pad_token_id=self.processor.tokenizer.eos_token_id,
        )
        input_len = inputs["input_ids"].shape[-1]
        return self.processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

    @torch.inference_mode()
    def ask(self, image_path: str, prompt: str, max_new_tokens: int = 64) -> str:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": Image.open(image_path).convert("RGB")},
            {"type": "text", "text": prompt},
        ]}]
        return self._run_messages(messages, max_new_tokens)

    @torch.inference_mode()
    def ask_multi(self, image_paths, prompt: str, max_new_tokens: int = 64) -> str:
        content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in image_paths]
        content.append({"type": "text", "text": prompt})
        return self._run_messages([{"role": "user", "content": content}], max_new_tokens)