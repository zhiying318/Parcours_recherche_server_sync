# spatial_eval/backends/qwen2.py
from dataclasses import dataclass
import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, AutoModelForVision2Seq
from .base import VLMBackend

@dataclass
class Qwen2Backend(VLMBackend):
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