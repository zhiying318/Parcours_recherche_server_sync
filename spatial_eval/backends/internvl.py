# spatial_eval/backends/internvl.py
from dataclasses import dataclass
import gc
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from .base import VLMBackend

R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different \
angles, potential solutions, and reason through the problem step-by-step. \
Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to \
the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the \
query. The final answer should be standalone and not reference the thinking \
section.
""".strip()

@dataclass
class InternVLBackend(VLMBackend):
    model_id: str
    device_map: str = "auto"
    dtype: torch.dtype = torch.float16
    enable_thinking: bool = False

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
        generate_kwargs = {
            "max_new_tokens": int(max_new_tokens),
            "pad_token_id": self.processor.tokenizer.eos_token_id,
        }
        if self.enable_thinking:
            generate_kwargs.update({
                "do_sample": True,
                "temperature": 0.6,
            })
        else:
            generate_kwargs["do_sample"] = False
        output_ids = self.model.generate(
            **inputs,
            **generate_kwargs,
        )
        input_len = inputs["input_ids"].shape[-1]
        response = self.processor.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()

        del inputs_raw, inputs, output_ids
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return response

    @torch.inference_mode()
    def ask(self, image_path: str, prompt: str, max_new_tokens: int = 64) -> str:
        messages = []
        if self.enable_thinking:
            messages.append({"role": "system", "content": R1_SYSTEM_PROMPT})
        messages.append({"role": "user", "content": [
            {"type": "image", "image": Image.open(image_path).convert("RGB")},
            {"type": "text", "text": prompt},
        ]})
        return self._run_messages(messages, max_new_tokens)

    @torch.inference_mode()
    def ask_multi(self, image_paths, prompt: str, max_new_tokens: int = 64) -> str:
        content = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in image_paths]
        content.append({"type": "text", "text": prompt})
        messages = []
        if self.enable_thinking:
            messages.append({"role": "system", "content": R1_SYSTEM_PROMPT})
        messages.append({"role": "user", "content": content})
        return self._run_messages(messages, max_new_tokens)
