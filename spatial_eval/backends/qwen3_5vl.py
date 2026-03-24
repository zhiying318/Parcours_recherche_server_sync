from dataclasses import dataclass
import os
from typing import Sequence

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

from .base import VLMBackend


@dataclass
class Qwen35VLBackend(VLMBackend):
    model_id: str
    device_map: str = "auto"
    dtype: str | torch.dtype = "auto"
    attn_implementation: str = "eager"

    # Qwen3.5 official recommended non-thinking / instruct decoding params
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    min_p: float = 0.0
    repetition_penalty: float = 1.0

    def __post_init__(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=self.dtype,
            device_map=self.device_map,
            attn_implementation=self.attn_implementation,
        )
        self.model.eval()

        # # 方便后面取 tokenizer
        # if hasattr(self.processor, "tokenizer"):
        #     self.tokenizer = self.processor.tokenizer
        # else:
        #     raise ValueError(
        #         "AutoProcessor does not expose a tokenizer. "
        #         "This backend expects processor.tokenizer to exist."
        #     )

    # def _build_messages(self, image_path: str, prompt: str):
    #     img_uri = "file://" + os.path.abspath(image_path)
    #     messages = [{
    #         "role": "user",
    #         "content": [
    #             {"type": "image", "image": img_uri},
    #             {"type": "text", "text": prompt},
    #         ],
    #     }]
    #     return messages

    def _build_inputs(self, image_path: str, prompt: str):
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
            enable_thinking=False,
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

        # target_device = self._get_input_device()
        # inputs = {k: v.to(target_device) if torch.is_tensor(v) else v for k, v in inputs.items()}
        return inputs

    # def _get_input_device(self):
    #     try:
    #         return self.model.get_input_embeddings().weight.device
    #     except Exception:
    #         return next(self.model.parameters()).device

    @torch.inference_mode()
    def ask(self, image_path: str, prompt: str, max_new_tokens: int = 512) -> str:
        inputs = self._build_inputs(image_path, prompt)

        output = self.model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=True,                    # non-thinking 推荐参数是采样模式
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self.processor.tokenizer.eos_token_id,
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

    @torch.inference_mode()
    def ask_logits(
        self,
        image_path: str,
        prompt: str,
        candidate_tokens: Sequence[str] = ("A", "B", "C", "D"),
    ) -> dict[str, float]:
        """
        返回“下一 token”层面的 logits，适合 MCQ 直接比较 A/B/C/D。

        返回格式示例:
        {
            "A": -1.23,
            "B":  0.52,
            "C": -0.11,
            "D": -2.04,
        }

        注意：
        1. 这里只看 prompt 之后“第一个生成 token”的 logits。
        2. candidate_tokens 最好都是单 token；A/B/C/D 通常没问题。
        """
        inputs = self._build_inputs(image_path, prompt)

        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]   # shape: [1, vocab_size]
        logits = logits[0]                  # shape: [vocab_size]

        result = {}
        for tok in candidate_tokens:
            token_ids = self.tokenizer.encode(tok, add_special_tokens=False)

            if len(token_ids) != 1:
                raise ValueError(
                    f"Candidate token {tok!r} is not a single token: token_ids={token_ids}. "
                    "For ask_logits(), please use single-token candidates."
                )

            result[tok] = float(logits[token_ids[0]].item())

        return result

    @torch.inference_mode()
    def ask_logits_full(
        self,
        image_path: str,
        prompt: str,
    ) -> torch.Tensor:
        """
        返回完整的下一 token logits，shape = [vocab_size]
        适合你之后自己在外面做更自由的后处理。
        """
        inputs = self._build_inputs(image_path, prompt)
        outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]   # [1, vocab_size]
        return logits[0].detach().float().cpu()