# spatial_eval/backends/qwen3vl_logits.py
from dataclasses import dataclass
import os
from typing import Dict, List, Any

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info

from .base import VLMBackend


@dataclass
class Qwen3VLLogitsBackend(VLMBackend):
    model_id: str
    device_map: str | dict = "auto"
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
            add_generation_prompt=True
        )
        images, videos = process_vision_info(messages, image_patch_size=16)

        inputs = self.processor(
            text=[text],
            images=images,
            videos=videos,
            do_resize=False,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to(self.model.device)
        return inputs

    def _single_token_variants(self, choice: str) -> List[int]:
        """
        给一个选项（如 'A'），找出它在 tokenizer 下可能对应的单 token id。
        会尝试：
          - "A"
          - " A"
          - "\nA"
        只保留编码后长度为 1 的情况。
        """
        candidates = [choice, " " + choice, "\n" + choice]
        token_ids = []

        for cand in candidates:
            ids = self.processor.tokenizer.encode(cand, add_special_tokens=False)
            if len(ids) == 1:
                token_ids.append(ids[0])

        # 去重，保持顺序
        seen = set()
        uniq = []
        for x in token_ids:
            if x not in seen:
                uniq.append(x)
                seen.add(x)
        return uniq

    def _extract_choice_logits(
        self,
        step_logits: torch.Tensor,
        choices: List[str],
    ) -> Dict[str, Any]:
        """
        step_logits: shape [vocab_size]，表示“第一个生成 token”的全词表 logits
        返回：
          {
            "option_logits": {"A": ..., "B": ..., ...},
            "option_probs": {"A": ..., "B": ..., ...},
            "token_ids": {"A": [...], "B": [...], ...}
          }
        """
        option_logits: Dict[str, float] = {}
        token_ids_map: Dict[str, List[int]] = {}

        for choice in choices:
            token_ids = self._single_token_variants(choice)
            token_ids_map[choice] = token_ids

            if len(token_ids) == 0:
                option_logits[choice] = float("-inf")
                continue

            # 如果同一个 choice 有多个单 token 变体，取最大 logit
            vals = step_logits[token_ids]
            option_logits[choice] = float(vals.max().item())

        # 在选项内部做 softmax，得到 option-level probability
        logits_tensor = torch.tensor(
            [option_logits[c] for c in choices],
            dtype=torch.float32
        )
        probs_tensor = torch.softmax(logits_tensor, dim=0)

        option_probs = {
            c: float(probs_tensor[i].item())
            for i, c in enumerate(choices)
        }

        return {
            "option_logits": option_logits,
            "option_probs": option_probs,
            "token_ids": token_ids_map,
        }

    @torch.inference_mode()
    def ask(self, image_path: str, prompt: str, max_new_tokens: int = 512) -> str:
        """
        和原 backend 保持一致：只返回生成文本。
        """
        inputs = self._build_inputs(image_path, prompt)

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

    @torch.inference_mode()
    def ask_with_logits(
        self,
        image_path: str,
        prompt: str,
        choices: List[str] | None = None,
        max_new_tokens: int = 32,
    ) -> Dict[str, Any]:
        """
        返回完整生成文本 + 第一个生成位置上各选项的 logits/probs。
        默认 choices = ["A", "B", "C", "D"]
        """
        if choices is None:
            choices = ["A", "B", "C", "D"]

        inputs = self._build_inputs(image_path, prompt)

        generation = self.model.generate(
            **inputs,
            max_new_tokens=int(max_new_tokens),
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # 完整输出文本
        sequences = generation.sequences
        trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs["input_ids"], sequences)
        ]
        resp = self.processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        # 第一个生成步的 logits，shape = [batch_size, vocab_size]
        if generation.scores is None or len(generation.scores) == 0:
            raise RuntimeError("No generation scores returned by model.generate().")

        first_step_logits = generation.scores[0][0]  # [vocab_size]
        choice_info = self._extract_choice_logits(first_step_logits, choices)

        pred_by_logits = max(
            choice_info["option_logits"].items(),
            key=lambda x: x[1]
        )[0]

        pred_by_probs = max(
            choice_info["option_probs"].items(),
            key=lambda x: x[1]
        )[0]

        return {
            "text": resp,
            "pred_by_logits": pred_by_logits,
            "pred_by_probs": pred_by_probs,
            "option_logits": choice_info["option_logits"],
            "option_probs": choice_info["option_probs"],
            "token_ids": choice_info["token_ids"],
        }

    @torch.inference_mode()
    def get_choice_logits(
        self,
        image_path: str,
        prompt: str,
        choices: List[str] | None = None,
    ) -> Dict[str, Any]:
        """
        只关心各选项 logits/probs 时可直接调这个。
        """
        if choices is None:
            choices = ["A", "B", "C", "D"]

        inputs = self._build_inputs(image_path, prompt)

        generation = self.model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

        if generation.scores is None or len(generation.scores) == 0:
            raise RuntimeError("No generation scores returned by model.generate().")

        first_step_logits = generation.scores[0][0]
        choice_info = self._extract_choice_logits(first_step_logits, choices)

        return choice_info