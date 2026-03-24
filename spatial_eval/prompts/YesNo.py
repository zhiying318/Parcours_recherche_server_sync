# spatial_eval/prompts/yn.py
from dataclasses import dataclass
from typing import Dict, Any
from ..utils import yn
from ..backends.base import VLMBackend

@dataclass
class YNAsker:
    max_new_tokens_where: int = 64
    max_new_tokens_yn: int = 8

    def evaluate_one(self, backend: VLMBackend, img_path: str, second_object: str) -> Dict[str, Any]:
        where_prompt = f"Where is the {second_object} in the view of the human in the image?"
        front_prompt = f"Is the {second_object} in the front of the human? Answer only yes or no."
        back_prompt  = f"Is the {second_object} in the back of the human? Answer only yes or no."
        left_prompt  = f"Is the {second_object} on the left of the human? Answer only yes or no."
        right_prompt = f"Is the {second_object} on the right of the human? Answer only yes or no."

        where_ans = backend.ask(img_path, where_prompt, self.max_new_tokens_where)
        front_ans = yn(backend.ask(img_path, front_prompt, self.max_new_tokens_yn))
        back_ans  = yn(backend.ask(img_path, back_prompt,  self.max_new_tokens_yn))
        left_ans  = yn(backend.ask(img_path, left_prompt,  self.max_new_tokens_yn))
        right_ans = yn(backend.ask(img_path, right_prompt, self.max_new_tokens_yn))

        return {
            "where_prompt": where_prompt,
            "where_answer": where_ans,
            "front_answer": front_ans,
            "back_answer": back_ans,
            "left_answer": left_ans,
            "right_answer": right_ans,
        }