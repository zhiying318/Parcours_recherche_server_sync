# spatial_eval/prompts/mcq.py
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple
import random
import re
from ..backends.base import VLMBackend

def _normalize_choice(x: str) -> str:
    s = (x or "").strip().upper()
    m = re.search(r"\b([ABCD])\b", s)
    if m:
        return m.group(1)
    m = re.match(r"^([ABCD])[\.\)\:\-]?", s)
    if m:
        return m.group(1)
    return s[:1]  # fallback

def _normalize_choice_thinking(x: str) -> str:
    s = (x or "").strip()
    s = s.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

    if "</think>" in s:
        s = s.split("</think>")[-1].strip()
    else:
        lines = [line.strip() for line in s.splitlines() if line.strip()]
        if lines:
            s = lines[-1]

    m = re.search(r"\b([ABCD])\b", s.upper())
    if m:
        return m.group(1)

    m = re.match(r"^([ABCD])[\.\)\:\-]?", s.upper())
    if m:
        return m.group(1)

    matches = re.findall(r"\b([ABCD])\b", (x or "").strip().upper())
    if matches:
        return matches[-1]

    return _normalize_choice(s)


@dataclass
class MCQAsker:
    seed: int = 0
    max_new_tokens_mcq: int = 512

    relations = {
        "front": "in the front of",
        "behind": "behind",
        "left": "on the left",
        "right": "on the right",
        }
    # relations = {
    #     "front": "front",
    #     "behind": "behind",
    #     "left": "left",
    #     "right": "right",
    #     }
    # relations = {
    #     "front": "From the person's perspective, the {second_object} is in front of them.",
    #     "behind": "From the person's perspective, the {second_object} is behind them.",
    #     "left": "From the person's perspective, the {second_object} is on their left.",
    #     "right": "From the person's perspective, the {second_object} is on their right.",
    #     }
    opposite_map = {
        "front": "behind",
        "behind": "front",
        "left": "right",
        "right": "left",
    }

    def evaluate_one(self, backend: VLMBackend, img_path: str, second_object: str, correct_relation: str) -> Dict[str, Any]:
        rng = random.Random(self.seed + hash(img_path))

        keys = list(self.relations.keys())
        rng.shuffle(keys)  # random change order of options

        letters = ["A", "B", "C", "D"]
        choices = {letters[i]: self.relations[keys[i]].format(second_object=second_object) for i in range(4)}

        correct_letter = letters[keys.index(correct_relation)]

        opposite_direction = self.opposite_map[correct_relation]
        opposite_letter = letters[keys.index(opposite_direction)]

        option_lines = "\n".join([f"{k}. {v}" for k, v in choices.items()])
        prompt = (
            f"Where is the {second_object} in the perspective of the person?\n"
            f"Choose ONE option and respond with ONLY the letter.\n"
            f"{option_lines}"
        )

        raw = backend.ask(img_path, prompt, self.max_new_tokens_mcq)
        # pred_letter = _normalize_choice(raw) # change to below: altomatic switch between thinking or not-thinking model
        if "thinking" in backend.model_id.lower() or "Qwen3.5" in backend.model_id:  # Qwen3.5 is thinking model by default
            pred_letter = _normalize_choice_thinking(raw)
        else:
            pred_letter = _normalize_choice(raw)

        return {
            "mcq_prompt": prompt,
            "options": choices,                
            "pred_letter": pred_letter,  
            "correct_relation": correct_relation,      
            "correct_letter": correct_letter,   
            "opposite_relation": opposite_direction,
            "opposite_letter": opposite_letter,
            "raw_answer": raw,
        }