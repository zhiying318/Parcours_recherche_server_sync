# spatial_eval/prompts/MCQ_pair.py
from dataclasses import dataclass
from typing import Dict, Any, List
import random
import re
from ..backends.base import VLMBackend
from .MCQ import _normalize_choice, _normalize_choice_thinking


@dataclass
class MCQPairAsker:
    """Same as MCQAsker but sends two images (same scene, different camera angle)."""
    answer_length: str
    seed: int = 0
    max_new_tokens_mcq: int = 512

    relations_short = {
        "front": "front",
        "behind": "behind",
        "left": "left",
        "right": "right",
    }
    relations_middle = {
        "front": "in the front of",
        "behind": "behind",
        "left": "on the left",
        "right": "on the right",
    }
    relations_long = {
        "front": "From the person's perspective, the {second_object} is in front of them.",
        "behind": "From the person's perspective, the {second_object} is behind them.",
        "left": "From the person's perspective, the {second_object} is on their left.",
        "right": "From the person's perspective, the {second_object} is on their right.",
    }

    def get_relations(self) -> Dict[str, str]:
        return {"short": self.relations_short, "middle": self.relations_middle, "long": self.relations_long}[self.answer_length]

    opposite_map = {"front": "behind", "behind": "front", "left": "right", "right": "left"}

    def evaluate_one(
        self,
        backend: VLMBackend,
        img_paths: List[str],
        second_object: str,
        correct_relation: str,
    ) -> Dict[str, Any]:
        # Use first image path as seed key so shuffling is deterministic per scene
        rng = random.Random(self.seed + hash(img_paths[0]))

        keys = list(self.get_relations().keys())
        rng.shuffle(keys)

        letters = ["A", "B", "C", "D"]
        choices = {letters[i]: self.get_relations()[keys[i]].format(second_object=second_object) for i in range(4)}

        correct_letter = letters[keys.index(correct_relation)]
        opposite_direction = self.opposite_map[correct_relation]
        opposite_letter = letters[keys.index(opposite_direction)]

        option_lines = "\n".join([f"{k}. {v}" for k, v in choices.items()])
        # Prompt identical to single-image MCQAsker
        prompt = (
            f"Here are two images of the same scene from different camera angles.\n"
            # f"The first image has an aligned perspective with the person.\n"
            f"Where is the {second_object} in the perspective of the person?\n"
            f"Choose ONE option and respond with ONLY the letter.\n"
            f"{option_lines}"
        )

        raw = backend.ask_multi(img_paths, prompt, self.max_new_tokens_mcq)

        if getattr(backend, "enable_thinking", False):
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
