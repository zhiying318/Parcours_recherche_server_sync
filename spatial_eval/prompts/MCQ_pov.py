# spatial_eval/prompts/MCQ_pov.py
# POV image-choice evaluation:
# - Input : 3 images sent to the model
#     Image 1: external camera view of the scene
#     Image 2: one POV option (either correct or distractor, randomly ordered)
#     Image 3: the other POV option
# - Question: which of Image 2 / Image 3 represents what the person actually sees?
# - Correct answer: cam_pov_front (person's facing direction)
# - Distractor:     cam_pov_back  (opposite direction)

from dataclasses import dataclass
from typing import Dict, Any, List
import random
import re
from ..backends.base import VLMBackend
from .MCQ import _normalize_choice, _normalize_choice_thinking


@dataclass
class MCQPovAsker:
    """Image-as-option MCQ: given 1 external view, pick the correct POV image."""
    seed: int = 0
    max_new_tokens: int = 512

    def evaluate_one(
        self,
        backend: VLMBackend,
        external_img: str,
        correct_pov_img: str,   # cam_pov_front
        distractor_pov_img: str,  # cam_pov_back
    ) -> Dict[str, Any]:
        rng = random.Random(self.seed + hash(external_img))

        # Randomly assign correct/distractor to option A or B
        if rng.random() < 0.5:
            img_A, img_B = correct_pov_img, distractor_pov_img
            correct_letter = "A"
        else:
            img_A, img_B = distractor_pov_img, correct_pov_img
            correct_letter = "B"

        prompt = (
            "The first image shows a scene from an external camera.\n"
            "A person is standing at the center of the scene.\n"
            "One of the following two images was taken from the person's eye level, "
            "looking in the direction the person is facing.\n"
            "Which image represents what the person sees?\n"
            "Choose ONE option and respond with ONLY the letter.\n"
            "A. Image 2\n"
            "B. Image 3"
        )

        raw = backend.ask_multi([external_img, img_A, img_B], prompt, self.max_new_tokens)

        if getattr(backend, "enable_thinking", False):
            pred_letter = _normalize_choice_thinking(raw)
        else:
            pred_letter = _normalize_choice(raw)

        return {
            "mcq_prompt": prompt,
            "img_option_A": img_A,
            "img_option_B": img_B,
            "correct_letter": correct_letter,
            "raw_answer": raw,
            "pred_letter": pred_letter,
            "correct": pred_letter == correct_letter,
        }
