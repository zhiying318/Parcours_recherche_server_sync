# spatial_eval/prompts/MCQ_pov_four.py
# 3-choice POV image MCQ (test03):
# - Input: 4 images sent to the model
#     Image 1: external camera view
#     Images 2-4: 3 POV options (randomly ordered A/B/C)
# - Correct answer: always cam_pov_front (person faces forward)

from dataclasses import dataclass
from typing import Dict, Any
import random
from ..backends.base import VLMBackend
from .MCQ import _normalize_choice, _normalize_choice_thinking


@dataclass
class MCQPovFourAsker:
    """3-choice image MCQ: given 1 external view, pick the correct POV image."""
    seed: int = 0
    max_new_tokens: int = 512

    def evaluate_one(
        self,
        backend: VLMBackend,
        external_img: str,
        pov_imgs: Dict[str, str],   # {cam_name: path} — exactly 3 entries
        correct_cam: str,           # always "cam_pov_front"
    ) -> Dict[str, Any]:
        rng = random.Random(self.seed + hash(external_img))

        cam_names = list(pov_imgs.keys())
        rng.shuffle(cam_names)

        letters = ["A", "B", "C"]
        ordered_imgs   = [pov_imgs[c] for c in cam_names]
        correct_letter = letters[cam_names.index(correct_cam)]

        prompt = (
            "The first image shows a scene from an external camera.\n"
            "A person is standing at the center of the scene.\n"
            "Three of the following images were each taken from the person's eye level, "
            "looking in a different direction.\n"
            "Which image represents what the person sees when looking straight ahead?\n"
            "Choose ONE option and respond with ONLY the letter.\n"
            "A. Image 2\n"
            "B. Image 3\n"
            "C. Image 4"
        )

        all_imgs = [external_img] + ordered_imgs
        raw = backend.ask_multi(all_imgs, prompt, self.max_new_tokens)

        if getattr(backend, "enable_thinking", False):
            pred_letter = _normalize_choice_thinking(raw)
        else:
            pred_letter = _normalize_choice(raw)

        return {
            "mcq_prompt":      prompt,
            "cam_order":       cam_names,
            "correct_cam":     correct_cam,
            "correct_letter":  correct_letter,
            "raw_answer":      raw,
            "pred_letter":     pred_letter,
            "correct":         pred_letter == correct_letter,
        }
