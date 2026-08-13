# spatial_eval/prompts/mcq.py
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional
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
    """Extract only a terminal answer from a completed thinking response.

    Thinking can contain many option letters and ordinary English articles such
    as ``a``.  Searching the upper-cased response for the first standalone A-D
    therefore produces false answers.  A response without ``</think>`` was cut
    off before its answer section and is deliberately recorded as incomplete.
    """
    s = (x or "").strip()
    s = s.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
    if "</think>" not in s:
        return ""

    answer = s.rsplit("</think>", 1)[-1].strip()
    if not answer:
        return ""

    tagged = re.findall(
        r"<answer>\s*([A-D])\s*</answer>", answer, flags=re.IGNORECASE
    )
    if tagged:
        return tagged[-1].upper()

    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    for line in reversed(lines):
        exact = re.fullmatch(
            r"(?:\*\*|`)?\s*([A-D])\s*[\.\)\:\-]?\s*(?:\*\*|`)?",
            line,
            flags=re.IGNORECASE,
        )
        if exact:
            return exact.group(1).upper()

        option_line = re.match(r"^([A-D])[\.\)\:\-]\s+\S", line)
        if option_line:
            return option_line.group(1)

    explicit_patterns = (
        r"(?:final\s+answer|answer|option|choice)\s*(?:is|:|=)?\s*"
        r"(?:\*\*|`)?([A-D])\b",
        r"\b([A-D])\s+is\s+(?:the\s+)?correct(?:\s+answer|\s+option)?\b",
    )
    explicit = []
    for pattern in explicit_patterns:
        explicit.extend(re.findall(pattern, answer, flags=re.IGNORECASE))
    return explicit[-1].upper() if explicit else ""


@dataclass
class MCQAsker:
    answer_length: str # no default value, put behind those with default value to avoid dataclass error: TypeError: non-default argument 'answer_length' follows default argument
    seed: int = 0
    max_new_tokens_mcq: int = 512
    prompt_note: str = ""
    prompt_info_by_image: Optional[Dict[str, str]] = None
    prompt_info_before_question: bool = False

    relations_middle = {
        "front": "in the front of",
        "behind": "behind",
        "left": "on the left",
        "right": "on the right",
        }
    relations_short = {
        "front": "front",
        "behind": "behind",
        "left": "left",
        "right": "right",
        }
    relations_long = {
        "front": "From the person's perspective, the {second_object} is in front of them.",
        "behind": "From the person's perspective, the {second_object} is behind them.",
        "left": "From the person's perspective, the {second_object} is on their left.",
        "right": "From the person's perspective, the {second_object} is on their right.",
        }
    
    def get_relations(self) -> Dict[str, str]:
        mapping = {
            "short": self.relations_short,
            "middle": self.relations_middle,
            "long": self.relations_long,
        }
        return mapping[self.answer_length]
    
    opposite_map = {
        "front": "behind",
        "behind": "front",
        "left": "right",
        "right": "left",
    }

    def evaluate_one(self, backend: VLMBackend, img_path: str, second_object: str, correct_relation: str) -> Dict[str, Any]:
        rng = random.Random(self.seed + hash(img_path))

        keys = list(self.get_relations().keys())
        rng.shuffle(keys)  # random change order of options

        letters = ["A", "B", "C", "D"]
        choices = {letters[i]: self.get_relations()[keys[i]].format(second_object=second_object) for i in range(4)}

        correct_letter = letters[keys.index(correct_relation)]

        opposite_direction = self.opposite_map[correct_relation]
        opposite_letter = letters[keys.index(opposite_direction)]

        option_lines = "\n".join([f"{k}. {v}" for k, v in choices.items()])
        question = f"Where is the {second_object} in the perspective of the person?"
        prompt_lines = []
        prompt_info = None
        if self.prompt_info_by_image is not None:
            lookup_key = img_path[2:] if img_path.startswith("./") else img_path
            if lookup_key not in self.prompt_info_by_image:
                raise KeyError(f"No per-image prompt information for {lookup_key!r}")
            prompt_info = self.prompt_info_by_image[lookup_key]
            if not isinstance(prompt_info, str) or not prompt_info.strip():
                raise ValueError(f"Invalid per-image prompt information for {lookup_key!r}")
            prompt_info = prompt_info.strip()

        if self.prompt_info_before_question and prompt_info is not None:
            prompt_lines.append(prompt_info)
        prompt_lines.append(question)
        if self.prompt_note:
            prompt_lines.append(self.prompt_note)
        if not self.prompt_info_before_question and prompt_info is not None:
            prompt_lines.append(prompt_info)
        prompt_lines.extend([
            "Choose ONE option and respond with ONLY the letter.",
            option_lines,
        ])
        prompt = "\n".join(prompt_lines)

        raw = backend.ask(img_path, prompt, self.max_new_tokens_mcq)
        # pred_letter = _normalize_choice(raw) # change to below: altomatic switch between thinking or not-thinking model
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
