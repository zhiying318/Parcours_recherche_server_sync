# spatial_eval/utils.py
import re
import torch
from typing import Tuple

def get_dtype() -> torch.dtype:
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float16

def yn(x: str) -> str:
    x = (x or "").strip().lower()
    x = re.split(r"\s+", x)[0].strip(".,;:!()[]{}<>\"'`")
    if x in ("yes", "y", "true", "1"):
        return "yes"
    if x in ("no", "n", "false", "0"):
        return "no"
    return x

def parse_relation_from_basename(base: str) -> Tuple[str, str]:
    """
    base example (no extension):
      book_right_of_chair_FACE-CAMERA
      beer-bottle_left_of_chair_FACE-LEFT
    returns: (second_object, correct_relation)
    """
    core = base.split("_FACE-", 1)[0]
    second_object = core.split("_")[0]
    direction1 = core.split("_")[1]
    direction2 = base.split("_FACE-", 1)[1]

    correct_relation = None

    if direction2 == "CAMERA":
        if direction1 == "right":
            correct_relation = "left"
        elif direction1 == "left":
            correct_relation = "right"
    elif direction2 == "LEFT":
        if direction1 == "right":
            correct_relation = "behind"
        elif direction1 == "left":
            correct_relation = "front"
    elif direction2 == "RIGHT":
        if direction1 == "right":
            correct_relation = "front"
        elif direction1 == "left":
            correct_relation = "behind"

    if correct_relation is None:
        raise ValueError(f"Unrecognized pattern for base='{base}'")

    return second_object, correct_relation