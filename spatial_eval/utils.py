# spatial_eval/utils.py
import re
from typing import Any, Tuple

def get_dtype() -> Any:
    import torch

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


import os

def parse_relation_from_basename(base: str):
    """
    处理这种名字：
    bottle-of-orange-juice_left_of_chair_FACE-CAMERA
    返回:
        second_object = bottle-of-orange-juice
        correct_relation = left
    """
    parts = base.split("_")

    if len(parts) < 4:
        raise ValueError(f"Unexpected basename format: {base}")

    second_object = parts[0]
    correct_relation = parts[1]  # left / right / front / behind

    return second_object, correct_relation


def parse_relation_for_COMFORT(img_path: str):
    """
    处理这种路径：
    COMFORT/data/comfort_human_car/behind/basketball__behind__cam_back/0.png

    这里先从目录名里提取：
        second_object = basketball
        correct_relation = behind   # 你后面如果要改成别的逻辑，再改这个函数

    说明：
    - 倒数第二级目录: basketball__behind__cam_back
    - 按 __ 拆分后:
        [0] second_object
        [1] relation
        [2] camera view
    """
    folder_name = os.path.basename(os.path.dirname(img_path))
    parts = folder_name.split("__",1) 

    second_object = parts[0]
    correct_relation = parts[1].split("__")[0] 
    print(f"Parsing COMFORT path: {img_path}, second_object: {second_object}, correct_relation: {correct_relation}")

    return second_object, correct_relation
