# comfort_human_car_config.py
import math
from data_generation.constants import *
import random


# =========================================================
# default config
# =========================================================

COMFORT_HUMAN_CAR_DEFAULT_CONFIG = {
    # render config
    "variation": "default",
    "num_steps": 1,  # 静态场景，每个 relation + camera 只渲染 1 张
    "save_path": None,

    # addressee object config: human 固定在中心
    "addressee": True,
    "addressee_shape": SOPHIA,
    "addressee_position": (0.0, 0.0, 0.0),
    "addressee_size": 0.015,
    "addressee_rotation": (90, 0, 270), # Changed from 90 to 270 on the Z-axis (Yaw) to rotate 180 degrees - so that camera in fromt of the human == see the face

    # variation config
    "num_distractors": 0,

    # 新增：相机看向的位置，默认看向 human
    "cam_look_at": (0.0, 0.0, 0.0),
}


# =========================================================
# object + camera variations
# 每个 variation = 一个物体 × 一个相机视角
# =========================================================
CAMERA_DISTANCE = 10.0
CAMERA_HEIGHT = 2.0 # match the rendering style of comfort_car_left

CAMERA_FRONT = {
    "view_name": "cam_front",
    "cam_position": (0.0, -CAMERA_DISTANCE, CAMERA_HEIGHT),
}

CAMERA_BACK = {
    "view_name": "cam_back",
    "cam_position": (0.0, CAMERA_DISTANCE, CAMERA_HEIGHT),
}

CAMERA_LEFT = {
    "view_name": "cam_left",
    "cam_position": (CAMERA_DISTANCE, 0.0, CAMERA_HEIGHT), #这样左右才没有反，之前反了。相机在人左边的时候camera position应该是10，0，2
}

CAMERA_RIGHT = {
    "view_name": "cam_right",
    "cam_position": (-CAMERA_DISTANCE, 0.0, CAMERA_HEIGHT),
}

ALL_CAMERAS = [
    CAMERA_FRONT,
    CAMERA_BACK,
    CAMERA_LEFT,
    CAMERA_RIGHT,
]


OBJECT_CAR = {
    "object_name": "car",
    "ref_shape": CAR_SEDAN,
    "ref_color": WHITE,
    "ref_size": 4.0, # change the size of the car to make it 'proportionate' to the human in the scene; 
    # 这是“物体在原点附近时”的 base pose；后面会加 cardinal_offset
    "ref_position": (0.0, -0.1, 0.45), # the car's position
    "ref_rotation": (90, 0, 180),
}

OBJECT_BICYCLE = {
    "object_name": "bicycle",
    "ref_shape": BICYCLE_MOUNTAIN,
    "ref_color": RED,
    "ref_size": 2.6,
    "ref_position": (0.05, 0.0, 0.65),
    "ref_rotation": (90, 0, 270),
}

OBJECT_DOG = {
    "object_name": "dog",
    "ref_shape": DOG,
    "ref_color": "",
    "ref_size": 0.1,
    "ref_position": (0.0, -0.6, 0.0),
    "ref_rotation": (0, 0, 0),
}

OBJECT_SOFA = {
    "object_name": "sofa",
    "ref_shape": COUCH,
    "ref_color": "",
    "ref_size": 1.8,
    "ref_position": (0.0, 0.0, 0.0),
    "ref_rotation": (0, 0, 360),
}

OBJECT_BED = {
    "object_name": "bed",
    "ref_shape": BED,
    "ref_color": "",
    "ref_size": 2.5,
    "ref_position": (0.0, 0.0, 0.0),
    "ref_rotation": (0, 0, 0),
}

OBJECT_BENCH = {
    "object_name": "bench",
    "ref_shape": BENCH,
    "ref_color": "",
    "ref_size": 1.8,
    "ref_position": (0.0, 0.0, 0.0),
    "ref_rotation": (0, 0, 0),
}

OBJECT_CHAIR = {
    "object_name": "chair",
    "ref_shape": CHAIR,
    "ref_color": "",
    "ref_size": 1.7,
    "ref_position": (0.0, 0.0, 0.0),
    "ref_rotation": (0, 0, 0),
}

OBJECT_DUCK = {
    "object_name": "duck",
    "ref_shape": DUCK,
    "ref_color": "",
    "ref_size": 1.8,
    "ref_position": (0.0, 0.0, 0.0),
    "ref_rotation": (0, 0, 0),
}

OBJECT_HORSE = {
    "object_name": "horse",
    "ref_shape": HORSE_L,   # 或 HORSE_R，看 constants.py 里的定义
    "ref_color": "",
    "ref_size": 2.0,
    "ref_position": (0.0, 0.0, 0.0),
    "ref_rotation": (0, 0, 0),
}

OBJECT_LAPTOP = {
    "object_name": "laptop",
    "ref_shape": LAPTOP,
    "ref_color": "",
    "ref_size": 3.0, #3.6有点大
    "ref_position": (0.0, 0.0, 0.0),
    "ref_rotation": (0, 0, 0),
}

OBJECT_BASKETBALL = {
    "object_name": "basketball",
    "ref_shape": BASKETBALL,
    "ref_color": "",
    "ref_size": 2.0, #3.6太大, 2.5有点大
    "ref_position": (0.0, 0.0, 0.0),
    "ref_rotation": (0, 0, 0),
}

ALL_OBJECTS = [
    OBJECT_CAR,
    OBJECT_BICYCLE,
    OBJECT_DOG,
    OBJECT_SOFA,
    # OBJECT_BED,
    OBJECT_BENCH,
    OBJECT_CHAIR,
    OBJECT_DUCK,
    # OBJECT_HORSE,
    OBJECT_LAPTOP,
    OBJECT_BASKETBALL,
]
COMFORT_HUMAN_CAR_OBJECTS = ALL_OBJECTS

# ALL_OBJECTS = [
#     OBJECT_CAR,
#     OBJECT_BICYCLE,
#     OBJECT_DOG,
#     OBJECT_SOFA,
# ]
# COMFORT_HUMAN_CAR_OBJECTS = ALL_OBJECTS


# =========================================================
# relations
# 这里 relation 表示：物体相对于 human 的位置
# =========================================================

CARDINAL_DISTANCE = 2.5

COMFORT_HUMAN_CAR_RELATIONS = {
    LEFT: {
        "relation": LEFT,
        "path_type": "cardinal_static",
        "cardinal_offset": (CARDINAL_DISTANCE, 0.0, 0.0), # 物体在 human 左边时，物体相对于 human 的位置是 (2.5, 0, 0)
    },
    RIGHT: {
        "relation": RIGHT,
        "path_type": "cardinal_static",
        "cardinal_offset": (-CARDINAL_DISTANCE, 0.0, 0.0),
    },
    FRONT: {
        "relation": FRONT,
        "path_type": "cardinal_static",
        "cardinal_offset": (0.0, -CARDINAL_DISTANCE, 0.0),
    },
    BEHIND: {
        "relation": BEHIND,
        "path_type": "cardinal_static",
        "cardinal_offset": (0.0, CARDINAL_DISTANCE, 0.0),
    },
}