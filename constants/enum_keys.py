from enum import Enum, auto


class HK(Enum):
    NATIVE_IMAGE = auto()
    KEYPOINTS = auto()
    VISIBILITIES = auto()
    BOXES = auto()


class HG(Enum):
    VIDEO_PATH = auto()
    VIDEO_NAME = 'VIDEO_NAME'
    GESTURE_LABEL = auto()

    COORD = auto()

    BONE_LENGTH = auto()
    BONE_DEPTH = auto()
    BONE_ANGLE_SIN = auto()
    BONE_ANGLE_COS = auto()

    OUT_SCORES = auto()
    OUT_ARGMAX = auto()

    PRED_GESTURE = 'PRED_GESTURE'

