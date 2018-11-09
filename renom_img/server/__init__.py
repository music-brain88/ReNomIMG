import os
import enum
import weakref


class Task(enum.Enum):
    CLASSIFICATION = 0
    DETECTION = 1
    SEGMENTATION = 2


class State(enum.Enum):
    CREATED = 0
    RESERVED = 1
    STARTED = 2
    STOPPED = 3


class RunningState(enum.Enum):
    PREPARING = 0
    TRAINING = 1
    VALIDATING = 2
    PREDICTING = 3
    STARTING = 4
    STOPPING = 5


class Algorithm(enum.Enum):
    YOLOV1 = 10
    YOLOV2 = 11
    SSD = 12


DB_DIR = "storage"
DB_DIR_TRAINED_WEIGHT = os.path.join(DB_DIR, "trained_weight")
DB_DIR_PRETRAINED_WEIGHT = os.path.join(DB_DIR, "pretrained_weight")

MAX_THREAD_NUM = 1
