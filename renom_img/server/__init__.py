import os
import enum
import weakref
from pathlib import Path

DB_DIR = Path("storage")
DB_DIR_TRAINED_WEIGHT = DB_DIR / "trained_weight"
DB_DIR_PRETRAINED_WEIGHT = DB_DIR / "pretrained_weight"

DATASET_DIR = Path("datasrc")
DATASET_IMG_DIR = DATASET_DIR / "img"
DATASET_LABEL_DIR = DATASET_DIR / "label"
DATASET_LABEL_CLASSIFICATION_DIR = DATASET_LABEL_DIR / "classification"
DATASET_LABEL_DETECTION_DIR = DATASET_LABEL_DIR / "detection"
DATASET_LABEL_SEGMENTATION_DIR = DATASET_LABEL_DIR / "segmentation"

DATASET_PREDICTION_DIR = DATASET_DIR / "prediction_set"
DATASET_PREDICTION_IMG_DIR = DATASET_PREDICTION_DIR / "img"

MAX_THREAD_NUM = 1

DATASET_NAME_MAX_LENGTH = 128
DATASET_NAME_MIN_LENGTH = 1
DATASET_DESCRIPTION_MAX_LENGTH = 1024
DATASET_DESCRIPTION_MIN_LENGTH = 0
DATASET_RATIO_MAX = 0.99
DATASET_RATIO_MIN = 0.3

EPOCH_MAX = 1000
EPOCH_MIN = 1
BATCH_MAX = 128
BATCH_MIN = 1
CELL_MAX = 10
CELL_MIN = 2
BBOX_MAX = 5
BBOX_MIN = 1


def create_directories():
    dirs = [
        DB_DIR, DB_DIR_TRAINED_WEIGHT, DB_DIR_PRETRAINED_WEIGHT,
        DATASET_IMG_DIR, DATASET_LABEL_DIR, DATASET_LABEL_CLASSIFICATION_DIR,
        DATASET_LABEL_DETECTION_DIR, DATASET_LABEL_SEGMENTATION_DIR,
        DATASET_PREDICTION_DIR, DATASET_PREDICTION_IMG_DIR
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


class Task(enum.Enum):
    CLASSIFICATION = 0
    DETECTION = 1
    SEGMENTATION = 2


class State(enum.Enum):
    CREATED = 0
    RESERVED = 1
    STARTED = 2
    STOPPED = 3

    PRED_CREATED = 4
    PRED_RESERVED = 5
    PRED_STARTED = 6


class RunningState(enum.Enum):
    PREPARING = 0
    TRAINING = 1
    VALIDATING = 2
    PREDICTING = 3
    STARTING = 4
    STOPPING = 5
    WEIGHT_DOWNLOADING = 6


class Algorithm(enum.Enum):
    RESNET = 1
    RESNEXT = 2
    DENSENET = 3
    VGG = 4
    INCEPTION = 5

    YOLOV1 = 30
    YOLOV2 = 31
    SSD = 32

    UNET = 60
    FCN = 61
    DEEPLABV3PLUS = 62


TASK_ID_BY_NAME = {
    "classification": Task.CLASSIFICATION.value,
    "detection": Task.DETECTION.value,
    "segmentation": Task.SEGMENTATION.value,
}

# name, description, min, max
ERROR_MESSAGE_TEMPLATE = "{} {}. Please input {} ~ {}."
