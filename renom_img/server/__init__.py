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

MAX_THREAD_NUM = 1


def create_directories():
    dirs = [
        DB_DIR, DB_DIR_TRAINED_WEIGHT, DB_DIR_PRETRAINED_WEIGHT,
        DATASET_IMG_DIR, DATASET_LABEL_DIR, DATASET_LABEL_CLASSIFICATION_DIR,
        DATASET_LABEL_DETECTION_DIR, DATASET_LABEL_SEGMENTATION_DIR
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
