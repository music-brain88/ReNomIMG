import os

# Constants
# Algorithms
ALG_YOLOV1 = 0
ALG_YOLOV2 = 1
ALG_SSD = 2

# Model state
STATE_FINISHED = 0
STATE_RUNNING = 0
STATE_DELETED = 0
STATE_RESERVED = 0

# Model running state
RUN_STATE_TRAINING = 0
RUN_STATE_VALIDATING = 1
RUN_STATE_PREDICTING = 2
RUN_STATE_STARTING = 0
RUN_STATE_STOPPING = 0

# Thread
MAX_THREAD_NUM = 2

# Directories
BASE_DIR = os.path.abspath(os.getcwd())
DATASRC_DIR = os.path.join(BASE_DIR, "datasrc")
DATASRC_IMG = os.path.join(DATASRC_DIR, "img")
DATASRC_LABEL = os.path.join(DATASRC_DIR, "label")

# DB directories
DB_DIR = os.path.join(BASE_DIR, "storage")
DB_DIR_TRAINED_WEIGHT = os.path.join(DB_DIR, "trained_weight")
DB_DIR_PRETRAINED_WEIGHT = os.path.join(DB_DIR, "pretrained_weight")

# Create directories
for path in [DATASRC_IMG, DATASRC_LABEL,
             DB_DIR_TRAINED_WEIGHT, DB_DIR_PRETRAINED_WEIGHT]:
    os.makedirs(path, exist_ok=True)
