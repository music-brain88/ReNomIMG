import os

# Constants
# Algorithms
ALG_YOLOV1 = 0
ALG_YOLOV2 = 1
ALG_SSD = 2

# Model state
STATE_CREATED = 0  # This is used only client env.
STATE_RUNNING = 1
STATE_FINISHED = 2
STATE_DELETED = 3
STATE_RESERVED = 4

# Model running state
RUN_STATE_TRAINING = 0
RUN_STATE_VALIDATING = 1
RUN_STATE_PREDICTING = 2
RUN_STATE_STARTING = 3
RUN_STATE_STOPPING = 4

# Thread
MAX_THREAD_NUM = 2
GPU_NUM = 0

try:
    import renom.cuda
    if renom.cuda.has_cuda():
        MAX_THREAD_NUM = GPU_NUM = renom.cuda.cuGetDeviceCount()
except Exception:
    pass


# Weight
WEIGHT_EXISTS = 0
WEIGHT_CHECKING = 1
WEIGHT_DOWNLOADING = 2

# Directories
DATASRC_DIR = "datasrc"
DATASRC_IMG = os.path.join(DATASRC_DIR, "img")
DATASRC_LABEL = os.path.join(DATASRC_DIR, "label")
DATASRC_PREDICTION_IMG = os.path.join(DATASRC_DIR, "prediction_set", "img")
DATASRC_PREDICTION_OUT = os.path.join(DATASRC_DIR, "prediction_set", "output")
DATASRC_PREDICTION_OUT_CSV = os.path.join(DATASRC_DIR, "prediction_set", "output", "csv")
DATASRC_PREDICTION_OUT_XML = os.path.join(DATASRC_DIR, "prediction_set", "output", "xml")

# DB directories
DB_DIR = "storage"
DB_DIR_TRAINED_WEIGHT = os.path.join(DB_DIR, "trained_weight")
DB_DIR_PRETRAINED_WEIGHT = os.path.join(DB_DIR, "pretrained_weight")


def create_dirs():
    # Create directories
    for path in [DATASRC_IMG, DATASRC_LABEL,
                 DB_DIR_TRAINED_WEIGHT, DB_DIR_PRETRAINED_WEIGHT, DATASRC_PREDICTION_IMG,
                 DATASRC_PREDICTION_OUT_CSV, DATASRC_PREDICTION_OUT_XML]:
        if not os.path.exists(path):
            os.makedirs(path)
