import os
import time
import numpy as np
from threading import Event, Semaphore
from renom.cuda import set_cuda_active, release_mem_pool, use_device

from renom_img.api.detection.yolo_v1 import Yolov1
from renom_img.api.detection.yolo_v2 import Yolov2, create_anchor
from renom_img.api.detection.ssd import SSD
from renom_img.api.classification.vgg import VGG16
from renom_img.api.utility.load import parse_xml_detection
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.augmentation.process import Shift, Rotate, Flip, WhiteNoise, ContrastNorm
from renom_img.api.utility.augmentation import Augmentation

from renom_img.api.utility.evaluate.detection import get_ap_and_map, get_prec_rec_iou

from renom_img.server import ALG_YOLOV1, ALG_YOLOV2, ALG_SSD
from renom_img.server import WEIGHT_EXISTS, WEIGHT_CHECKING, WEIGHT_DOWNLOADING
from renom_img.server import DB_DIR_TRAINED_WEIGHT, DB_DIR_PRETRAINED_WEIGHT
from renom_img.server import DATASRC_IMG, DATASRC_LABEL
from renom_img.server import STATE_RUNNING, STATE_RESERVED
from renom_img.server import RUN_STATE_TRAINING, RUN_STATE_VALIDATING, \
    RUN_STATE_PREDICTING, RUN_STATE_STARTING, RUN_STATE_STOPPING, RUN_STATE_NONE
from renom_img.server import GPU_NUM

from renom_img.server.utility.storage import storage


class TrainThread(object):

    def __init__(self, model_id):
        # Thread attrs.
        self.model = None
        self.train_dist = None
        self.valid_dist = None
        self.stop_event = Event()
        self.model_id = model_id
        self.state = STATE_CREATED
        self.running_state = RUN_STATE_NONE

        self.task_id, self.dataset_id, self.algorithm_id, self.hyper_parameters = \
            storage.fetch_model_train_params(model_id)
        self.class_map, self.train_data, self.valid_data = \
            storage.fetch_train_dataset(self.dataset_id)
        self.common_params = [
            'total_epoch',
            'batch_size',
            'imsize',
            'train_whole_network',
            'load_pretrained_weight',
        ]
        self._prepare_model()
        assert self.model is not None
        self.state = STATE_RESERVED

    def __call__(self):
        pass

    def run(self):
        self.state = STATE_RUNNING
        self.running_state = RUN_STATE_STARTING
        if self.stop_event: return

    def stop(self):
        self.stop_event.set()
        self.running_state = RUN_STATE_STOPPING

    def weight_download(self, path):
        # Perform this in polling progress.
        if not os.path.exists(path):
            pass

    def _prepare_model(self):
        if self.algorithm_id == 0:
            self._setting_yolov1()
        else:
            assert False

    def _setting_yolov1(self):
        required_params = [*self.common_params, 'cell', 'box']
        assert all([k in self.hyper_parameters.keys() for k in required_params]
        assert self.task_id == 1
        self.model=Yolov1(**self.hyper_parameters)
        aug=Augmentation([
          Shift(10, 10)
        ])
        self.train_dist=ImageDistributor(
            self.train_data['img'], self.train_data['annotation'], augmentation=aug
        )
        self.valid_dist=ImageDistributor(self.valid_data['img'], self.valid_data['annotation'])

    def _setting_yolov2(self):
        pass

    def _setting_ssd(self):
        pass

    def _setting_resnet(self):
        pass
