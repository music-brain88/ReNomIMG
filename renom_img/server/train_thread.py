import os
import weakref
import time
import json
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
from renom_img.server.utility.storage import storage
from renom_img.server import State, RunningState


class TrainThread(object):

    jobs = weakref.WeakValueDictionary()

    def __new__(cls, model_id):
        ret = super(TrainThread, cls).__new__(cls)
        cls.jobs[model_id] = ret
        return ret

    def __init__(self, model_id):
        # Thread attrs.
        set_cuda_active(True)
        self.model = None
        self.train_dist = None
        self.valid_dist = None
        self.stop_event = Event()
        self.model_id = model_id
        self.state = State.CREATED
        self.running_state = RunningState.PREPARING

        params = storage.fetch_model(model_id)
        self.task_id = params["task_id"]
        self.dataset_id = params["dataset_id"]
        self.algorithm_id = params["algorithm_id"]
        self.hyper_parameters = params["hyper_parameters"]

        dataset = storage.fetch_dataset(self.dataset_id)
        self.class_map = dataset["class_map"]
        self.train_img = dataset["train_data"]["img"]
        self.train_target = dataset["train_data"]["target"]
        self.valid_img = dataset["valid_data"]["img"]
        self.valid_target = dataset["valid_data"]["target"]

        self.common_params = [
            'total_epoch',
            'batch_size',
            'imsize_w',
            'imsize_h',
            'train_whole',
            # 'load_pretrained_weight',
        ]
        self._prepare_model()
        assert self.model is not None
        self.state = State.RESERVED

    def __call__(self):
        pass

    def run(self):
        self.model.fit(self.train_img, self.train_target, self.valid_img, self.valid_target)
        self.state = State.TRAINING
        self.running_state = RunningState.STARTING
        if self.stop_event:
            return

    def stop(self):
        self.stop_event.set()
        self.running_state = RunningState.STOPPING

    def weight_download(self, path):
        # Perform this in polling progress.
        if not os.path.exists(path):
            pass

    def _prepare_model(self):
        if self.algorithm_id == 10:
            self._setting_yolov1()
        else:
            assert False

    def _setting_yolov1(self):
        required_params = [*self.common_params, 'cell', 'box']
        assert all([k in self.hyper_parameters.keys() for k in required_params])
        assert self.task_id == 1, self.task_id
        self.model = Yolov1(
            class_map=self.class_map,
            imsize=(self.hyper_parameters["imsize_w"], self.hyper_parameters["imsize_h"]),
            train_whole_network=self.hyper_parameters["train_whole"],
            load_pretrained_weight=True
        )

    def _setting_yolov2(self):
        pass

    def _setting_ssd(self):
        pass

    def _setting_resnet(self):
        pass
