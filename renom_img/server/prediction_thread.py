import os
import sys
import time
import json
import weakref
import traceback
from threading import Event
from PIL import Image
sys.setrecursionlimit(10000)
import numpy as np

from renom.cuda import set_cuda_active, release_mem_pool, use_device
from renom_img.api.classification.vgg import VGG11, VGG16, VGG19
from renom_img.api.classification.resnet import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from renom_img.api.classification.resnext import ResNeXt50, ResNeXt101
from renom_img.api.classification.densenet import DenseNet121, DenseNet169, DenseNet201
from renom_img.api.classification.inception import InceptionV1, InceptionV2, InceptionV3, InceptionV4
from renom_img.api.detection.yolo_v1 import Yolov1
from renom_img.api.detection.yolo_v2 import Yolov2, create_anchor
from renom_img.api.detection.ssd import SSD
from renom_img.api.segmentation.unet import UNet
from renom_img.api.segmentation.fcn import FCN8s, FCN16s, FCN32s

from renom_img.server.utility.semaphore import EventSemaphore, Semaphore
from renom_img.server.utility.storage import storage
from renom_img.server import State, RunningState, MAX_THREAD_NUM, Algorithm, Task


class PredictionThread(object):

    jobs = weakref.WeakValueDictionary()
    semaphore = EventSemaphore(MAX_THREAD_NUM)  # Cancellable semaphore.
    # semaphore = Semaphore(MAX_THREAD_NUM)

    def __new__(cls, model_id):
        ret = super(PredictionThread, cls).__new__(cls)
        cls.jobs[model_id] = ret
        return ret

    def set_future(self, future_obj):
        self.future = future_obj

    def __init__(self, model_id):
        # Thread attrs.
        set_cuda_active(True)
        self.model = None
        self.stop_event = Event()
        self.model_id = model_id
        self.state = State.PRED_CREATED
        self.running_state = RunningState.PREPARING
        self.sync_state()

        # If any value (train_loss or ...) is changed, this will be True.
        self.updated = True

        # This will be changed from web API.
        self.error_msg = None

        # Data path
        self.img_dir = os.path.join("datasrc", "prediction_set", "img")

        # Define attr
        self.total_batch = 0
        self.nth_batch = 0
        self.need_pull = False
        self.prediction_result = []

    def __call__(self):
        try:
            self.state = State.PRED_RESERVED
            self.sync_state()

            # This guarantees the state information returns immediately.
            self.updated = True

            PredictionThread.semaphore.acquire(self.stop_event)
            if self.stop_event.is_set():
                # Watch stop event
                self.updated = True
                return

            self.state = State.PRED_STARTED
            self._prepare_params()
            self._prepare_model()
            release_mem_pool()
            self.running_state = RunningState.STARTING
            assert self.model is not None
            self.sync_state()
            self.run()
        except Exception as e:
            traceback.print_exc()
            self.error_msg = e
            self.model = None
        finally:
            release_mem_pool()
            PredictionThread.semaphore.release()
            self.state = State.STOPPED
            self.running_state = RunningState.STOPPING
            self.sync_state()

    def returned2client(self):
        self.updated = False

    def consume_error(self):
        if self.error_msg is not None:
            e = self.error_msg
            self.error_msg = None
            raise e

    def run(self):
        model = self.model
        self.state = State.STARTED
        self.running_state = RunningState.TRAINING

        if self.stop_event.is_set():
            # Watch stop event
            self.updated = True
            return
        names = list(os.listdir(self.img_dir))
        N = len(names)
        results = []
        sizes = []
        imgs = []
        self.total_batch = N
        for i, p in enumerate(names):
            self.nth_batch = i
            path = os.path.join(self.img_dir, p)
            pred = self.model.predict(path)
            if isinstance(pred, np.ndarray):
                pred = pred.tolist()
            results.append(pred)
            sizes.append(Image.open(path).size)
            imgs.append(path)
            self.updated = True
        self.prediction_result = {
            "img": imgs,
            "size": sizes,
            "prediction": results,
        }
        self.need_pull = True
        self.sync_result()
        return

    def stop(self):
        self.stop_event.set()
        self.running_state = RunningState.STOPPING

    def sync_state(self):
        storage.update_model(self.model_id, state=self.state.value,
                             running_state=self.running_state.value)

    def sync_result(self):
        storage.update_model(self.model_id, last_prediction_result=self.prediction_result)

    def _prepare_params(self):
        if self.stop_event.is_set():
            # Watch stop event
            self.updated = True
            return

        params = storage.fetch_model(self.model_id)
        self.task_id = int(params["task_id"])
        self.dataset_id = int(params["dataset_id"])
        self.algorithm_id = int(params["algorithm_id"])
        self.hyper_parameters = params["hyper_parameters"]
        self.last_weight_path = params["last_weight"]
        self.best_weight_path = params["best_epoch_weight"]

        dataset = storage.fetch_dataset(self.dataset_id)
        self.class_map = dataset["class_map"]

        self.common_params = [
            'total_epoch',
            'batch_size',
            'imsize_w',
            'imsize_h',
            'train_whole',
            'load_pretrained_weight'
        ]

        assert all([k in self.hyper_parameters.keys() for k in self.common_params])

        # Training States
        # TODO: Need getter for json decoding.
        self.load_pretrained_weight = False
        self.train_whole = bool(self.hyper_parameters["train_whole"])
        self.imsize = (int(self.hyper_parameters["imsize_w"]),
                       int(self.hyper_parameters["imsize_h"]))
        self.batch_size = int(self.hyper_parameters["batch_size"])

    def _prepare_model(self):
        if self.stop_event.is_set():
            # Watch stop event
            self.updated = True
            return

        if self.algorithm_id == Algorithm.RESNET.value:
            self._setting_resnet()
        elif self.algorithm_id == Algorithm.RESNEXT.value:
            self._setting_resnext()
        elif self.algorithm_id == Algorithm.DENSENET.value:
            self._setting_densenet()
        elif self.algorithm_id == Algorithm.VGG.value:
            self._setting_vgg()
        elif self.algorithm_id == Algorithm.INCEPTION.value:
            self._setting_inception()

        elif self.algorithm_id == Algorithm.YOLOV1.value:
            self._setting_yolov1()
        elif self.algorithm_id == Algorithm.YOLOV2.value:
            self._setting_yolov2()
        elif self.algorithm_id == Algorithm.SSD.value:
            self._setting_ssd()

        elif self.algorithm_id == Algorithm.FCN.value:
            self._setting_fcn()
        else:
            assert False

    # Detection Algorithm
    def _setting_yolov1(self):
        required_params = ['cell', 'box']
        # check hyper parameters value are set
        assert all([k in self.hyper_parameters.keys() for k in required_params])
        assert self.task_id == Task.DETECTION.value, self.task_id
        self.model = Yolov1(
            class_map=self.class_map,
            imsize=self.imsize,
            train_whole_network=self.train_whole,
            load_pretrained_weight=self.load_pretrained_weight)

    def _setting_yolov2(self):
        required_params = ['anchor']
        assert all([k in self.hyper_parameters.keys() for k in required_params])
        assert self.task_id == Task.DETECTION.value, self.task_id
        self.model = Yolov2(
            class_map=self.class_map,
            imsize=self.imsize,
            train_whole_network=self.train_whole,
            load_pretrained_weight=self.load_pretrained_weight
        )

    def _setting_ssd(self):
        assert all([self.hyper_parameters.keys()])
        assert self.task_id == Task.DETECTION.value, self.task_id
        self.model = SSD(
            class_map=self.class_map,
            imsize=self.imsize,
            train_whole_network=self.train_whole,
            load_pretrained_weight=self.load_pretrained_weight,
        )

    # Classification Algorithm
    def _setting_resnet(self):
        required_params = ['layer', 'plateau']
        assert all([k in self.hyper_parameters.keys() for k in required_params])
        assert self.task_id == Task.CLASSIFICATION.value, self.task_id
        num_layer = int(self.hyper_parameters['layer'])
        assert num_layer in [18, 34, 50, 101, 152], "Not supported layer num. - ResNet"

        if num_layer == 18:
            ResNet = ResNet18
        elif num_layer == 34:
            ResNet = ResNet34
        elif num_layer == 50:
            ResNet = ResNet50
        elif num_layer == 101:
            ResNet = ResNet101
        elif num_layer == 152:
            ResNet = ResNet152

        self.model = ResNet(
            class_map=self.class_map,
            imsize=self.imsize,
            train_whole_network=self.train_whole,
            load_pretrained_weight=self.load_pretrained_weight,
            plateau=self.hyper_parameters["plateau"]
        )

    def _setting_resnext(self):
        required_params = ['layer', 'plateau']
        assert all([k in self.hyper_parameters.keys() for k in required_params])
        assert self.task_id == Task.CLASSIFICATION.value, self.task_id
        num_layer = int(self.hyper_parameters['layer'])
        assert num_layer in [50, 101], "Not supported layer num. - ResNeXt"

        if num_layer == 50:
            ResNeXt = ResNeXt50
        elif num_layer == 101:
            ResNeXt = ResNeXt101

        self.model = ResNeXt(
            class_map=self.class_map,
            imsize=self.imsize,
            train_whole_network=self.train_whole,
            load_pretrained_weight=self.load_pretrained_weight,
            plateau=self.hyper_parameters["plateau"]
        )

    def _setting_densenet(self):
        required_params = ['layer']
        assert all([k in self.hyper_parameters.keys() for k in required_params])
        assert self.task_id == Task.CLASSIFICATION.value, self.task_id
        num_layer = int(self.hyper_parameters['layer'])
        assert num_layer in [121, 169, 201], "Not supported layer num. - DenseNet"

        if num_layer == 121:
            DenseNet = DenseNet121
        elif num_layer == 169:
            DenseNet = DenseNet169
        elif num_layer == 201:
            DenseNet = DenseNet201

        self.model = DenseNet(
            class_map=self.class_map,
            imsize=self.imsize,
            load_pretrained_weight=self.load_pretrained_weight,
            train_whole_network=self.train_whole
        )

    def _setting_vgg(self):
        required_params = ['layer']
        assert all([k in self.hyper_parameters.keys() for k in required_params])
        assert self.task_id == Task.CLASSIFICATION.value, self.task_id
        num_layer = int(self.hyper_parameters['layer'])
        assert num_layer in [11, 16, 19], "Not supported layer num. - Vgg"

        if num_layer == 11:
            VGG = VGG11
        elif num_layer == 16:
            VGG = VGG16
        elif num_layer == 19:
            VGG = VGG19

        self.model = VGG(
            class_map=self.class_map,
            imsize=self.imsize,
            load_pretrained_weight=self.load_pretrained_weight,
            train_whole_network=self.train_whole
        )

    def _setting_inception(self):
        required_params = ['version']
        assert all([k in self.hyper_parameters.keys() for k in required_params])
        assert self.task_id == Task.CLASSIFICATION.value, self.task_id
        version_num = int(self.hyper_parameters['version'])
        assert version_num in [1, 2, 3, 4], "Not supported version number. - InceptionNet"

        if version_num == 1:
            Inception = InceptionV1
        elif num_layer == 2:
            Inception = InceptionV2
        elif num_layer == 3:
            Inception = InceptionV3
        elif num_layer == 4:
            Inception = InceptionV4

        self.model = Inception(
            class_map=self.class_map,
            imsize=self.imsize,
            load_pretrained_weight=self.load_pretrained_weight,
            train_whole_network=self.train_whole
        )

    def _setting_fcn(self):
        required_params = ['layer']
        assert all([k in self.hyper_parameters.keys() for k in required_params])
        assert self.task_id == Task.SEGMENTATION.value, self.task_id
        num_layer = int(self.hyper_parameters['layer'])
        assert num_layer in [8, 16, 32], "Not supported layer num. - FCN"

        if num_layer == 8:
            FCN = FCN8s
        elif num_layer == 16:
            FCN = FCN16s
        elif num_layer == 32:
            FCN = FCN32s

        self.model = FCN(
            class_map=self.class_map,
            imsize=self.imsize,
            load_pretrained_weight=self.load_pretrained_weight,
            train_whole_network=self.train_whole
        )
