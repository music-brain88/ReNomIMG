import os
import sys
import time
import json
import weakref
import traceback
from threading import Event
sys.setrecursionlimit(10000)
import numpy as np

import renom as rm
from renom.cuda import set_cuda_active, release_mem_pool, is_cuda_active, use_device
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
from renom_img.api.segmentation.deeplab import Deeplabv3plus
from renom_img.api.utility.load import parse_xml_detection
from renom_img.api.utility.evaluate.detection import get_ap_and_map, get_prec_rec_iou
from renom_img.api.utility.evaluate.classification import precision_recall_f1_score
from renom_img.api.utility.evaluate.segmentation import get_segmentation_metrics
from renom_img.api.utility.augmentation.process import Shift, Rotate, Flip, WhiteNoise, ContrastNorm, HorizontalFlip
from renom_img.api.utility.augmentation import Augmentation
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.misc.download import download
from renom_img.api.utility.optimizer import BaseOptimizer
from renom_img.api.utility.exceptions.exceptions import ReNomIMGError
from renom_img.api.observer.trainer import TrainObserverBase, ObservableTrainer

from renom_img.server.utility.semaphore import EventSemaphore, Semaphore
from renom_img.server.utility.storage import storage
from renom_img.server import State, RunningState, MAX_THREAD_NUM, Algorithm, Task, DB_DIR_PRETRAINED_WEIGHT


class AppObserver(TrainObserverBase):
    def __init__(self, ts):
        self.ts = ts

    def start(self):
        self.ts.state = State.STARTED
        self.ts.running_state = RunningState.TRAINING
        self.ts.sync_state()
        self.ts.updated = True

    def start_train_batches(self, notification):
        self.ts.running_state = RunningState.TRAINING
        self.ts.nth_epoch = notification["epoch"]
        self.ts.sync_state()
        self.ts.updated = True

    def start_batch(self, notification):
        self.ts.nth_batch = notification["batch"]
        self.ts.total_batch = notification["total_batch"]
        self.ts.updated = True

    def end_batch(self, notification):
        self.ts.last_batch_loss = notification["loss"]
        self.ts.sync_batch_result()
        self.ts.updated = True

    def end_train_batches(self, notification):
        self.ts.train_loss_list.append(notification["avg_train_loss"])
        self.ts.sync_train_loss()
        self.ts.updated = True

    def start_valid_batches(self):
        self.ts.running_state = RunningState.VALIDATING
        self.ts.sync_state()
        self.ts.updated = True

    def end_valid_batches(self, notification):
        self.ts.valid_loss_list.append(notification["avg_valid_loss"])
        self.ts.sync_valid_loss()
        self.ts.updated = True

    def start_evaluate(self):
        self.ts.best_valid_changed = False

    def end_evaluate(self, notification):
        loss = self.ts.valid_loss_list[-1]
        if self.ts.task_id == Task.CLASSIFICATION.value:
            if self.ts.best_epoch_valid_result:
                if self.ts.best_epoch_valid_result["f1"] <= notification["evaluation_matrix"]["f1"]:
                    self.ts.best_valid_changed = True
                    self.ts.save_best_model()
                    self.ts.best_epoch_valid_result = {
                        "nth_epoch": self.ts.nth_epoch + 1,
                        "prediction": notification["prediction"],
                        "recall": float(notification["evaluation_matrix"]["recall"]),
                        "precision": float(notification["evaluation_matrix"]["precision"]),
                        "f1": float(notification["evaluation_matrix"]["f1"]),
                        "loss": float(loss)
                    }
                    self.ts.sync_best_valid_result()
                    self.ts.updated = True
            else:
                self.ts.best_valid_changed = True
                self.ts.save_best_model()
                self.ts.best_epoch_valid_result = {
                    "nth_epoch": self.ts.nth_epoch + 1,
                    "prediction": notification["prediction"],
                    "recall": float(notification["evaluation_matrix"]["recall"]),
                    "precision": float(notification["evaluation_matrix"]["precision"]),
                    "f1": float(notification["evaluation_matrix"]["f1"]),
                    "loss": float(loss)
                }
                self.ts.sync_best_valid_result()
                self.ts.updated = True

        elif self.ts.task_id == Task.DETECTION.value:
            if self.ts.best_epoch_valid_result:
                if self.ts.best_epoch_valid_result["mAP"] <= notification["evaluation_matrix"]["mAP"]:
                    self.ts.best_valid_changed = True
                    self.ts.save_best_model()
                    self.ts.best_epoch_valid_result = {
                        "nth_epoch": self.ts.nth_epoch + 1,
                        "prediction": notification["prediction"],
                        "mAP": float(notification["evaluation_matrix"]["mAP"]),
                        "IOU": float(notification["evaluation_matrix"]["iou"]),
                        "loss": float(loss)
                    }
                    self.ts.sync_best_valid_result()
                    self.ts.updated = True
            else:
                self.ts.best_valid_changed = True
                self.ts.save_best_model()
                self.ts.best_epoch_valid_result = {
                    "nth_epoch": self.ts.nth_epoch + 1,
                    "prediction": notification["prediction"],
                    "mAP": float(notification["evaluation_matrix"]["mAP"]),
                    "IOU": float(notification["evaluation_matrix"]["iou"]),
                    "loss": float(loss)
                }
                self.ts.sync_best_valid_result()
                self.ts.updated = True

        elif self.ts.task_id == Task.SEGMENTATION.value:
            if self.ts.best_epoch_valid_result:
                if self.ts.best_epoch_valid_result["f1"] <= notification["evaluation_matrix"]["f1"]:
                    self.ts.best_valid_changed = True
                    self.ts.save_best_model()
                    self.ts.best_epoch_valid_result = {
                        "nth_epoch": self.ts.nth_epoch + 1,
                        "prediction": notification["prediction"],
                        "recall": float(notification["evaluation_matrix"]["recall"]),
                        "precision": float(notification["evaluation_matrix"]["precision"]),
                        "f1": float(notification["evaluation_matrix"]["f1"]),
                        "loss": float(loss)
                    }
                    self.ts.sync_best_valid_result()
                    self.ts.updated = True
            else:
                self.ts.best_valid_changed = True
                self.ts.save_best_model()
                self.ts.best_epoch_valid_result = {
                    "nth_epoch": self.ts.nth_epoch + 1,
                    "prediction": notification["prediction"],
                    "recall": float(notification["evaluation_matrix"]["recall"]),
                    "precision": float(notification["evaluation_matrix"]["precision"]),
                    "f1": float(notification["evaluation_matrix"]["f1"]),
                    "loss": float(loss)
                }
                self.ts.sync_best_valid_result()
                self.ts.updated = True

    def end(self, train_result):
        self.ts.state = State.STOPPED
        self.ts.sync_state()
        self.ts.updated = True
        if is_cuda_active():
            release_mem_pool()


class TrainThread(object):

    jobs = weakref.WeakValueDictionary()
    semaphore = EventSemaphore(MAX_THREAD_NUM)  # Cancellable semaphore.
    # semaphore = Semaphore(MAX_THREAD_NUM)

    def __new__(cls, model_id):
        ret = super(TrainThread, cls).__new__(cls)
        cls.jobs[model_id] = ret
        return ret

    @classmethod
    def add_thread(cls, model_id):
        ret = super(TrainThread, cls).__new__(cls)
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
        self.state = State.CREATED
        self.running_state = RunningState.PREPARING
        self.sync_state()

        # If any value (train_loss or ...) is changed, this will be True.
        self.updated = True

        # This will be changed from web API.
        self.best_epoch_valid_result = {}
        self.best_valid_changed = False
        self.error_msg = None

    def __call__(self):
        try:
            self.state = State.RESERVED
            self.sync_state()

            # This guarantees the state information returns immediately.
            self.updated = True

            TrainThread.semaphore.acquire(self.stop_event)
            if self.stop_event.is_set():
                # Watch stop event
                self.updated = True
                return

            self.state = State.STARTED
            self._prepare_params()
            self._prepare_model()
            release_mem_pool()
            self.running_state = RunningState.STARTING
            assert self.model is not None
            self.sync_state()
            self.run()
        except ReNomIMGError as e:
            traceback.print_exc()
            self.state = State.STOPPED
            self.running_state = RunningState.STOPPING
            self.error_msg = e
            self.model = None
            self.sync_state()
        except Exception as e:
            traceback.print_exc()
            self.state = State.STOPPED
            self.running_state = RunningState.STOPPING
            self.error_msg = UnknownError()
            self.model = None
            self.sync_state()
        finally:
            release_mem_pool()
            TrainThread.semaphore.release()
            self.state = State.STOPPED
            self.running_state = RunningState.STOPPING
            self.sync_state()

    def returned2client(self):
        self.updated = False

    def returned_best_result2client(self):
        self.best_valid_changed = False

    def consume_error(self):
        if self.error_msg is not None:
            e = self.error_msg
            self.error_msg = None
            raise e

    def run(self):
        observer = AppObserver(self)
        self.trainer = ObservableTrainer(self.model)
        self.trainer.add_observer(observer)
        self.trainer.train(self.train_dist, self.valid_dist, self.total_epoch, self.batch_size)

    def stop(self):
        self.stop_event.set()
        self.trainer.stop()
        self.running_state = RunningState.STOPPING
        self.sync_state()

    def save_best_model(self):
        self.model.save(self.best_weight_path)

    def save_last_model(self):
        self.model.save(self.last_weight_path)

    def sync_state(self):
        storage.update_model(self.model_id, state=self.state.value,
                             running_state=self.running_state.value)

    def sync_batch_result(self):
        storage.update_model(self.model_id, total_epoch=self.total_epoch, nth_epoch=self.nth_epoch,
                             total_batch=self.total_batch, nth_batch=self.nth_batch,
                             last_batch_loss=self.last_batch_loss)

    def sync_train_loss(self):
        storage.update_model(self.model_id, train_loss_list=self.train_loss_list)

    def sync_valid_loss(self):
        storage.update_model(self.model_id, valid_loss_list=self.valid_loss_list)

    def sync_best_valid_result(self):
        storage.update_model(self.model_id,
                             best_epoch_valid_result=self.best_epoch_valid_result)

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
        self.train_img = dataset["train_data"]["img"]
        self.train_target = dataset["train_data"]["target"]
        self.valid_img = dataset["valid_data"]["img"]
        self.valid_target = dataset["valid_data"]["target"]

        for path in self.train_img + self.valid_img:
            if not os.path.exists(path):
                raise FileNotFoundError("The image file {} is not found.".format(path))

        self.train_dist = None
        self.valid_dist = None

        n_data = len(self.train_img)

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
        self.load_pretrained_weight = bool(self.hyper_parameters["load_pretrained_weight"])
        self.train_whole = bool(self.hyper_parameters["train_whole"])
        self.imsize = (int(self.hyper_parameters["imsize_w"]),
                       int(self.hyper_parameters["imsize_h"]))
        self.batch_size = int(self.hyper_parameters["batch_size"])
        self.total_epoch = int(self.hyper_parameters["total_epoch"])
        self.nth_epoch = 0
        self.total_batch = int(np.ceil(n_data / self.batch_size))
        self.nth_batch = 0
        self.last_batch_loss = 0
        self.train_loss_list = []
        self.valid_loss_list = []
        self.best_epoch_valid_result = {}

        # Augmentation Setting.
        if self.task_id == Task.SEGMENTATION.value:
            self.augmentation = Augmentation([
                HorizontalFlip()
                #Rotate(),
                #Flip(),
                #ContrastNorm(),
            ])
        else:
            self.augmentation = Augmentation([
                Shift(10, 10),
                Rotate(),
                Flip(),
                ContrastNorm(),
            ])

    def _prepare_model(self):
        if self.stop_event.is_set():
            # Watch stop event
            self.updated = True
            return
        self.running_state = RunningState.WEIGHT_DOWNLOADING
        self.sync_state()
        self.updated = True

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
        elif self.algorithm_id == Algorithm.DEEPLABV3PLUS.value:
            self._setting_deeplabv3plus()
        elif self.algorithm_id == Algorithm.UNET.value:
            self._setting_unet()
        else:
            assert False

        self.running_state = RunningState.STARTING
        self.sync_state()
        self.updated = True

    def get_weight_path(self, cls):
        """This function returns pretrained weight path or False value.
        This modifies weight file path.
        """
        if self.load_pretrained_weight:
            return str(DB_DIR_PRETRAINED_WEIGHT / "{}.h5".format(cls.__name__))
        else:
            return False

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
            load_pretrained_weight=self.get_weight_path(Yolov1))
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=self.augmentation,
            target_builder=self.model.build_data())
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data())

    def _setting_yolov2(self):
        required_params = ['anchor']
        assert all([k in self.hyper_parameters.keys() for k in required_params])
        assert self.task_id == Task.DETECTION.value, self.task_id
        self.model = Yolov2(
            class_map=self.class_map,
            imsize=self.imsize,
            anchor=create_anchor(self.train_target, int(
                self.hyper_parameters.get('anchor')), base_size=self.imsize),
            train_whole_network=self.train_whole,
            load_pretrained_weight=self.get_weight_path(Yolov2))
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=self.augmentation,
            target_builder=self.model.build_data(
                imsize_list=[(i * 32, i * 32) for i in range(9, 14)]))
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data())

    def _setting_ssd(self):
        assert all([self.hyper_parameters.keys()])
        assert self.task_id == Task.DETECTION.value, self.task_id
        self.model = SSD(
            class_map=self.class_map,
            imsize=self.imsize,
            train_whole_network=self.train_whole,
            load_pretrained_weight=self.get_weight_path(SSD))
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=self.augmentation,
            target_builder=self.model.build_data()
        )
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data()
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
            load_pretrained_weight=self.get_weight_path(ResNet),
            plateau=self.hyper_parameters["plateau"]
        )
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=self.augmentation,
            target_builder=self.model.build_data()
        )
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data()
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
            load_pretrained_weight=self.get_weight_path(ResNeXt),
            plateau=self.hyper_parameters["plateau"]
        )
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=self.augmentation,
            target_builder=self.model.build_data()
        )
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data()
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
            load_pretrained_weight=self.get_weight_path(DenseNet),
            train_whole_network=self.train_whole
        )

        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=self.augmentation,
            target_builder=self.model.build_data()
        )
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data()
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
            load_pretrained_weight=self.get_weight_path(VGG),
            train_whole_network=self.train_whole
        )
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=self.augmentation,
            target_builder=self.model.build_data()
        )
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data()
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
            load_pretrained_weight=self.get_weight_path(Inception),
            train_whole_network=self.train_whole
        )
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=self.augmentation,
            target_builder=self.model.build_data()
        )
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data()
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
            load_pretrained_weight=self.get_weight_path(FCN),
            train_whole_network=self.train_whole
        )
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=self.augmentation,
            target_builder=self.model.build_data()
        )
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data()
        )

    def _setting_deeplabv3plus(self):
        assert self.task_id == Task.SEGMENTATION.value, self.task_id
        self.model = Deeplabv3plus(
            class_map=self.class_map,
            imsize=self.imsize,
            load_pretrained_weight=self.get_weight_path(Deeplabv3plus),
            train_whole_network=self.train_whole
        )
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=self.augmentation,
            target_builder=self.model.build_data()
        )
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data()
        )

    def _setting_unet(self):
        assert self.task_id == Task.SEGMENTATION.value, self.task_id
        self.model = UNet(
            class_map=self.class_map,
            imsize=self.imsize,
            load_pretrained_weight=self.get_weight_path(UNet),
            train_whole_network=self.train_whole
        )
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=self.augmentation,
            target_builder=self.model.build_data()
        )
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data()
        )
