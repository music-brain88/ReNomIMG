import os
import time
import json
import weakref
import traceback
from threading import Event, Semaphore
import numpy as np

from renom.cuda import set_cuda_active, release_mem_pool, use_device
from renom_img.api.classification.vgg import VGG16
from renom_img.api.classification.resnet import ResNet18
from renom_img.api.detection.yolo_v1 import Yolov1
from renom_img.api.detection.yolo_v2 import Yolov2, create_anchor
from renom_img.api.detection.ssd import SSD
from renom_img.api.segmentation.unet import UNet
from renom_img.api.utility.load import parse_xml_detection
from renom_img.api.utility.evaluate.detection import get_ap_and_map, get_prec_rec_iou
from renom_img.api.utility.augmentation.process import Shift, Rotate, Flip, WhiteNoise, ContrastNorm
from renom_img.api.utility.augmentation import Augmentation
from renom_img.api.utility.distributor.distributor import ImageDistributor

from renom_img.server.utility.storage import storage
from renom_img.server import State, RunningState, MAX_THREAD_NUM, Algorithm, Task


class TrainThread(object):

    jobs = weakref.WeakValueDictionary()
    semaphore = Semaphore(MAX_THREAD_NUM)

    def __new__(cls, model_id):
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

        # If any value (train_loss or ...) is changed, this will be True.
        self.updated = False
        # This will be changed from web API.
        self.best_valid_changed = False
        self.error_msg = None

    def __call__(self):
        try:
            self.state = State.RESERVED
            self.sync_state()
            self.semaphore.acquire()
            self._prepare_params()
            self._prepare_model()
            release_mem_pool()
            self.state = State.STARTED
            self.running_state = RunningState.STARTING
            assert self.model is not None
            self.sync_state()
            self.run()
        except Exception as e:
            traceback.print_exc()
            self.error_msg = e
        finally:
            release_mem_pool()
            self.semaphore.release()
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
        model = self.model
        self.state = State.STARTED
        self.running_state = RunningState.TRAINING
        valid_target = self.valid_dist.get_resized_annotation_list(self.imsize)
        if self.stop_event.is_set():
            # Watch stop event
            return

        for e in range(self.total_epoch):
            self.nth_epoch = e
            if self.stop_event.is_set():
                # Watch stop event
                return

            temp_train_batch_loss_list = []
            for b, (train_x, train_y) in enumerate(self.train_dist.batch(self.batch_size)):
                self.nth_batch = b
                if self.stop_event.is_set():
                    # Watch stop event
                    return
                self.sync_count()

                with model.train():
                    loss = model.loss(model(train_x), train_y)
                    reg_loss = loss + model.regularize()

                try:
                    loss = loss.as_ndarray()[0]
                except:
                    loss = loss.as_ndarray()
                loss = float(loss)

                temp_train_batch_loss_list.append(loss)
                self.last_batch_loss = loss
                self.sync_batch_result()

                if self.stop_event.is_set():
                    # Watch stop event
                    return

                reg_loss.grad().update(model.get_optimizer(
                    current_loss=loss,
                    current_epoch=e,
                    total_epoch=self.total_epoch,
                    current_batch=b,
                    total_batch=self.total_batch))

                # Thread value changed.
                self.updated = True

            self.train_loss_list.append(np.mean(temp_train_batch_loss_list))
            self.sync_train_loss()

            self.updated = True

            self.running_state = RunningState.VALIDATING
            self.sync_state()
            valid_prediction = []
            temp_valid_batch_loss_list = []
            for b, (valid_x, valid_y) in enumerate(self.valid_dist.batch(self.batch_size, shuffle=False)):

                if self.stop_event.is_set():
                    # Watch stop event
                    return

                valid_prediction_in_batch = model(train_x)
                loss = model.loss(valid_prediction_in_batch, train_y)
                valid_prediction.append(valid_prediction_in_batch)
                try:
                    loss = loss.as_ndarray()[0]
                except:
                    loss = loss.as_ndarray()
                loss = float(loss)
                temp_valid_batch_loss_list.append(loss)
            self.valid_loss_list.append(np.mean(temp_valid_batch_loss_list))
            self.sync_valid_loss()

            if self.stop_event.is_set():
                # Watch stop event
                return
            valid_prediction = np.concatenate(valid_prediction, axis=0)
            n_valid = min(len(valid_prediction), len(valid_target))
            if False:
                pass
            elif self.task_id == Task.DETECTION.value:
                prediction_box = model.get_bbox(valid_prediction[:n_valid])
                prec, rec, _, iou = get_prec_rec_iou(
                    prediction_box,
                    valid_target[:n_valid]
                )
                _, mAP = get_ap_and_map(prec, rec)
                if self.best_epoch_valid_result:
                    if self.best_epoch_valid_result["mAP"] <= mAP:
                        self.best_valid_changed = True
                        self.best_epoch_valid_result = {
                            "nth_epoch": e,
                            "prediction_box": prediction_box,
                            "target_box": valid_target,
                            "mAP": float(mAP),
                            "IOU": float(iou),
                            "loss": float(loss)
                        }
                else:
                    self.best_valid_changed = True
                    self.best_epoch_valid_result = {
                        "nth_epoch": e,
                        "prediction_box": prediction_box,
                        "target_box": valid_target,
                        "mAP": float(mAP),
                        "IOU": float(iou),
                        "loss": float(loss)
                    }
                self.sync_best_valid_result()

            # Thread value changed.
            self.updated = True
            break

    def stop(self):
        self.stop_event.set()
        self.running_state = RunningState.STOPPING

    def weight_download(self, path):
        # Perform this in polling progress.
        if not os.path.exists(path):
            pass

    def sync_state(self):
        storage.update_model(self.model_id, state=self.state.value,
                             running_state=self.running_state.value)

    def sync_count(self):
        storage.update_model(self.model_id,
                             total_epoch=self.total_epoch, nth_epoch=self.nth_epoch,
                             total_batch=self.total_batch, nth_batch=self.nth_batch)

    def sync_batch_result(self):
        storage.update_model(self.model_id,
                             last_batch_loss=self.last_batch_loss)

    def sync_train_loss(self):
        storage.update_model(self.model_id, train_loss_list=self.train_loss_list)

    def sync_valid_loss(self):
        storage.update_model(self.model_id, valid_loss_list=self.valid_loss_list)

    def sync_best_valid_result(self):
        storage.update_model(self.model_id,
                             best_epoch_valid_result=self.best_epoch_valid_result)

    def _prepare_params(self):
        params = storage.fetch_model(self.model_id)
        self.task_id = int(params["task_id"])
        self.dataset_id = int(params["dataset_id"])
        self.algorithm_id = int(params["algorithm_id"])
        self.hyper_parameters = params["hyper_parameters"]

        dataset = storage.fetch_dataset(self.dataset_id)
        self.class_map = dataset["class_map"]
        self.train_img = dataset["train_data"]["img"]
        self.train_target = dataset["train_data"]["target"]
        self.valid_img = dataset["valid_data"]["img"]
        self.valid_target = dataset["valid_data"]["target"]

        self.train_dist = None
        self.valid_dist = None

        n_data = len(self.train_img)

        self.common_params = [
            'total_epoch',
            'batch_size',
            'imsize_w',
            'imsize_h',
            'train_whole',
            # 'load_pretrained_weight',
        ]

        assert all([k in self.hyper_parameters.keys() for k in self.common_params])

        # Training States
        # TODO: Need getter for json decoding.
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

    def _prepare_model(self):
        if self.algorithm_id == Algorithm.YOLOV1.value:
            self._setting_yolov1()
        elif self.algorithm_id == Algorithm.YOLOV2.value:
            self._setting_yolov2()
        elif self.algorithm_id == Algorithm.SSD.value:
            self._setting_ssd()
        elif self.algorithm_id == Algorithm.RESNET18.value:
            self._setting_resnet18()
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
            train_whole_network=self.hyper_parameters["train_whole"],
            load_pretrained_weight=True)
        aug = Augmentation([
            Shift(10, 10)
        ])
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=aug,
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
            anchor=create_anchor(self.valid_target, int(self.hyper_parameters.get('anchor')), base_size=self.imsize),
            train_whole_network=self.hyper_parameters["train_whole"],
            load_pretrained_weight=True
        )
        aug = Augmentation([
            Shift(10, 10)
        ])
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=aug,
            target_builder=self.model.build_data())
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
            load_pretrained_weight=True,
            train_whole_network=self.hyper_parameters["train_whole"]
        )
        aug = Augmentation([
            Shift(10, 10)
        ])
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=aug,
            target_builder=self.model.build_data()
        )
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data()
        )

    # Classification Algorithm
    def _setting_resnet18(self):
        assert all([self.hyper_parameters.keys()])
        assert self.task_id == Task.CLASSIFICATION.value, self.task_id
        self.model = ResNet18(
            class_map=self.class_map,
            imsize=self.imsize,
            load_pretrained_weight=True,
            train_whole_network=self.hyper_parameters["train_whole"],
            plateau=self.hyper_parameters["plateau"]
        )
        aug = Augmentation([
            Shift(10, 10)
        ])
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=aug,
            target_builder=self.model.build_data()
        )
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data()
        )

    def _setting_resnet34(self):
        assert all([self.hyper_parameters.keys()])
        assert self.task_id == Task.CLASSIFICATION.value, self.task_id
        self.model = ResNet34(
            class_map=self.class_map,
            imsize=self.imsize,
            load_pretrained_weight=True,
            train_whole_network=self.hyper_parameters["train_whole"],
            plateau=self.hyper_parameters["plateau"]
        )
        aug = Augmentation([
            Shift(10, 10)
        ])
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=aug,
            target_builder=self.model.build_data()
        )
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data()
        )

    def _setting_resnet50(self):
        assert all([self.hyper_parameters.keys()])
        assert self.task_id == Task.CLASSIFICATION.value, self.task_id
        self.model = ResNet50(
            class_map=self.class_map,
            imsize=self.imsize,
            load_pretrained_weight=True,
            train_whole_network=self.hyper_parameters["train_whole"],
            plateau=self.hyper_parameters["plateau"]
        )
        aug = Augmentation([
            Shift(10, 10)
        ])
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=aug,
            target_builder=self.model.build_data()
        )
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data()
        )
    
    def _setting_resnet101(self):
        assert all([self.hyper_parameters.keys()])
        assert self.task_id == Task.CLASSIFICATION.value, self.task_id
        self.model = ResNet101(
            class_map=self.class_map,
            imsize=self.imsize,
            load_pretrained_weight=True,
            train_whole_network=self.hyper_parameters["train_whole"],
            plateau=self.hyper_parameters["plateau"]
        )
        aug = Augmentation([
            Shift(10, 10)
        ])
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=aug,
            target_builder=self.model.build_data()
        )
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data()
        )
    
    def _setting_resnet152(self):
        assert all([self.hyper_parameters.keys()])
        assert self.task_id == Task.CLASSIFICATION.value, self.task_id
        self.model = ResNet152(
            class_map=self.class_map,
            imsize=self.imsize,
            load_pretrained_weight=True,
            train_whole_network=self.hyper_parameters["train_whole"],
            plateau=self.hyper_parameters["plateau"]
        )
        aug = Augmentation([
            Shift(10, 10)
        ])
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=aug,
            target_builder=self.model.build_data()
        )
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data()
        )

    def _setting_densenet121(self):
        assert all([self.hyper_parameters.keys()])
        assert self.task_id == Task.CLASSIFICATION.value, self.task_id
        self.model = DenseNet121(
            class_map=self.class_map,
            imsize=self.imsize,
            load_pretrained_weight=True,
            train_whole_network=self.hyper_parameters["train_whole"]
        )
        aug = Augmentation([
            Shift(10, 10)
        ])
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=aug,
            target_builder=self.model.build_data()
        )
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data()
        )
    
    def _setting_densenet169(self):
        assert all([self.hyper_parameters.keys()])
        assert self.task_id == Task.CLASSIFICATION.value, self.task_id
        self.model = DenseNet169(
            class_map=self.class_map,
            imsize=self.imsize,
            load_pretrained_weight=True,
            train_whole_network=self.hyper_parameters["train_whole"]
        )
        aug = Augmentation([
            Shift(10, 10)
        ])
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=aug,
            target_builder=self.model.build_data()
        )
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data()
        )


    def _setting_densenet201(self):
        assert all([self.hyper_parameters.keys()])
        assert self.task_id == Task.CLASSIFICATION.value, self.task_id
        self.model = DenseNet169(
            class_map=self.class_map,
            imsize=self.imsize,
            load_pretrained_weight=True,
            train_whole_network=self.hyper_parameters["train_whole"]
        )
        aug = Augmentation([
            Shift(10, 10)
        ])
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=aug,
            target_builder=self.model.build_data()
        )
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data()
        )
    
    def _setting_vgg11(self):
        assert all([self.hyper_parameters.keys()])
        assert self.task_id == Task.CLASSIFICATION.value, self.task_id
        self.model = VGG11(
            class_map=self.class_map,
            imsize=self.imsize,
            load_pretrained_weight=True,
            train_whole_network=self.hyper_parameters["train_whole"]
        )
        aug = Augmentation([
            Shift(10, 10)
        ])
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=aug,
            target_builder=self.model.build_data()
        )
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data()
        )


    def _setting_vgg16(self):
        assert all([self.hyper_parameters.keys()])
        assert self.task_id == Task.CLASSIFICATION.value, self.task_id
        self.model = VGG16(
            class_map=self.class_map,
            imsize=self.imsize,
            load_pretrained_weight=True,
            train_whole_network=self.hyper_parameters["train_whole"]
        )
        aug = Augmentation([
            Shift(10, 10)
        ])
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=aug,
            target_builder=self.model.build_data()
        )
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data()
        )
    
    def _setting_vgg16_no_dense(self):
        assert all([self.hyper_parameters.keys()])
        assert self.task_id == Task.CLASSIFICATION.value, self.task_id
        self.model = VGG16_NODENSE(
            class_map=self.class_map,
            imsize=self.imsize,
            load_pretrained_weight=True,
            train_whole_network=self.hyper_parameters["train_whole"]
        )
        aug = Augmentation([
            Shift(10, 10)
        ])
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=aug,
            target_builder=self.model.build_data()
        )
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data()
        )

    def _setting_vgg19(self):
        assert all([self.hyper_parameters.keys()])
        assert self.task_id == Task.CLASSIFICATION.value, self.task_id
        self.model = VGG16(
            class_map=self.class_map,
            imsize=self.imsize,
            load_pretrained_weight=True,
            train_whole_network=self.hyper_parameters["train_whole"]
        )
        aug = Augmentation([
            Shift(10, 10)
        ])
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=aug,
            target_builder=self.model.build_data()
        )
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data()
        )

    def _inseption_v1(self):
        assert all([self.hyper_parameters.keys()])
        assert self.task_id == Task.CLASSIFICATION.value, self.task_id
        self.model = InceptionV1(
            class_map=self.class_map,
            imsize=self.imsize,
            load_pretrained_weight=True,
            train_whole_network=self.hyper_parameters["train_whole"]
        )
        aug = Augmentation([
            Shift(10, 10)
        ])
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=aug,
            target_builder=self.model.build_data()
        )
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data()
        )
    
    def _insception_v2(self):
        assert all([self.hyper_parameters.keys()])
        assert self.task_id == Task.CLASSIFICATION.value, self.task_id
        self.model = InceptionV2(
            class_map=self.class_map,
            imsize=self.imsize,
            load_pretrained_weight=True,
            train_whole_network=self.hyper_parameters["train_whole"]
        )
        aug = Augmentation([
            Shift(10, 10)
        ])
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=augm,
            target_builder=self.model.build_data()
        )
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data()
        )

    def _insception_v3(self):
        assert all([self.hyper_parameters.keys()])
        assert self.task_id == Task.CLASSIFICATION.value, self.task_id
        self.model = InceptionV3(
            class_map=self.class_map,
            imsize=self.imsize,
            load_pretrained_weight=True,
            train_whole_network=self.hyper_parameters["train_whole"]
        )
        aug = Augmentation([
            Shift(10, 10)
        ])
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=aug,
            target_builder=self.model.build_data()
        )
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data()
        )
    
    def _insception_v4(self):
        assert all([self.hyper_parameters.keys()])
        assert self.task_id == Task.CLASSIFICATION.value, self.task_id
        self.model = InceptionV4(
            class_map=self.class_map,
            imsize=self.imsize,
            load_pretrained_weight=True,
            train_whole_network=self.hyper_parameters["train_whole"]
        )
        aug = Augmentation([
            Shift(10, 10)
        ])
        self.train_dist = ImageDistributor(
            self.train_img,
            self.train_target,
            augmentation=aug,
            target_builder=self.model.build_data()
        )
        self.valid_dist = ImageDistributor(
            self.valid_img,
            self.valid_target,
            target_builder=self.model.build_data()
        )

    def _unet(self):
        assert all([self.hyper_parameters.keys()])
        assert self.task_id == Task.SEGEMENTATION.value, self.task_id
        self.model = UNet(
            class_map=self.class_map,
            imsize=self.imsize,
            load_pretrained_weight=True,
            train_whole_network=self.hyper_parameters["train_whole"]
        )

