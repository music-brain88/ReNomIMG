import os
import time
import numpy as np
import traceback
from threading import Event

from renom.cuda import set_cuda_active, release_mem_pool

from renom_img.api.detection.yolo_v1 import Yolov1
from renom_img.api.utility.load import parse_xml_detection
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.augmentation.process import Shift, Rotate, Flip, WhiteNoise
from renom_img.api.utility.augmentation.augmentation import Augmentation

from renom_img.server import ALG_YOLOV1
from renom_img.server import DB_DIR_TRAINED_WEIGHT
from renom_img.server import DATASRC_IMG, DATASRC_LABEL
from renom_img.server import STATE_RUNNING
from renom_img.server import RUN_STATE_TRAINING, RUN_STATE_VALIDATING, \
    RUN_STATE_PREDICTING, RUN_STATE_STARTING, RUN_STATE_STOPPING

from renom_img.server.utility.storage import storage


class TrainThread(object):

    def __init__(self, thread_id, project_id, model_id, dataset_id, hyper_parameters,
                 algorithm, algorithm_params):

        self.model_id = model_id

        # State of thread.
        # The variable _running_state has setter and getter.
        self._running_state = RUN_STATE_STARTING
        self.nth_batch = 0
        self.total_batch = 0
        self.last_batch_loss = 0
        self.nth_epoch = 0
        self.train_loss_list = []
        self.valid_loss_list = []
        self.valid_iou_list = []
        self.valid_map_list = []
        self.valid_predict_box = []

        # Error message caused in thread.
        self.error_msg = None

        # Train hyperparameters
        self.total_epoch = int(hyper_parameters["total_epoch"])
        self.batch_size = int(hyper_parameters["batch_size"])
        self.imsize = (int(hyper_parameters["image_width"]),
                       int(hyper_parameters["image_height"]))
        self.stop_event = Event()

        # Prepare dataset
        rec = storage.fetch_dataset_def(dataset_id)
        (_, name, ratio, train_files, valid_files, class_map, _, _) = rec
        self.train_dist = self.create_dist(train_files)
        self.valid_dist = self.create_dist(valid_files, False)

        # Algorithm
        # Pretrained weights are must be prepared.
        self.algorithm = algorithm
        if algorithm == ALG_YOLOV1:
            cell_size = int(algorithm_params["cells"])
            num_bbox = int(algorithm_params["bounding_box"])
            self.model = Yolov1(len(class_map), cell_size, num_bbox,
                                imsize=self.imsize, load_weight=True)
        else:
            self.error_msg = "{} is not supported algorithm id.".format(algorithm)

    @property
    def running_state(self):
        return self._running_state

    @running_state.setter
    def running_state(self, state):
        """
        If thread's state becomes RUN_STATE_STOPPING once, 
        state will never be changed.
        """
        if self._running_state != RUN_STATE_STOPPING:
            self._running_state = state

    def __call__(self):

        storage.update_model_state(self.model_id, STATE_RUNNING)
        # This func works as thread.
        epoch = self.total_epoch
        batch_size = self.batch_size
        filename = '{}.h5'.format(int(time.time()))
        best_valid_loss = np.Inf

        try:
            i = 0
            set_cuda_active(True)
            release_mem_pool()
            for e in range(epoch):

                epoch_id = storage.register_epoch(
                    model_id=self.model_id,
                    nth_epoch=e
                )

                # Train
                self.nth_epoch = e
                self.running_state = RUN_STATE_TRAINING
                if self.is_stopped():
                    return
                display_loss = 0
                batch_gen = self.train_dist.batch(batch_size, self.model.build_data)
                self.total_batch = int(np.ceil(len(self.train_dist) // batch_size))
                for i, (train_x, train_y) in enumerate(batch_gen):
                    self.nth_batch = i
                    if self.is_stopped():
                        return
                    self.model.set_models(inference=False)
                    with self.model.train():
                        loss = self.model.loss(self.model(train_x), train_y)
                        reg_loss = loss + self.model.regularize()
                    reg_loss.grad().update(self.model.get_optimizer(e, epoch,
                                                                    i, self.total_batch))
                    display_loss += float(loss.as_ndarray()[0])
                    self.last_batch_loss = float(loss.as_ndarray()[0])
                avg_train_loss = display_loss / (i + 1)

                # Validation
                self.running_state = RUN_STATE_VALIDATING
                if self.is_stopped():
                    return
                valid_predict_box = []
                display_loss = 0
                batch_gen = self.valid_dist.batch(batch_size, self.model.build_data, shuffle=False)
                self.model.set_models(inference=True)
                for i, (valid_x, valid_y) in enumerate(batch_gen):
                    if self.is_stopped():
                        return
                    valid_predict_box.extend(self.model.predict(valid_x))
                    loss = self.model.loss(self.model(valid_x), valid_y)
                    display_loss += float(loss.as_ndarray()[0])
                avg_valid_loss = display_loss / (i + 1)

                self.valid_predict_box.append(valid_predict_box)
                self.train_loss_list.append(avg_train_loss)
                self.valid_loss_list.append(avg_valid_loss)
                self.valid_iou_list.append(1)
                self.valid_map_list.append(1)

                # Store epoch data tp DB.
                storage.update_model_loss_list(
                    model_id=self.model_id,
                    train_loss_list=self.train_loss_list,
                    validation_loss_list=self.valid_loss_list,
                )
                if best_valid_loss > avg_valid_loss:
                    # modelのweightを保存する
                    self.model.save(os.path.join(DB_DIR_TRAINED_WEIGHT, filename))
                    storage.update_model_best_epoch(self.model_id, e, 1,
                                                    1, filename, self.valid_predict_box)

                storage.update_epoch(
                    epoch_id=epoch_id,
                    train_loss=avg_train_loss,
                    validation_loss=avg_valid_loss,
                    epoch_iou=1,
                    epoch_map=1)

        except Exception as e:
            traceback.print_exc()
            self.error_msg = str(e)
            self.model = None
            release_mem_pool()

    def get_running_info():
        return {
            "model_id": self.model_id,
            "total_batch": self.total_batch,
            "nth_batch": self.nth_batch,
            "last_batch_loss": self.last_batch_loss,
            "running_state": self.running_state,
        }

    def stop(self):
        # Thread can be canceled only if it have not been started.
        # This method is for stopping running thread.
        self.stop_event.set()
        self.running_state = RUN_STATE_STOPPING

    def is_stopped(self):
        return self.stop_event.is_set()

    def create_dist(self, filename_list, train=True):
        """
        This function creates img path list and annotation list from
        filename list.

        Image file name and label file must be same.
        Because of that, data_list is a list of file names.

        Data formats are bellow.  

        image path list: [path_to_img1, path_to_img2, ...]
        annotation list: [
                            [ # Annotations of each image.
                                {"box":[x, y, w, h], "name":"dog", "class":1},
                                {"box":[x, y, w, h], "name":"cat", "class":0},
                            ],
                            [
                                {"box":[x, y, w, h], "name":"cat", "class":0},
                            ],
                            ...
                          ]

        Args:
            filename_list(list): [filename1, filename2, ...]
            train(bool): If it's ture, augmentation will be added to distributor.

        Returns:
            (ImageDistributor): ImageDistributor object with augmentation.
        """
        img_path_list = []
        label_path_list = []
        for path in sorted(filename_list):
            name = os.path.splitext(path)[0]
            img_path = os.path.join(DATASRC_IMG, path)
            label_path = os.path.join(DATASRC_LABEL, name + ".xml")

            if os.path.exists(img_path) and os.path.exists(label_path):
                img_path_list.append(img_path)
                label_path_list.append(label_path)
            else:
                print("{} not found.".format(name))
        annotation_list, _ = parse_xml_detection(label_path_list)
        augmentation = Augmentation([
            Shift(40, 40),
            Flip(),
            Rotate(),
            WhiteNoise()
        ])
        return ImageDistributor(img_path_list, annotation_list,
                                augmentation=augmentation)
