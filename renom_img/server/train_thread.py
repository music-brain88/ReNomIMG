import os
import time
import numpy as np
import traceback
from threading import Event
import urllib.request

from renom.cuda import set_cuda_active, release_mem_pool

from renom_img.api.detection.yolo_v1 import Yolov1
from renom_img.api.detection.yolo_v2 import Yolov2, create_anchor
from renom_img.api.utility.load import parse_xml_detection
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.augmentation.process import Shift, Rotate, Flip, WhiteNoise, ContrastNorm
from renom_img.api.utility.augmentation import Augmentation

from renom_img.api.utility.evaluate.detection import get_ap_and_map, get_prec_rec_iou

from renom_img.server import ALG_YOLOV1, ALG_YOLOV2
from renom_img.server import WEIGHT_EXISTS, WEIGHT_CHECKING, WEIGHT_DOWNLOADING
from renom_img.server import DB_DIR_TRAINED_WEIGHT, DB_DIR_PRETRAINED_WEIGHT
from renom_img.server import DATASRC_IMG, DATASRC_LABEL
from renom_img.server import STATE_RUNNING
from renom_img.server import RUN_STATE_TRAINING, RUN_STATE_VALIDATING, \
    RUN_STATE_PREDICTING, RUN_STATE_STARTING, RUN_STATE_STOPPING

from renom_img.server.utility.storage import storage


class TrainThread(object):

    def __init__(self, thread_id, project_id, model_id, dataset_id, hyper_parameters,
                 algorithm, algorithm_params):

        # Model will be created in __call__ function.
        self.model = None
        self.model_id = model_id

        # For weight download
        self.percentage = 0
        self.weight_existance = WEIGHT_CHECKING

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
        self.train_whole_network = bool(hyper_parameters["train_whole_network"])
        self.total_epoch = int(hyper_parameters["total_epoch"])
        self.batch_size = int(hyper_parameters["batch_size"])
        self.imsize = (int(hyper_parameters["image_width"]),
                       int(hyper_parameters["image_height"]))
        self.algorithm = algorithm
        self.algorithm_params = algorithm_params

        self.stop_event = Event()

        # Prepare dataset
        rec = storage.fetch_dataset_def(dataset_id)
        (_, name, ratio, train_files, valid_files, class_map, _, _) = rec
        self.class_map = class_map
        self.train_dist = self.create_dist(train_files)
        self.valid_dist = self.create_dist(valid_files, False)

    def download_weight(self, url, filename):

        pretrained_weight_path = os.path.join(DB_DIR_PRETRAINED_WEIGHT, filename)
        if os.path.exists(pretrained_weight_path):
            self.weight_existance = WEIGHT_EXISTS
            return pretrained_weight_path

        self.weight_existance = WEIGHT_DOWNLOADING

        def progress(block_count, block_size, total_size):
            self.percentage = 100.0 * block_count * block_size / total_size
        urllib.request.urlretrieve(url, pretrained_weight_path, progress)
        self.weight_existance = WEIGHT_EXISTS

        return pretrained_weight_path

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
        # This func works as thread.
        try:
            # Algorithm and model preparation.
            # Pretrained weights are must be prepared.
            # This have to be done in thread.
            if self.algorithm == ALG_YOLOV1:
                cell_size = int(self.algorithm_params["cells"])
                num_bbox = int(self.algorithm_params["bounding_box"])
                path = self.download_weight(Yolov1.WEIGHT_URL, Yolov1.__name__ + '.h5')
                self.model = Yolov1(self.class_map, cell_size, num_bbox,
                                    imsize=self.imsize, load_pretrained_weight=path, train_whole_network=self.train_whole_network)
            elif self.algorithm == ALG_YOLOV2:
                anchor = int(self.algorithm_params["anchor"])
                path = self.download_weight(Yolov2.WEIGHT_URL, Yolov2.__name__ + '.h5')
                annotations = self.train_dist.annotation_list
                self.model = Yolov2(self.class_map, create_anchor(annotations, anchor, base_size=self.imsize),
                                    imsize=self.imsize, load_pretrained_weight=path, train_whole_network=self.train_whole_network)
            else:
                self.error_msg = "{} is not supported algorithm id.".format(self.algorithm)

            i = 0
            set_cuda_active(True)
            release_mem_pool()
            filename = '{}.h5'.format(int(time.time()))

            epoch = self.total_epoch
            batch_size = self.batch_size
            best_valid_loss = np.Inf
            valid_annotation_list = self.valid_dist.get_resized_annotation_list(self.imsize)
            storage.update_model_state(self.model_id, STATE_RUNNING)
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
                    try:
                        loss = loss.as_ndarray()[0]
                    except:
                        loss = loss.as_ndarray()

                    display_loss += float(loss)
                    self.last_batch_loss = float(loss)
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
                    valid_z = self.model(valid_x)
                    valid_predict_box.extend(self.model.get_bbox(valid_z))
                    loss = self.model.loss(valid_z, valid_y)

                    try:
                        loss = loss.as_ndarray()[0]
                    except:
                        loss = loss.as_ndarray()
                    display_loss += float(loss)

                if self.is_stopped():
                    return
                prec, recl, _, iou = get_prec_rec_iou(valid_predict_box, valid_annotation_list)
                _, mAP = get_ap_and_map(prec, recl)

                mAP = float(0 if np.isnan(mAP) else mAP)
                iou = float(0 if np.isnan(iou) else iou)

                if self.is_stopped():
                    return

                avg_valid_loss = display_loss / (i + 1)

                self.valid_predict_box.append(valid_predict_box)
                self.train_loss_list.append(avg_train_loss)
                self.valid_loss_list.append(avg_valid_loss)
                self.valid_iou_list.append(iou)
                self.valid_map_list.append(mAP)

                # Store epoch data tp DB.
                storage.update_model_loss_list(
                    model_id=self.model_id,
                    train_loss_list=self.train_loss_list,
                    validation_loss_list=self.valid_loss_list,
                )
                if best_valid_loss > avg_valid_loss:
                    # modelのweightを保存する
                    self.model.save(os.path.join(DB_DIR_TRAINED_WEIGHT, filename))
                    storage.update_model_best_epoch(self.model_id, e, iou,
                                                    mAP, filename, valid_predict_box)

                storage.update_epoch(
                    epoch_id=epoch_id,
                    train_loss=avg_train_loss,
                    validation_loss=avg_valid_loss,
                    epoch_iou=iou,
                    epoch_map=mAP)

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
        for path in filename_list:
            name = os.path.splitext(path)[0]
            img_path = os.path.join(DATASRC_IMG, path)
            label_path = os.path.join(DATASRC_LABEL, name + ".xml")

            if os.path.exists(img_path) and os.path.exists(label_path):
                img_path_list.append(img_path)
                label_path_list.append(label_path)
            else:
                print("{} not found.".format(name))
        annotation_list, _ = parse_xml_detection(label_path_list)
        if train:
            augmentation = Augmentation([
                Shift(min(self.imsize[0] // 10, 20), min(self.imsize[1] // 10, 20)),
                Flip(),
                Rotate(),
                WhiteNoise(),
                ContrastNorm([0.5, 1.0])
            ])
            return ImageDistributor(img_path_list, annotation_list,
                                    augmentation=augmentation)
        else:
            return ImageDistributor(img_path_list, annotation_list)
