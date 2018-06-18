import os
import numpy as np
import traceback
from threading import Event
from renom.cuda import set_cuda_active, release_mem_pool

from renom_img.api.detection.yolo_v1 import Yolov1
from renom_img.api.utility.load import parse_xml_detection
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.augmentation.process import Shift
from renom_img.api.utility.augmentation.augmentation import Augmentation

from renom_img.server import ALG_YOLOV1
from renom_img.server import DB_DIR_TRAINED_WEIGHT
from renom_img.server import DATASRC_IMG, DATASRC_LABEL
from renom_img.server import STATE_RUNNING
from renom_img.server import RUN_STATE_TRAINING, RUN_STATE_VALIDATING, \
    RUN_STATE_PREDICTING, RUN_STATE_STARTING, RUN_STATE_STOPPING

from renom_img.server.utility.storage import storage


class PredictionThread(object):

    def __init__(self, thread_id, project_id, model_id, hyper_parameters,
                 algorithm, algorithm_params):

        self.model_id = model_id

        # State of thread.
        # The variable _running_state has setter and getter.
        self._running_state = RUN_STATE_STARTING
        self.nth_batch = 0
        self.total_batch = 0
        self.last_batch_loss = 0

        # Error message caused in thread.
        self.error_msg = None

        # hyperparameters
        self.batch_size = int(hyper_parameters["batch_size"])
        self.imsize = (int(hyper_parameters["image_width"]),
                       int(hyper_parameters["image_height"]))
        self.stop_event = Event()

        # Prepare dataset
        self.prediction_dist = self.create_dist(train_files)

        # Algorithm
        # Pretrained weights are must be prepared.
        self.algorithm = algorithm
        if algorithm == ALG_YOLOV1:
            cell_size = int(algorithm_params["cells"])
            num_bbox = int(algorithm_params["bounding_box"])
            self.model = Yolov1(len(class_map), cell_size, num_bbox, load_weight=True)
        else:
            self.error_msg = "{} is not supported algorithm id.".format(algorithm)

        self.predict_results = {}


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
        batch_size = self.batch_size

        try:
            i = 0
            set_cuda_active(True)
            release_mem_pool()
            self.model.set_models(inference=True)
            result = []
            # Prediction
            self.running_state = RUN_STATE_VALIDATING
            if self.is_stopped():
                return
            display_loss = 0
            batch_gen = self.prediction_dist.batch(batch_size, self.model.build_data)
            for i, pred_x in enumerate(batch_gen):
                if self.is_stopped():
                    return
                result.extend(self.model.predict(pred_x))
            ## Set result.
            

        # Store epoch data tp DB.
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
            Shift(40, 40)
        ])
        return ImageDistributor(img_path_list, annotation_list,
                                augmentation=augmentation)

    def save_predict_result_to_csv(self):
        try:
            # modelのcsvを保存する
            if not os.path.isdir(CSV_DIR):
                os.makedirs(CSV_DIR)

            self.csv_filename = '{}.csv'.format(int(time.time()))
            filepath = os.path.join(CSV_DIR, self.csv_filename)

            with open(filepath, 'w') as f:
                writer = csv.writer(f, lineterminator="\n")

                for i in range(len(self.predict_results["bbox_path_list"])):
                    row = []
                    row.append(self.predict_results["bbox_path_list"][i])
                    for j in range(len(self.predict_results["bbox_list"][i])):
                        b = self.predict_results["bbox_list"][i][j]
                        row.append(b["class"])
                        row.append(b["box"])

                    writer.writerow(row)
        except Exception as e:
            traceback.print_exc()
            self.error_msg = e.args[0]

    def save_predict_result_to_xml(self):
        try:
            for i in range(len(self.predict_results["bbox_path_list"])):
                img_path = self.predict_results["bbox_path_list"][i]
                filename = img_path.split("/")[-1]
                xml_filename = '{}.xml'.format(filename.split(".")[0])
                filepath = os.path.join(XML_DIR, xml_filename)
                img_path = os.path.join(IMG_DIR, filename)
                im = Image.open(img_path)
                width, height = im.size

                annotation = et.Element("annotation")
                name = et.SubElement(annotation, "filename")
                name.text = filename

                for j in range(len(self.predict_results["bbox_list"][i])):
                    b = self.predict_results["bbox_list"][i][j]
                    obj = et.SubElement(annotation, "object")
                    bbox_class = et.SubElement(obj, "class")
                    bbox_class.text = str(b["class"])

                    bbox_position = et.SubElement(obj, "bndbox")
                    bbox_xmax = et.SubElement(bbox_position, "xmax")
                    xmax = width * b["box"][0] + (width * b["box"][2] / 2.)
                    bbox_xmax.text = str(xmax)
