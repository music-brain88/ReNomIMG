#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import threading
import traceback
import csv
import numpy as np
import xml.etree.ElementTree as et
from PIL import Image
from renom.cuda import set_cuda_active, release_mem_pool
from .model.yolo import YoloDarknet
from .utils.data_preparation import create_train_valid_dists, create_pred_dist

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT_DIR = os.path.join(BASE_DIR, "../.storage/weight")
CSV_DIR = os.path.join(BASE_DIR, "../.storage/csv")
IMG_DIR = os.path.join(BASE_DIR, "../dataset/prediction_set/img")
XML_DIR = os.path.join(BASE_DIR, "../dataset/prediction_set/out")

STATE_RUNNING = 1
STATE_FINISHED = 2

TRAIN = 0
VALID = 1
PRED = 2
ERROR = -1

DEBUG = False


class PredictionThread(threading.Thread):
    def __init__(self, thread_id, hyper_parameters, algorithm, algorithm_params, weight_name):
        super(PredictionThread, self).__init__()
        self.stop_event = threading.Event()
        self.setDaemon(True)

        self.thread_id = thread_id
        self.algorithm = algorithm
        self.total_epoch = int(hyper_parameters['total_epoch'])
        self.batch_size = int(hyper_parameters['batch_size'])
        self.seed = int(hyper_parameters['seed'])
        self.img_size = (int(hyper_parameters['image_width']), int(
            hyper_parameters['image_height']))

        self.cell_h = int(algorithm_params['cells'])
        self.cell_v = int(algorithm_params['cells'])
        self.num_bbox = int(algorithm_params['bounding_box'])

        self.model = None
        self.weight_name = weight_name
        self.predict_results = {}
        self.csv_filename = ''

        self.error_msg = None
        self.total_batch = 0
        self.last_batch = 0

    def run(self):
        try:
            set_cuda_active(True)
            release_mem_pool()
            if DEBUG:
                print("run prediction")
            class_list, train_dist, valid_dist = create_train_valid_dists(
                self.img_size)
            self.model = self.set_train_config(len(class_list))
            self.model.load(os.path.join(WEIGHT_DIR, self.weight_name))
            self.run_prediction()
        except Exception as e:
            traceback.print_exc()
            self.error_msg = e.args[0]

    def run_prediction(self):
        try:
            distributor = create_pred_dist(self.img_size)

            v_bbox = []
            v_bbox_imgs = distributor._data_table

            self.model.set_models(inference=True)

            self.total_batch = np.ceil(len(distributor) / float(self.batch_size))

            start_t = time.time()
            for i, prediction_x in enumerate(distributor.batch(self.batch_size, False)):
                self.last_batch = i

                h = self.model.freezed_forward(prediction_x / 255. * 2 - 1)
                z = self.model(h)
                bbox = self.model.get_bbox(z.as_ndarray())
                v_bbox.extend(bbox)

            end_t = time.time()
            if DEBUG:
                print("predict time: {} [s]".format(end_t - start_t))

            self.predict_results = {
                "bbox_list": v_bbox[:len(v_bbox_imgs)],
                "bbox_path_list": v_bbox_imgs
            }
            self.save_predict_result_to_csv()
            self.save_predict_result_to_xml()
        except Exception as e:
            traceback.print_exc()
            self.error_msg = e.args[0]

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

                    bbox_xmin = et.SubElement(bbox_position, "xmin")
                    xmin = width * b["box"][0] - (width * b["box"][2] / 2.)
                    bbox_xmin.text = str(xmin)

                    bbox_ymax = et.SubElement(bbox_position, "ymax")
                    ymax = height * b["box"][1] + height * b["box"][3] / 2.
                    bbox_ymax.text = str(ymax)

                    bbox_ymin = et.SubElement(bbox_position, "ymin")
                    ymin = height * b["box"][1] - height * b["box"][3] / 2.
                    bbox_ymin.text = str(ymin)

                    bbox_score = et.SubElement(obj, "score")
                    bbox_score.text = str(b["score"])

                size = et.SubElement(annotation, "size")
                size_h = et.SubElement(size, "height")
                size_h.text = str(height)
                size_w = et.SubElement(size, "width")
                size_w.text = str(width)

                tree = et.ElementTree(annotation)

                tree.write(filepath)
        except Exception as e:
            traceback.print_exc()
            self.error_msg = e.args[0]

    def set_train_config(self, class_num):
        try:
            self._class_num = int(class_num)
            return YoloDarknet(cell=self.cell_h, bbox=self.num_bbox, class_num=class_num, img_size=self.img_size)
        except Exception as e:
            traceback.print_exc()
            self.error_msg = e.args[0]
