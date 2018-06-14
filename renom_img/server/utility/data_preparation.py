#!/usr/bin/env python
# encoding:utf-8

from __future__ import division, print_function
import os
import sys
import numpy as np
from xml.etree import ElementTree

from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.target import TargetBuilderYolov1
from renom_img.api.utility.load import parse_xml_detection

from renom_img.api.utility.augmentation.augmentation import Augmentation
from renom_img.api.utility.augmentation.process import Flip, Shift, Rotate, WhiteNoise


def create_train_valid_dists(img_size, cell):
    train_img_path = 'dataset/train_set/img'
    train_xml_path = 'dataset/train_set/label'
    valid_img_path = 'dataset/valid_set/img'
    valid_xml_path = 'dataset/valid_set/label'

    train_xml_path_list = [os.path.join(train_xml_path, path) for path in sorted(os.listdir(train_xml_path))]
    train_img_path_list = [os.path.join(train_img_path, path)
                           for path in sorted(os.listdir(train_img_path))]
    valid_xml_path_list = [os.path.join(valid_xml_path, path)
                           for path in sorted(os.listdir(valid_xml_path))]
    valid_img_path_list = [os.path.join(valid_img_path, path)
                           for path in sorted(os.listdir(valid_img_path))]

    # Check if the xml filename and img name is same.
    train_label, class_mapping = create_label(train_xml_path_list)
    valid_label, _ = create_label(valid_xml_path_list, class_mapping)

    aug = Augmentation([
        Flip(),
        Shift(40, 40),
        Rotate(),
        WhiteNoise(0.1),
    ])

    train_dist = ImageDetectionDistributor(train_img_path_list, train_label, TargetBuilderYolov1(), aug)
    valid_dist = ImageDetectionDistributor(valid_img_path_list, valid_label)
    class_list = [k for k, v in sorted(class_mapping.items(), key=lambda x: x[0])]
    return class_list, train_dist, valid_dist


def create_pred_dist(img_size):
    pred_img_path = 'dataset/prediction_set/img'
    pred_img_path_list = sorted(os.listdir(pred_img_path))
    pred_img_path_list = [os.path.join(pred_img_path, x)
                          for x in pred_img_path_list]
    pred_dist = ImageDetectionDistributor(pred_img_path_list, None, img_size)
    return pred_dist
