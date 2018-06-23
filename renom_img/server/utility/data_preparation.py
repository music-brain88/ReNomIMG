#!/usr/bin/env python
# encoding:utf-8

from __future__ import division, print_function
import pathlib
import os
import sys
import numpy as np
from xml.etree import ElementTree

from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.target import TargetBuilderYolov1
from renom_img.api.utility.load import parse_xml_detection

from renom_img.api.utility.augmentation.augmentation import Augmentation
from renom_img.api.utility.augmentation.process import Flip, Shift, Rotate, WhiteNoise

from .. import server


def create_train_valid_dists(img_size, train_imgs, valid_imgs):
    datasrc = pathlib.Path('datasrc')
    imgdir = (datasrc / "img")
    xmldir = (datasrc / "label")

    train_img_path_list = [str(imgdir / img) for img in train_imgs]
    train_xml_path_list = [str((xmldir / img).with_suffix(".xml")) for img in train_imgs]
    valid_img_path_list = [str(imgdir / img) for img in valid_imgs]
    valid_xml_path_list = [str((xmldir / img).with_suffix(".xml")) for img in valid_imgs]

    def create_label(xml_path_list, class_mapping=None):
        annotation_list = parse_xml_detection(xml_path_list)
        return build_target(annotation_list, img_size, class_mapping)

    # Check if the xml filename and img name is same.
    train_label, class_mapping = create_label(train_xml_path_list)
    valid_label, _ = create_label(valid_xml_path_list, class_mapping)
    aug = Augmentation([
        Flip(),
        Shift(40, 40),
        Rotate(),
        WhiteNoise(0.1),
    ])

    train_dist = ImageDetectionDistributor(
        train_img_path_list, train_label, TargetBuilderYolov1(), aug)
    valid_dist = ImageDetectionDistributor(valid_img_path_list, valid_label)
    class_list = [k for k, v in sorted(class_mapping.items(), key=lambda x: x[0])]
    return class_list, train_dist, valid_dist


def create_pred_dist(img_size):
    print('create_pred_dist')
    pred_img_path = 'dataset/prediction_set/img'
    pred_img_path_list = sorted(os.listdir(pred_img_path))
    pred_img_path_list = [os.path.join(pred_img_path, x)
                          for x in pred_img_path_list]
    print(pred_img_path_list)
    pred_dist = ImageDetectionDistributor(pred_img_path_list, None, img_size)
    return pred_dist
