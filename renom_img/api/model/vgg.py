#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import numpy as np
import renom as rm
from renom_img.api.utils.misc.download import download

DIR = os.path.split(os.path.abspath(__file__))[0]


def layer_factory(channel=32, conv_layer_num=2):
    layers = []
    for _ in range(conv_layer_num):
        layers.append(rm.Conv2d(channel=channel, padding=1, filter=3))
        layers.append(rm.Relu())
    layers.append(rm.MaxPool2d(filter=2, stride=2))
    return rm.Sequential(layers)


class VGG16(rm.Sequential):
    """VGG16 model. 

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        num_class(int): 
        load_weight(bool): 

    Note:
        if the argument num_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Karen Simonyan, Andrew Zisserman  
    Very Deep Convolutional Networks for Large-Scale Image Recognition
    https://arxiv.org/abs/1409.1556
    """

    WEIGHT_URL = "https://app.box.com/shared/static/o81vwdp4qsm88zt93jvpskqfzobhfx6s.h5"
    WEIGHT_PATH = os.path.join(DIR, 'vgg16.h5')

    def __init__(self, num_class=1000, load_weight=False):
        super(VGG16, self).__init__([
            layer_factory(channel=64, conv_layer_num=2),
            layer_factory(channel=128, conv_layer_num=2),
            layer_factory(channel=256, conv_layer_num=3),
            layer_factory(channel=512, conv_layer_num=3),
            layer_factory(channel=512, conv_layer_num=3),
            rm.Flatten(),
            rm.Dense(4096),
            rm.Relu(),
            rm.Dropout(0.5),
            rm.Dense(4096),
            rm.Relu(),
            rm.Dropout(0.5),
            rm.Dense(num_class)
        ])
        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if num_class != 1000:
            self._layers[-1].params = {}


class VGG19(rm.Sequential):
    """VGG19

    Karen Simonyan, Andrew Zisserman  
    Very Deep Convolutional Networks for Large-Scale Image Recognition
    https://arxiv.org/abs/1409.1556
    """

    WEIGHT_URL = "https://app.box.com/shared/static/eovmxxgzyh5vg2kpcukjj8ypnxng4j5v.h5"
    WEIGHT_PATH = os.path.join(DIR, 'vgg19.h5')

    def __init__(self, num_class=1000, load_weight=False):
        super(VGG19, self).__init__([
            layer_factory(channel=64, conv_layer_num=2),
            layer_factory(channel=128, conv_layer_num=2),
            layer_factory(channel=256, conv_layer_num=4),
            layer_factory(channel=512, conv_layer_num=4),
            layer_factory(channel=512, conv_layer_num=4),
            rm.Flatten(),
            rm.Dense(4096),
            rm.Relu(),
            rm.Dropout(0.5),
            rm.Dense(4096),
            rm.Relu(),
            rm.Dropout(0.5),
            rm.Dense(num_class)
        ])
        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if num_class != 1000:
            self._layers[-1].params = {}

