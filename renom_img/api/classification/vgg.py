#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import numpy as np
import renom as rm
from tqdm import tqdm
from renom_img.api.utility.misc.download import download
from renom_img.api.model.classification_base import ClassificationBase
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.target import DataBuilderClassification

DIR = os.path.split(os.path.abspath(__file__))[0]


def layer_factory(channel=32, conv_layer_num=2):
    layers = []
    for _ in range(conv_layer_num):
        layers.append(rm.Conv2d(channel=channel, padding=1, filter=3))
        layers.append(rm.Relu())
    layers.append(rm.MaxPool2d(filter=2, stride=2))
    return rm.Sequential(layers)


class VGGBase(ClassificationBase):
    def get_optimizer(self, current_epoch=None, total_epoch=None, current_batch=None, total_batch=None, **kwargs):
        """Returns an instance of Optimiser for training Yolov1 algorithm.

        Args:
            current_epoch:
            total_epoch:
            current_batch:
            total_epoch:
        """
        if any([num is None for num in [current_epoch, total_epoch, current_batch, total_batch]]):
            return self._opt
        else:
            avg_valid_loss_list = kwargs['avg_valid_loss_list']
            if avg_valid_loss[-1] > avg_valid_loss[-2]:
                self._opt._lr = lr / 10.
            return self._opt

    def preprocess(self, x):
        """Image preprocess for VGG.

        Args:
            x (ndarray):

        Returns:
            (ndarray): Preprocessed data.
        """
        x[:, 0, :, :] -= 123.68  # R
        x[:, 1, :, :] -= 116.779  # G
        x[:, 2, :, :] -= 103.939  # B
        return x

class VGG16(VGGBase):
    """VGG16 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        class_map(array): Array of class names
        load_weight(bool):
        imsize(int or tuple): Input image size
        train_whole_network(bool): True if the overall model is trained, otherwise False


    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition
    https://arxiv.org/abs/1409.1556
    """

    WEIGHT_URL = "http://docs.renom.jp/downloads/weights/Vgg16.h5"
    WEIGHT_PATH = os.path.join(DIR, 'vgg16.h5')

    def __init__(self, class_map, load_weight=False, imsize=(224, 224), decay_rate=0.0005, train_whole_network=False):
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)

        self.n_class = len(class_map)
        self.class_map = class_map
        self._train_whole_network = train_whole_network
        self.imsize = imsize
        self._freezed_network = CNN_VGG19()
        self._network = rm.Dense(self.n_class)
        self._opt = rm.Sgd(0.01, 0.9)
        self.decay_rate = decay_rate

        if self.n_class != 1000:
            for layer in self._network.iter_models():
                if hasattr(layer, "params"):
                    layer.params = {}

        super(VGG16, self).__init__(class_map)


class VGG19(VGGBase):
    """VGG19 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        class_map(array): Array of class names
        load_weight(bool):
        imsize(int or tuple): Input image size
        train_whole_network(bool): True if the overall model is trained, otherwise False


    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition
    https://arxiv.org/abs/1409.1556
    """

    WEIGHT_URL = "http://docs.renom.jp/downloads/weights/Vgg16.h5"
    WEIGHT_PATH = os.path.join(DIR, 'vgg19.h5')

    def __init__(self, class_map, imsize=(224, 224), decay_rate=0.0005, load_weight=False, train_whole_network=False):
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)

        self.n_class = len(class_map)
        self.class_map = class_map
        self._train_whole_network = train_whole_network
        self.imsize = imsize
        self._model = CNN_VGG16()
        self._opt = rm.Sgd(0.01, 0.9)
        self.decay_rate = decay_rate

        if load_weight:
            try:
                self._model.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self._model.load(self.WEIGHT_PATH)
        if self.n_class != 1000:
            self._model.params = {}

class CNN_VGG16(rm.Model):
    def __init__(self, n_class, train_whole_network=False):
        model = [
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
            rm.Dense(n_class)
        ]
        self._freezed_network = rm.Sequential(model[:5])
        self._network = rm.Sequential(model[5:])
        self._train_whole_network = train_whole_network

    @property
    def freezed_network(self):
        return self._freezed_network

    @property
    def network(self):
        return self._network

    def forward(self, x):
        self.freezed_network.set_auto_update(self._train_whole_network)
        return self.network(self.freezed_network(x))

class CNN_VGG19(rm.Sequential):
    def __init__(self, n_class, train_whole_network=False):
        model = [
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
            rm.Dense(n_class)
        ]
        self._freezed_network = rm.Sequential(model[:5])
        self._network = rm.Sequential(model[5:])
        self._train_whole_network = train_whole_network

    @property
    def freezed_network(self):
        return self._freezed_network

    @property
    def network(self):
        return self._network

    def forward(self, x):
        self.freezed_network.set_auto_update(self._train_whole_network)
        return self.network(self.freezed_network(x))
