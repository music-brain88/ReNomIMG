#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import sys

import numpy as np
import renom as rm
from tqdm import tqdm
from renom_img.api.utility.misc.download import download
from renom_img.api.classification import Classification
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.utility.distributor.distributor import ImageDistributor

DIR = os.path.split(os.path.abspath(__file__))[0]


def conv_block(growth_rate):
    return rm.Sequential([
        rm.BatchNormalize(epsilon=0.001, mode='feature'),
        rm.Relu(),
        rm.Conv2d(growth_rate * 4, 1, padding=0),
        rm.BatchNormalize(epsilon=0.001, mode='feature'),
        rm.Relu(),
        rm.Conv2d(growth_rate, 3, padding=1),
    ])


def transition_layer(growth_rate):
    return rm.Sequential([
        rm.BatchNormalize(epsilon=0.001, mode='feature'),
        rm.Relu(),
        rm.Conv2d(growth_rate, filter=1, padding=0, stride=1),
        rm.AveragePool2d(filter=2, stride=2)
    ])


class DenseNetBase(Classification):
    def __init__(self, class_map):
        super(DenseNetBase, self).__init__(class_map)
        self._opt = rm.Sgd(0.1, 0.9)

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
            if current_epoch == 30:
                self._opt._lr = self._opt._lr / 10
            elif current_epoch == 60:
                self._opt._lr = self._opt._lr / 10
            return self._opt

    def _freeze(self):
        self._model.base.set_auto_update(self._train_whole_network)


class CNN_DenseNet(rm.Model):
    """
    DenseNet (Densely Connected Convolutional Network) https://arxiv.org/pdf/1608.06993.pdf

    Input
        class_map: Array of class names
        layer_per_block: array specifing number of layers in a block.
        growth_rate(int): Growth rate of the number of filters.
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.
    """

    def __init__(self, num_class, layer_per_block=[6, 12, 24, 16], growth_rate=32, train_whole_network=False):
        self.layer_per_block = layer_per_block
        self.growth_rate = growth_rate

        layers = []
        layers.append(rm.Conv2d(64, 7, padding=3, stride=2))
        layers.append(rm.BatchNormalize(epsilon=0.001, mode='feature'))
        for i in layer_per_block[:-1]:
            for j in range(i):
                layers.append(conv_block(growth_rate))
            layers.append(transition_layer(growth_rate))
        for i in range(layer_per_block[-1]):
            layers.append(conv_block(growth_rate))

        self.base = rm.Sequential(layers)
        self.fc = rm.Dense(num_class)

    def forward(self, x):
        i = 0
        t = self.base[i](x)
        i += 1
        t = rm.relu(self.base[i](t))
        i += 1
        t = rm.max_pool2d(t, filter=3, stride=2, padding=1)
        for j in self.layer_per_block[:-1]:
            for k in range(j):
                tmp = t
                t = self.base[i](t)
                i += 1
                t = rm.concat(tmp, t)
            t = self.base[i](t)
            i += 1
        for j in range(self.layer_per_block[-1]):
            tmp = t
            t = self.base[i](t)
            i += 1
            t = rm.concat(tmp, t)
        t = rm.average_pool2d(t, filter=7, stride=1)
        t = rm.flatten(t)
        t = self.fc(t)
        return t


class DenseNet121(DenseNetBase):
    """ DenseNet121 Model

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        class_map: Array of class names
        growth_rate(int): Growth rate of the number of filters.
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument num_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    References:
        Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
        Densely Connected Convolutional Network
        https://arxiv.org/pdf/1608.06993.pdf
    """

    SERIALIZED = ("imsize", "class_map", "num_class")
    WEIGHT_URL = "http://docs.renom.jp/downloads/weights/DenseNet121.h5"

    def __init__(self, class_map=[], imsize=(224, 224), load_pretrained_weight=False, train_whole_network=False):
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)

        layer_per_block = [6, 12, 24, 16]
        growth_rate = 32
        self.imsize = imsize
        self.num_class = len(class_map)
        self.class_map = [c.encode("ascii", "ignore") for c in class_map]
        self._train_whole_network = train_whole_network
        self._model = CNN_DenseNet(self.num_class, layer_per_block,
                                   growth_rate, train_whole_network)
        self._opt = rm.Sgd(0.01, 0.9)
        self.decay_rate = 0.0005

        if load_pretrained_weight:
            if isinstance(load_pretrained_weight, bool):
                load_pretrained_weight = self.__class__.__name__ + '.h5'

            if not os.path.exists(load_pretrained_weight):
                download(self.weight_url, load_pretrained_weight)

            self._model.load(load_pretrained_weight)
            for layer in self._model._network.iter_models():
                layer.params = {}
        if self.num_class != 1000:
            self._model.params = {}
        self._freeze()


class DenseNet169(DenseNetBase):
    """ DenseNet169 Model

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        class_map: Array of class names
        growth_rate(int): Growth rate of the number of filters.
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument num_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    References:
        Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
        Densely Connected Convolutional Network
        https://arxiv.org/pdf/1608.06993.pdf
    """

    SERIALIZED = ("imsize", "class_map", "num_class")
    WEIGHT_URL = "http://docs.renom.jp/downloads/weights/DenseNet169.h5"

    def __init__(self, class_map=[], imsize=(224, 224), load_pretrained_weight=False, train_whole_network=False):
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)

        layer_per_block = [6, 12, 32, 32]
        growth_rate = 32
        self.imsize = imsize
        self.num_class = len(class_map)
        self.class_map = [c.encode("ascii", "ignore") for c in class_map]
        self._train_whole_network = train_whole_network
        self._model = CNN_DenseNet(self.num_class, layer_per_block,
                                   growth_rate, train_whole_network)
        self._opt = rm.Sgd(0.01, 0.9)
        self.decay_rate = 0.0005

        if load_pretrained_weight:
            if isinstance(load_pretrained_weight, bool):
                load_pretrained_weight = self.__class__.__name__ + '.h5'

            if not os.path.exists(load_pretrained_weight):
                download(self.weight_url, load_pretrained_weight)

            self._model.load(load_pretrained_weight)
            for layer in self._model._network.iter_models():
                layer.params = {}
        if self.num_class != 1000:
            self._model.params = {}
        self._freeze()


class DenseNet201(DenseNetBase):
    """ DenseNet201 Model

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        class_map: Array of class names
        growth_rate(int): Growth rate of the number of filters.
        load_weight(bool): True if the pre-trained weight is loaded.
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument num_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    References:
        Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
        Densely Connected Convolutional Network
        https://arxiv.org/pdf/1608.06993.pdf
    """

    SERIALIZED = ("imsize", "class_map", "num_class")
    WEIGHT_URL = "http://docs.renom.jp/downloads/weights/DenseNet201.h5"

    def __init__(self, class_map=[], imsize=(224, 224), load_pretrained_weight=False, train_whole_network=False):
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)

        layer_per_block = [6, 12, 48, 32]
        growth_rate = 32
        self.imsize = imsize
        self.num_class = len(class_map)
        self.class_map = [c.encode("ascii", "ignore") for c in class_map]
        self._train_whole_network = train_whole_network
        self._model = CNN_DenseNet(self.num_class, layer_per_block,
                                   growth_rate, train_whole_network)
        self._opt = rm.Sgd(0.01, 0.9)
        self.decay_rate = 0.0005

        if load_pretrained_weight:
            if isinstance(load_pretrained_weight, bool):
                load_pretrained_weight = self.__class__.__name__ + '.h5'

            if not os.path.exists(load_pretrained_weight):
                download(self.WEIGHT_URL, load_pretrained_weight)

            self._model.load(load_pretrained_weight)
            for layer in self._model._network.iter_models():
                layer.params = {}
        if self.num_class != 1000:
            self._model.params = {}
        self._freeze()
