#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import sys

import numpy as np
import renom as rm
from renom_img.api.utility.misc.download import download
from renom_img.api.model.classification_base import ClassificationBase
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

class DenseNetBase(ClassificationBase):
    def __init__(self, class_map):
        super(DenseNetBase, self).__init__(class_map)
        self._opt = rm.Sgd(0.1, 0.9)

    def get_optimizer(self, current_epoch=None, total_epoch=None, current_batch=None, total_batch=None):
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
            ind1 = int(total_epoch * 0.5)
            ind2 = int(total_epoch * 0.3)
            ind3 = total_epoch - (ind1 + ind2 + 1)
            lr_list = [0] + [0.01] * ind1 + [0.001] * ind2 + [0.0001] * ind3
            if current_epoch == 0:
                lr = 0.0001 + (0.01 - 0.0001) / float(total_batch) * current_batch
            else:
                lr = lr_list[current_epoch]
            self._opt._lr = lr
            return self._opt


    def preprocess(self, x):
        """Image preprocess for VGG.

        Args:
            x (ndarray):

        Returns:
            (ndarray): Preprocessed data.
        """
        return x / 255.

    def regularize(self, decay_rate=0.0005):
        """L2 Regularization term. You can use this function to add L2 regularization term to a loss function.

        In VGG16, weight decay of 0.0005 is used.

        Example:
            >>> import numpy as np
            >>> from renom_img.api.model.vgg import VGG16
            >>> x = np.random.rand(1, 3, 224, 224)
            >>> y = np.random.rand(1, (5*2+20)*7*7)
            >>> model = VGG16()
            >>> loss = model.loss(x, y)
            >>> reg_loss = loss + model.regularize() # Add weight decay term.

        """
        return super().regularize(decay_rate)




class DenseNet(DenseNetBase):
    """
    DenseNet (Densely Connected Convolutional Network) https://arxiv.org/pdf/1608.06993.pdf

    Input
        class_map: Number of classes
        layer_per_block: array specifing number of layers in a block
        growth_rate: int1
    """
    def __init__(self, class_map, layer_per_block=[6, 12, 24, 16], growth_rate=32, imsize=(224, 224), train_whole_network=False):
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        self.layer_per_block = layer_per_block
        self.growth_rate = growth_rate
        self.n_class = len(class_map)

        layers = []
        layers.append(rm.Conv2d(64, 7, padding=3, stride=2))
        layers.append(rm.BatchNormalize(epsilon=0.001, mode='feature'))
        for i in layer_per_block[:-1]:
            for j in range(i):
                layers.append(conv_block(growth_rate))
            layers.append(transition_layer(growth_rate))
        for i in range(layer_per_block[-1]):
            layers.append(conv_block(growth_rate))

        self._freezed_network = rm.Sequential(layers)
        self._network = rm.Dense(self.n_class)
        self._train_whole_network = train_whole_network
        self.imsize = imsize

        super(DenseNet, self).__init__(class_map)

    @property
    def freezed_network(self):
        return self._freezed_network

    @property
    def network(self):
        return self._network

    def forward(self, x):
        self.freezed_network.set_auto_update(self._train_whole_network)
        i = 0
        t = self.freezed_network[i](x)
        i += 1
        t = rm.relu(self._layers[i](t))
        i += 1
        t = rm.max_pool2d(t, filter=3, stride=2, padding=1)
        for j in self.layer_per_block[:-1]:
            for k in range(j):
                tmp = t
                t = self.freezed_network[i](t)
                i += 1
                t = rm.concat(tmp, t)
            t = self.freezed_network[i](t)
            i += 1
        for j in range(self.layer_per_block[-1]):
            tmp = t
            t = self.freezed_network[i](t)
            i += 1
            t = rm.concat(tmp, t)
        t = rm.average_pool2d(t, filter=7, stride=1)
        t = rm.flatten(t)
        t = self.network(t)
        return t


class DenseNet121(DenseNet):
    """ DenseNet121 Model

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        n_class(int): The number of class
        layer_per_block: The number of layers in each block
        load_weight(bool): 

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
    Densely Connected Convolutional Network
    https://arxiv.org/pdf/1608.06993.pdf
    """

    WEIGHT_URL = "https://app.box.com/shared/static/eovmxxgzyh5vg2kpcukjj8ypnxng4j5v.h5"
    WEIGHT_PATH = os.path.join(DIR, 'densenet121.h5')

    def __init__(self, class_map, growth_rate=32, load_weight=False):
        layer_per_block = [6, 12, 24, 16]
        super(DenseNet121, self).__init__(class_map, layer_per_block, growth_rate=32, imsize=(224, 224), train_whole_network=False)
        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if n_class != 1000:
            self._layers[-1].params = {}


class DenseNet169(DenseNet):
    """ DenseNet169 Model

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        n_class(int): The number of class
        layer_per_block: The number of layers in each block
        load_weight(bool): 

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
    Densely Connected Convolutional Network
    https://arxiv.org/pdf/1608.06993.pdf
    """

    def __init__(self, class_map, growth_rate=32, load_weight=False):
        layer_per_block = [6, 12, 32, 32]
        super(DenseNet121, self).__init__(class_map, layer_per_block, growth_rate=32, imsize=(224, 224), train_whole_network=False)
        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if n_class != 1000:
            self._layers[-1].params = {}


class DenseNet201(DenseNet):
    """ DenseNet201 Model

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        n_class(int): The number of class
        layer_per_block: The number of layers in each block
        load_weight(bool): 

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
    Densely Connected Convolutional Network
    https://arxiv.org/pdf/1608.06993.pdf
    """

    WEIGHT_URL = "https://app.box.com/shared/static/eovmxxgzyh5vg2kpcukjj8ypnxng4j5v.h5"
    WEIGHT_PATH = os.path.join(DIR, 'densenet201.h5')

    def __init__(self, class_map, growth_rate=32, load_weight=False):
        layer_per_block = [6, 12, 48, 32]
        super(DenseNet121, self).__init__(class_map, layer_per_block, growth_rate=32, imsize=(224, 224), train_whole_network=False)
        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if n_class != 1000:
            self._layers[-1].params = {}
