#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import numpy as np
import renom as rm
from renom_img.api.utility.misc.download import download
from renom_img.api.model.classification_base import ClassificationBase

DIR = os.path.split(os.path.abspath(__file__))[0]


def layer_factory(channel=32, conv_layer_num=2):
    layers = []
    for _ in range(conv_layer_num):
        layers.append(rm.Conv2d(channel=channel, padding=1, filter=3))
        layers.append(rm.Relu())
    layers.append(rm.MaxPool2d(filter=2, stride=2))
    return rm.Sequential(layers)

class VGGBase(ClassificationBase):
    def __init__(self, class_map):
        super(VGGBase, self).__init__(class_map)
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
        x[:, 0, :, :] -= 123.68  # R
        x[:, 1, :, :] -= 116.779 # G
        x[:, 2, :, :] -= 103.939 # B
        return x

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


class VGG16(VGGBase):
    """VGG16 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        n_class(int):
        load_weight(bool):

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition
    https://arxiv.org/abs/1409.1556
    """

    WEIGHT_URL = "http://docs.renom.jp/downloads/weights/Vgg16.h5"
    WEIGHT_PATH = os.path.join(DIR, 'vgg16.h5')

    def __init__(self, class_map, imsize=(224, 224), load_weight=False, train_whole_network=False):
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)

        self.n_class = len(class_map)

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
            rm.Dense(self.n_class)
        ]
        self.class_map = class_map
        self._train_whole_network = train_whole_network
        self.imsize = imsize
        self._freezed_network = rm.Sequential(model[:5])
        self._network = rm.Sequential(model[5:])

        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if self.n_class != 1000:
            self._freezed_network[-1].params = {}
            self._network[-1].params = {}

        super(VGG16, self).__init__(class_map)

    @property
    def freezed_network(self):
        return self._freezed_network

    @property
    def network(self):
        return self._network

    def forward(self, x):
        self.freezed_network.set_auto_update(self._train_whole_network)
        return self.network(self.freezed_network(x))




class VGG19(VGGBase):
    """VGG19 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        n_class(int):
        load_weight(bool):

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition
    https://arxiv.org/abs/1409.1556
    """

    WEIGHT_URL = "http://docs.renom.jp/downloads/weights/Vgg16.h5"
    WEIGHT_PATH = os.path.join(DIR, 'vgg19.h5')

    def __init__(self, class_map, imsize=(224, 224), load_weight=False, train_whole_network=False):
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)

        self.n_class = len(class_map)

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
            rm.Dense(self.n_class)
        ]
        self.class_map = class_map
        self._train_whole_network = train_whole_network
        self.imsize = imsize
        self._freezed_network = rm.Sequential(model[:5])
        self._network = rm.Sequential(model[5:])

        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if self.n_class != 1000:
            self._freezed_network[-1].params = {}
            self._network[-1].params = {}

        super(VGG16, self).__init__(class_map)

    @property
    def freezed_network(self):
        return self._freezed_network

    @property
    def network(self):
        return self._network

    def forward(self, x):
        self.freezed_network.set_auto_update(self._train_whole_network)
        return self.network(self.freezed_network(x))
