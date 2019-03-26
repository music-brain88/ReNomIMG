#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import sys

import numpy as np
import renom as rm
from tqdm import tqdm

from renom_img import __version__
from renom_img.api.utility.misc.download import download
from renom_img.api.classification import Classification
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.cnn import CnnBase

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

class CNN_DenseNet(CnnBase):
    """
    DenseNet (Densely Connected Convolutional Network) https://arxiv.org/pdf/1608.06993.pdf

    Input
        class_map: Array of class names
        layer_per_block: array specifing number of layers in a block.
        growth_rate(int): Growth rate of the number of filters.
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.
    """

    def __init__(self, num_class, layer_per_block=[6, 12, 24, 16], growth_rate=32):
        super(CNN_DenseNet, self).__init__()
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
        self._freeze()
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

    def _freeze(self):
        self.base.set_auto_update(self.train_whole)

    def set_output_size(self, output_size):
        self.output_size = output_size
        self.fc._output_size = output_size

    

