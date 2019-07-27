from __future__ import print_function, division
import os
import numpy as np
import renom as rm
from tqdm import tqdm

from renom_img import __version__
from renom_img.api import Base, adddoc
from renom_img.api.classification import Classification
from renom_img.api.utility.misc.download import download
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.cnn import CnnBase
from renom_img.api.utility.exceptions.exceptions import *


def layer_factory(channel=32, conv_layer_num=2):
    layers = []
    for _ in range(conv_layer_num):
        layers.append(rm.Conv2d(channel=channel, padding=1, filter=3))
        layers.append(rm.Relu())
    layers.append(rm.MaxPool2d(filter=2, stride=2))
    return rm.Sequential(layers)


class CNN_VGG19(CnnBase):

    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/classification/VGG19.h5".format(
        __version__)

    def __init__(self, num_class=1000):
        super(CNN_VGG19, self).__init__()
        self.has_bn = False
        self.block1 = layer_factory(channel=64, conv_layer_num=2)
        self.block2 = layer_factory(channel=128, conv_layer_num=2)
        self.block3 = layer_factory(channel=256, conv_layer_num=4)
        self.block4 = layer_factory(channel=512, conv_layer_num=4)
        self.block5 = layer_factory(channel=512, conv_layer_num=4)
        self.fc1 = rm.Dense(4096)
        self.dropout1 = rm.Dropout(dropout_ratio=0.5)
        self.fc2 = rm.Dense(4096)
        self.dropout2 = rm.Dropout(dropout_ratio=0.5)
        self.fc3 = rm.Dense(num_class)
        self.relu = rm.Relu()

    def forward(self, x):
        self._freeze()
        t = self.block1(x)
        t = self.block2(t)
        t = self.block3(t)
        t = self.block4(t)
        t = self.block5(t)
        t = rm.flatten(t)
        t = self.relu(self.fc1(t))
        t = self.dropout1(t)
        t = self.relu(self.fc2(t))
        t = self.dropout2(t)
        t = self.fc3(t)
        return t

    def _freeze(self):
        self.block1.set_auto_update(self.train_whole)
        self.block2.set_auto_update(self.train_whole)
        self.block3.set_auto_update(self.train_whole)
        self.block4.set_auto_update(self.train_whole)
        self.block5.set_auto_update(self.train_whole)

    def set_output_size(self, output_size):
        self.output_size = output_size
        self.fc3._output_size = output_size

    def load_pretrained_weight(self, path):
        try:
            self.load(path)
        except:
            raise WeightLoadError(
                'The pretrained weights path {} can not be loaded into the class {}.'.format(path, self.__class__))


class CNN_VGG16(CnnBase):

    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/classification/VGG16.h5".format(
        __version__)

    def __init__(self, num_class=1000):
        super(CNN_VGG16, self).__init__()
        self.has_bn = False
        self.block1 = layer_factory(channel=64, conv_layer_num=2)
        self.block2 = layer_factory(channel=128, conv_layer_num=2)
        self.block3 = layer_factory(channel=256, conv_layer_num=3)
        self.block4 = layer_factory(channel=512, conv_layer_num=3)
        self.block5 = layer_factory(channel=512, conv_layer_num=3)
        self.fc1 = rm.Dense(4096)
        self.dropout1 = rm.Dropout(dropout_ratio=0.5)
        self.fc2 = rm.Dense(4096)
        self.dropout2 = rm.Dropout(dropout_ratio=0.5)
        self.fc3 = rm.Dense(num_class)
        self.relu = rm.Relu()

    def forward(self, x):
        self._freeze()
        t = self.block1(x)
        t = self.block2(t)
        t = self.block3(t)
        t = self.block4(t)
        t = self.block5(t)
        t = rm.flatten(t)
        t = self.relu(self.fc1(t))
        t = self.dropout1(t)
        t = self.relu(self.fc2(t))
        t = self.dropout2(t)
        t = self.fc3(t)
        return t

    def _freeze(self):
        self.block1.set_auto_update(self.train_whole)
        self.block2.set_auto_update(self.train_whole)
        self.block3.set_auto_update(self.train_whole)
        self.block4.set_auto_update(self.train_whole)
        self.block5.set_auto_update(self.train_whole)

    def set_output_size(self, output_size):
        self.output_size = output_size
        self.fc3._output_size = output_size

    def load_pretrained_weight(self, path):
        try:
            self.load(path)
        except:
            raise WeightLoadError(
                'The pretrained weights path {} can not be loaded into the class {}.'.format(path, self.__class__))


class CNN_VGG16_NODENSE(CnnBase):

    def __init__(self, num_class=1000):
        super(CNN_VGG16_NODENSE, self).__init__()
        self.has_bn = False
        self.conv1_1 = rm.Conv2d(64, padding=1, filter=3)
        self.conv1_2 = rm.Conv2d(64, padding=1, filter=3)
        self.conv2_1 = rm.Conv2d(128, padding=1, filter=3)
        self.conv2_2 = rm.Conv2d(128, padding=1, filter=3)
        self.conv3_1 = rm.Conv2d(256, padding=1, filter=3)
        self.conv3_2 = rm.Conv2d(256, padding=1, filter=3)
        self.conv3_3 = rm.Conv2d(256, padding=1, filter=3)
        self.conv4_1 = rm.Conv2d(512, padding=1, filter=3)
        self.conv4_2 = rm.Conv2d(512, padding=1, filter=3)
        self.conv4_3 = rm.Conv2d(512, padding=1, filter=3)
        self.conv5_1 = rm.Conv2d(512, padding=1, filter=3)
        self.conv5_2 = rm.Conv2d(512, padding=1, filter=3)
        self.conv5_3 = rm.Conv2d(512, padding=1, filter=3)
        self.relu = rm.Relu()

    def forward(self, x):
        t = self.relu(self.conv1_1(x))
        t = self.relu(self.conv1_2(t))
        t = rm.max_pool2d(t, filter=2, stride=2)

        t = self.relu(self.conv2_1(t))
        t = self.relu(self.conv2_2(t))
        t = rm.max_pool2d(t, filter=2, stride=2)

        t = self.relu(self.conv3_1(t))
        t = self.relu(self.conv3_2(t))
        t = self.relu(self.conv3_3(t))
        t = rm.max_pool2d(t, filter=2, stride=2)

        t = self.relu(self.conv4_1(t))
        t = self.relu(self.conv4_2(t))
        t = self.relu(self.conv4_3(t))
        t = rm.max_pool2d(t, filter=2, stride=2)

        t = self.relu(self.conv5_1(t))
        t = self.relu(self.conv5_2(t))
        t = self.relu(self.conv5_3(t))
        t = rm.max_pool2d(t, filter=2, stride=2)

        return t


class CNN_VGG11(CnnBase):

    def __init__(self, num_class=1000):
        super(CNN_VGG11, self).__init__()
        self.has_bn = False
        self.block1 = layer_factory(channel=64, conv_layer_num=1)
        self.block2 = layer_factory(channel=128, conv_layer_num=1)
        self.block3 = layer_factory(channel=256, conv_layer_num=2)
        self.block4 = layer_factory(channel=512, conv_layer_num=2)
        self.block5 = layer_factory(channel=512, conv_layer_num=2)
        self.fc1 = rm.Dense(4096)
        self.dropout1 = rm.Dropout(dropout_ratio=0.5)
        self.fc2 = rm.Dense(4096)
        self.dropout2 = rm.Dropout(dropout_ratio=0.5)
        self.fc3 = rm.Dense(num_class)
        self.relu = rm.Relu()

    def forward(self, x):
        self._freeze()
        t = self.block1(x)
        t = self.block2(t)
        t = self.block3(t)
        t = self.block4(t)
        t = self.block5(t)
        t = rm.flatten(t)
        t = self.relu(self.fc1(t))
        t = self.dropout1(t)
        t = self.relu(self.fc2(t))
        t = self.dropout2(t)
        t = self.fc3(t)
        return t

    def _freeze(self):
        self.block1.set_auto_update(self.train_whole)
        self.block2.set_auto_update(self.train_whole)
        self.block3.set_auto_update(self.train_whole)
        self.block4.set_auto_update(self.train_whole)
        self.block5.set_auto_update(self.train_whole)

    def set_output_size(self, output_size):
        self.output_size = output_size
        self.fc3._output_size = output_size
