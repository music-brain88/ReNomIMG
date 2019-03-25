import os
import sys
import renom as rm
import numpy as np
from tqdm import tqdm

from renom_img import __version__
from renom_img.api import Base, adddoc
from renom_img.api.utility.misc.download import download
from renom_img.api.classification import Classification
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.target import DataBuilderClassification
from renom_img.api.cnn import CnnBase

class Bottleneck(rm.Model):
    expansion = 2

    def __init__(self, planes, stride=1, downsample=None, cardinality=32):
        super(Bottleneck, self).__init__()
        self.cardinality = cardinality
        self.conv1 = rm.Conv2d(planes, filter=1, ignore_bias=True)
        self.bn1 = rm.BatchNormalize(epsilon=0.00001, mode='feature')
        self.conv2 = rm.GroupConv2d(planes, filter=3, stride=stride,
                                    padding=1, ignore_bias=True, groups=self.cardinality)
        self.bn2 = rm.BatchNormalize(epsilon=0.00001, mode='feature')
        self.conv3 = rm.Conv2d(planes * self.expansion, filter=1, ignore_bias=True)
        self.bn3 = rm.BatchNormalize(epsilon=0.00001, mode='feature')
        self.relu = rm.Relu()
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class CnnResNeXt(CnnBase):

    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/classification/ResNeXt.h5".format(
        __version__)

    def __init__(self, num_classes, block, layers, cardinality):
        self.inplanes = 128
        self.cardinality = cardinality
        super(CnnResNeXt, self).__init__()
        self.conv1 = rm.Conv2d(64, filter=7, stride=2, padding=3, ignore_bias=True)
        self.bn1 = rm.BatchNormalize(epsilon=0.00001, mode='feature')
        self.relu = rm.Relu()
        self.maxpool = rm.MaxPool2d(filter=3, stride=2, padding=1)
        self.layer1 = self._make_layer(
            block, 128, layers[0], stride=1, cardinality=self.cardinality)
        self.layer2 = self._make_layer(
            block, 256, layers[1], stride=2, cardinality=self.cardinality)
        self.layer3 = self._make_layer(
            block, 512, layers[2], stride=2, cardinality=self.cardinality)
        self.layer4 = self._make_layer(
            block, 1024, layers[3], stride=2, cardinality=self.cardinality)
        self.flat = rm.Flatten()
        self.fc = rm.Dense(num_classes)

    def _make_layer(self, block, planes, blocks, stride=1, cardinality=32):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = rm.Sequential([
                rm.Conv2d(planes * block.expansion, filter=1, stride=stride, ignore_bias=True),
                rm.BatchNormalize(epsilon=0.00001, mode='feature')
            ])

        layers = []
        layers.append(block(planes, stride, downsample, cardinality=cardinality))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(planes, cardinality=cardinality))

        return rm.Sequential(layers)

    def forward(self, x):
        self._freeze()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = rm.average_pool2d(x, filter=(x.shape[2], x.shape[3]))
        x = self.flat(x)
        x = self.fc(x)

        return x

    def set_output_size(self, output_size):
        self.output_size = output_size
        self.fc._output_size = output_size


    def _freeze(self):
        self.conv1.set_auto_update(self.train_whole)
        self.bn1.set_auto_update(self.train_whole)
        self.layer1.set_auto_update(self.train_whole)
        self.layer2.set_auto_update(self.train_whole)
        self.layer3.set_auto_update(self.train_whole)
        self.layer4.set_auto_update(self.train_whole)




