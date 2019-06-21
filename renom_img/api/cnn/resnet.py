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
from renom_img.api.utility.exceptions.exceptions import *




def conv3x3(out_planes, stride=1):
    """3x3 convolution with padding"""
    return rm.Conv2d(out_planes, filter=3, stride=stride, padding=1, ignore_bias=True)


class BasicBlock(rm.Model):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(planes, stride)
        self.bn1 = rm.BatchNormalize(mode='feature')
        self.relu = rm.Relu()
        self.conv2 = conv3x3(planes)
        self.bn2 = rm.BatchNormalize(mode='feature')
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(rm.Model):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = rm.Conv2d(planes, filter=1, ignore_bias=True)
        self.bn1 = rm.BatchNormalize(mode='feature')
        self.conv2 = rm.Conv2d(planes, filter=3, stride=stride, padding=1, ignore_bias=True)
        self.bn2 = rm.BatchNormalize(mode='feature')
        self.conv3 = rm.Conv2d(planes * self.expansion, filter=1, ignore_bias=True)
        self.bn3 = rm.BatchNormalize(mode='feature')
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

class CnnResNet(CnnBase):

    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        super(CnnResNet, self).__init__()
        self.conv1 = rm.Conv2d(64, filter=7, stride=2, padding=3, ignore_bias=True)
        self.bn1 = rm.BatchNormalize(mode='feature')
        self.relu = rm.Relu()
        self.maxpool = rm.MaxPool2d(filter=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.flat = rm.Flatten()
        self.fc = rm.Dense(num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = rm.Sequential([
                rm.Conv2d(planes * block.expansion, filter=1, stride=stride, ignore_bias=True),
                rm.BatchNormalize(mode='feature')
            ])

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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

    def load_pretrained_weight(self,path):
        try:
            self.load(path)
        except:
            raise WeightLoadError('Followng path {} can not be loaded to class {}.'.format(path,self.__class__))


