import os
import sys
import numpy as np
import renom as rm
from renom_img import __version__

from .base import CnnBase


class CnnDarknet(CnnBase):

    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/classification/Darknet.h5".format(
        __version__)

    def __init__(self):
        self.feature_extractor = rm.Sequential([
            # 1st Block
            rm.Conv2d(channel=64, filter=7, stride=2, padding=3, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.MaxPool2d(stride=2, filter=2),

            # 2nd Block
            rm.Conv2d(channel=192, filter=3, padding=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.MaxPool2d(stride=2, filter=2),

            # 3rd Block
            rm.Conv2d(channel=128, filter=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=256, filter=3, padding=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=256, filter=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=3, padding=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.MaxPool2d(stride=2, filter=2),

            # 4th Block
            rm.Conv2d(channel=256, filter=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=3, padding=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=256, filter=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=3, padding=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=256, filter=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),

            # layers for ImageNet

            rm.Conv2d(channel=num_class, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.AveragePool2d(filter=7),
            rm.Softmax()
        ])
