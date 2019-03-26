import os
import sys
import numpy as np
import renom as rm

from renom_img import __version__
from renom_img.api.utility.misc.download import download


class Darknet(rm.Sequential):
    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/classification/Darknet.h5".format(
        __version__)

    def __init__(self, num_class=1000, load_pretrained_weight=False):

        super(Darknet, self).__init__([
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
            rm.Conv2d(channel=512, filter=3, padding=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=256, filter=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=3, padding=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, padding=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.MaxPool2d(stride=2, filter=2),

            # 5th Block
            rm.Conv2d(channel=512, filter=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, padding=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, padding=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),

            # layers for ImageNet
            rm.Conv2d(channel=num_class, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.AveragePool2d(filter=7),
            rm.Softmax()
        ])
        if load_pretrained_weight:
            if isinstance(load_pretrained_weight, bool):
                load_pretrained_weight = self.__class__.__name__ + '.h5'

            if not os.path.exists(load_pretrained_weight):
                download(self.WEIGHT_URL, load_pretrained_weight)

            self.load(load_pretrained_weight)

# Darknet19


class DarknetConv2dBN(rm.Model):

    def __init__(self, channel, filter=3, prev_ch=None):
        pad = int((filter - 1) / 2)
        if prev_ch is not None:
            self._conv = rm.Conv2d(channel=channel, filter=filter, padding=pad)
            self._conv.params = {
                "w": rm.Variable(self._conv._initializer((channel, prev_ch, filter, filter)), auto_update=True),
                "b": rm.Variable(np.zeros((1, channel, 1, 1), dtype=np.float32), auto_update=False),
            }
            self._bn = rm.BatchNormalize(mode='feature', momentum=0.99)
        else:
            self._conv = rm.Conv2d(channel=channel, filter=filter, padding=pad)
            self._bn = rm.BatchNormalize(mode='feature', momentum=0.99)

    def forward(self, x):
        return rm.leaky_relu(self._bn(self._conv(x)), 0.1)


class Darknet19Base(rm.Model):

    def __init__(self):
        self.block1 = rm.Sequential([
            DarknetConv2dBN(32, prev_ch=3),
            rm.MaxPool2d(filter=2, stride=2)
        ])
        self.block2 = rm.Sequential([
            DarknetConv2dBN(64, prev_ch=32),
            rm.MaxPool2d(filter=2, stride=2)
        ])
        self.block3 = rm.Sequential([
            DarknetConv2dBN(128, prev_ch=64),
            DarknetConv2dBN(64, filter=1, prev_ch=128),
            DarknetConv2dBN(128, prev_ch=64),
            rm.MaxPool2d(filter=2, stride=2)
        ])
        self.block4 = rm.Sequential([
            DarknetConv2dBN(256, prev_ch=128),
            DarknetConv2dBN(128, filter=1, prev_ch=256),
            DarknetConv2dBN(256, prev_ch=128),
            rm.MaxPool2d(filter=2, stride=2)
        ])
        self.block5 = rm.Sequential([
            DarknetConv2dBN(512, prev_ch=256),
            DarknetConv2dBN(256, filter=1, prev_ch=512),
            DarknetConv2dBN(512, prev_ch=256),
            DarknetConv2dBN(256, filter=1, prev_ch=512),
            DarknetConv2dBN(512, prev_ch=256),
        ])
        self.block6 = rm.Sequential([
            # For concatenation.
            rm.MaxPool2d(filter=2, stride=2),
            DarknetConv2dBN(1024, prev_ch=512),
            DarknetConv2dBN(512, filter=1, prev_ch=1024),
            DarknetConv2dBN(1024, prev_ch=512),
            DarknetConv2dBN(512, filter=1, prev_ch=1024),
            DarknetConv2dBN(1024, prev_ch=512),
        ])

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        f = self.block5(h)
        h = self.block6(f)
        return h, f


class Darknet19(rm.Model):

    WEIGHT_URL = "https://docs.renom.jp/downloads/weights/Darknet/Darknet19.h5"

    def __init__(self, num_class=1000, load_pretrained_weight=False):
        self._num_class = num_class
        self._base = Darknet19Base()
        self._last = rm.Conv2d(num_class, filter=1)
        self._last.params = {
            "w": rm.Variable(self._last._initializer((num_class, 1024, 1, 1)), auto_update=True),
            "b": rm.Variable(self._last._initializer((1, num_class, 1, 1)), auto_update=False),
        }
        super(Darknet19, self).__init__()

        if load_pretrained_weight:
            if isinstance(load_pretrained_weight, bool):
                load_pretrained_weight = self.__class__.__name__ + '.h5'

            if not os.path.exists(load_pretrained_weight):
                download(self.WEIGHT_URL, load_pretrained_weight)

            self.load(load_pretrained_weight)

    def forward(self, x):
        assert self._num_class > 0, \
            "Class map is empty. Please set the attribute class_map when instantiating a model. " +\
            "Or, please load a pre-trained model using the ‘load()’ method."
        N = len(x)
        h, _ = self._base(x)
        D = h.shape[2] * h.shape[3]
        h = rm.sum(self._last(h).reshape(N, self._num_class, -1), axis=2)
        h /= D
        return h
