import os
import sys
import numpy as np
import renom as rm
from tqdm import tqdm

from renom_img import __version__
from renom_img.api import adddoc
from renom_img.api.segmentation import SemanticSegmentation
from renom.utility.initializer import Initializer
from renom.config import precision
from renom.layers.function.pool2d import pool_base, max_pool2d
from renom.layers.function.utils import tuplize
from renom_img.api.cnn import CnnBase
from renom_img.api.utility.exceptions.exceptions import *

class PoolBase(object):

    def __init__(self, filter=3,
                 padding=0, stride=1, ceil_mode=False):
        self._padding, self._stride, self._kernel = (tuplize(x) for x in (padding, stride, filter))
        self._ceil_mode = ceil_mode

    def __call__(self, x):
        return self.forward(x)


class MaxPool2d(PoolBase):
    '''Max pooling function.
    In the case of int input, filter, padding, and stride, the shape will be symmetric.

    Args:
        filter (tuple,int): Filter size of the convolution kernel.
        padding (tuple,int): Size of the zero-padding around the image.
        stride (tuple,int): Stride-size of the convolution.
        ceil_mode (bool): output size is larger (smaller) of two possible sizes when true (false)
                   ceiling mode is used in maxpool2d layers in official Caffe FCN implementation

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>>
        >>> x = np.random.rand(3, 3, 31, 31)
        >>> layer = MaxPool2d(filter=3, stride=2)
        >>> z = layer(x)
        >>> z.shape
        (3, 3, 15, 15)
        >>> layer = MaxPool2d(filter=3, stride=2, ceil_mode=True)
        >>> z = layer(x)
        >>> z.shape
        (3, 3, 16, 16)

    '''

    def forward(self, x):
        return max_pool2d(x, self._kernel, self._stride, self._padding, self._ceil_mode)


def layer_factory(channel=64, conv_layer_num=2, first=None):
    layers = []
    for _ in range(conv_layer_num):
        if first is not None:
            layers.append(rm.Conv2d(channel=channel, padding=100, filter=3))
            layers.append(rm.Relu())
            first = None
        else:
            layers.append(rm.Conv2d(channel=channel, padding=1, filter=3))
            layers.append(rm.Relu())
    layers.append(MaxPool2d(filter=2, stride=2, ceil_mode=True))
    return rm.Sequential(layers)


class DeconvInitializer(Initializer):
    def __init__(self):
        super(DeconvInitializer, self).__init__()

    def __call__(self, shape):
        filter = np.zeros(shape)
        kh, kw = shape[2], shape[3]
        size = kh
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        filter[range(shape[0]), range(shape[0]), :, :] = (1 - abs(og[0] - center) / factor) * \
            (1 - abs(og[1] - center) / factor)
        return filter.astype(precision)

class CNN_FCN8s(CnnBase):

    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/segmentation/FCN8s.h5".format(__version__)
    def __init__(self, num_class):
        super(CNN_FCN8s, self).__init__()
        self.has_bn = False
        self.train_final_upscore=False
        init_deconv = DeconvInitializer()
        self.block1 = layer_factory(channel=64, conv_layer_num=2, first=True)
        self.block2 = layer_factory(channel=128, conv_layer_num=2)
        self.block3 = layer_factory(channel=256, conv_layer_num=3)
        self.block4 = layer_factory(channel=512, conv_layer_num=3)
        self.block5 = layer_factory(channel=512, conv_layer_num=3)

        self.fc6 = rm.Conv2d(4096, filter=7, stride=1, padding=0)
        self.dr1 = rm.Dropout(dropout_ratio=0.5)
        self.fc7 = rm.Conv2d(4096, filter=1, stride=1, padding=0)
        self.dr2 = rm.Dropout(dropout_ratio=0.5)

        self.score_fr = rm.Conv2d(num_class, filter=1, stride=1, padding=0)  # n_classes
        self.score_pool3 = rm.Conv2d(num_class, filter=1, padding=0)
        self.score_pool4 = rm.Conv2d(num_class, filter=1, padding=0)

        self.upscore2 = rm.Deconv2d(num_class, filter=4, stride=2, padding=0,
                                    ignore_bias=True, initializer=init_deconv)  # n_classes
        self.upscore_pool4 = rm.Deconv2d(
            num_class, filter=4, stride=2, padding=0, ignore_bias=True, initializer=init_deconv)
        self.upscore8 = rm.Deconv2d(num_class, filter=16, stride=8,
                                    padding=0, ignore_bias=True, initializer=init_deconv)

    def forward(self, x):
        self._freeze()
        t = x
        t = self.block1(t)
        t = self.block2(t)
        t = self.block3(t)
        pool3 = t
        t = self.block4(t)
        pool4 = t
        t = self.block5(t)

        t = rm.relu(self.fc6(t))
        t = self.dr1(t)
        t = rm.relu(self.fc7(t))
        t = self.dr2(t)

        t = self.score_fr(t)
        t = self.upscore2(t)
        upscore2 = t

        pool4 = 0.01 * pool4
        t = self.score_pool4(pool4)
        score_pool4 = t

        score_pool4c = score_pool4[:, :, 5:5 + upscore2.shape[2], 5:5 + upscore2.shape[3]]
        t = upscore2 + score_pool4c

        fuse_pool4 = t
        t = self.upscore_pool4(fuse_pool4)
        upscore_pool4 = t

        pool3 = 0.0001 * pool3
        t = self.score_pool3(pool3)
        score_pool3 = t

        score_pool3c = score_pool3[:, :, 9:9 + upscore_pool4.shape[2], 9:9 + upscore_pool4.shape[3]]
        t = upscore_pool4 + score_pool3c

        fuse_pool3 = t
        t = self.upscore8(fuse_pool3)
        upscore8 = t

        t = upscore8[:, :, 31:31 + x.shape[2],
                     31:31 + x.shape[3]]
        score = t

        return t

    def set_output_size(self, output_size):
        self.output_size=output_size
        self.score_fr._channel = output_size
        self.score_pool3._channel = output_size
        self.score_pool4._channel = output_size
        self.upscore2._channel = output_size
        self.upscore_pool4._channel = output_size
        self.upscore8._channel = output_size

    def set_train_whole(self,whole,upscore):
        self.train_whole =whole
        self.train_final_upscore=upscore

    def _freeze(self):
        self.block1.set_auto_update(self.train_whole)
        self.block2.set_auto_update(self.train_whole)
        self.block3.set_auto_update(self.train_whole)
        self.block4.set_auto_update(self.train_whole)
        self.block5.set_auto_update(self.train_whole)
        self.upscore8.set_auto_update(self.train_final_upscore)

    def load_pretrained_weight(self,path):
        try:
            self.load(path)
        except:
            raise WeightLoadError('The pretrained weights path {} can not be loaded into the class {}.'.format(path,self.__class__))

class CNN_FCN16s(CnnBase):

    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/segmentation/FCN16s.h5".format(__version__)
    def __init__(self, num_class):
        super(CNN_FCN16s, self).__init__()
        self.has_bn = False
        self.train_final_upscore=False
        init_deconv = DeconvInitializer()
        self.block1 = layer_factory(channel=64, conv_layer_num=2, first=True)
        self.block2 = layer_factory(channel=128, conv_layer_num=2)
        self.block3 = layer_factory(channel=256, conv_layer_num=3)
        self.block4 = layer_factory(channel=512, conv_layer_num=3)
        self.block5 = layer_factory(channel=512, conv_layer_num=3)

        self.fc6 = rm.Conv2d(4096, filter=7, stride=1, padding=0)
        self.dr1 = rm.Dropout(dropout_ratio=0.5)
        self.fc7 = rm.Conv2d(4096, filter=1, stride=1, padding=0)
        self.dr2 = rm.Dropout(dropout_ratio=0.5)

        self.score_fr = rm.Conv2d(num_class, filter=1, stride=1, padding=0)  # n_classes
        self.score_pool4 = rm.Conv2d(num_class, filter=1, padding=0)

        self.upscore2 = rm.Deconv2d(num_class, filter=4, stride=2, padding=0,
                                    ignore_bias=True, initializer=init_deconv)  # n_classes
        self.upscore16 = rm.Deconv2d(num_class, filter=32, stride=16,
                                     padding=0, ignore_bias=True, initializer=init_deconv)  # n_classes

    def forward(self, x):
        self._freeze()
        t = x
        t = self.block1(t)
        t = self.block2(t)
        t = self.block3(t)
        t = self.block4(t)
        pool4 = t
        t = self.block5(t)

        t = rm.relu(self.fc6(t))
        t = self.dr1(t)
        t = rm.relu(self.fc7(t))
        t = self.dr2(t)

        t = self.score_fr(t)

        t = self.upscore2(t)
        upscore2 = t

        pool4 = 0.01 * pool4
        t = self.score_pool4(pool4)
        score_pool4 = t

        score_pool4c = score_pool4[:, :, 5:5 + upscore2.shape[2], 5:5 + upscore2.shape[3]]
        t = upscore2 + score_pool4c
        fuse_pool4 = t
        t = self.upscore16(fuse_pool4)
        upscore16 = t

        t = t[:, :, 27:27 + x.shape[2], 27:27 + x.shape[3]]
        score = t

        return t

    def set_output_size(self, output_size):
        self.output_size = output_size
        self.score_fr._channel = output_size
        self.score_pool4._channel = output_size
        self.upscore2._channel = output_size
        self.upscore16._channel = output_size

    def set_train_whole(self,whole,upscore):
        self.train_whole =whole
        self.train_final_upscore=upscore

    def _freeze(self):
        self.block1.set_auto_update(self.train_whole)
        self.block2.set_auto_update(self.train_whole)
        self.block3.set_auto_update(self.train_whole)
        self.block4.set_auto_update(self.train_whole)
        self.block5.set_auto_update(self.train_whole)
        self.upscore16.set_auto_update(self.train_final_upscore)

    def load_pretrained_weight(self,path):
        try:
            self.load(path)
        except:
            raise WeightLoadError('The pretrained weights path {} can not be loaded into the class {}.'.format(path,self.__class__))

class CNN_FCN32s(CnnBase):
    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/segmentation/FCN32s.h5".format(__version__)
    def __init__(self, num_class):
        super(CNN_FCN32s, self).__init__()
        self.has_bn =False
        self.train_final_upscore=False
        init_deconv = DeconvInitializer()
        self.block1 = layer_factory(channel=64, conv_layer_num=2, first=True)
        self.block2 = layer_factory(channel=128, conv_layer_num=2)
        self.block3 = layer_factory(channel=256, conv_layer_num=3)
        self.block4 = layer_factory(channel=512, conv_layer_num=3)
        self.block5 = layer_factory(channel=512, conv_layer_num=3)

        self.fc6 = rm.Conv2d(4096, filter=7, stride=1, padding=0)
        self.dr1 = rm.Dropout(dropout_ratio=0.5)
        self.fc7 = rm.Conv2d(4096, filter=1, stride=1, padding=0)
        self.dr2 = rm.Dropout(dropout_ratio=0.5)

        self.score_fr = rm.Conv2d(num_class, filter=1, stride=1, padding=0)  # n_classes
        self.upscore = rm.Deconv2d(num_class, filter=64, stride=32, padding=0,
                                   ignore_bias=True, initializer=init_deconv)  # n_classes

    def forward(self, x):
        self._freeze()
        t = x
        t = self.block1(t)
        t = self.block2(t)
        t = self.block3(t)
        t = self.block4(t)
        t = self.block5(t)

        t = rm.relu(self.fc6(t))
        t = self.dr1(t)
        t = rm.relu(self.fc7(t))
        t = self.dr2(t)

        t = self.score_fr(t)
        t = self.upscore(t)

        t = t[:, :, 19:19 + x.shape[2], 19:19 + x.shape[3]]
        score = t

        return t

    def set_output_size(self, output_size):
        self.output_size=output_size
        self.score_fr._channel=output_size
        self.upscore._channel=output_size

    def _freeze(self):
        self.block1.set_auto_update(self.train_whole)
        self.block2.set_auto_update(self.train_whole)
        self.block3.set_auto_update(self.train_whole)
        self.block4.set_auto_update(self.train_whole)
        self.block5.set_auto_update(self.train_whole)
        self.upscore.set_auto_update(self.train_final_upscore)

    def set_train_whole(self,whole,upscore):
        self.train_whole =whole
        self.train_final_upscore=upscore

    def load_pretrained_weight(self,path):
        try:
            self.load(path)
        except:
            raise WeightLoadError('The pretrained weights path {} can not be loaded into the class {}.'.format(path,self.__class__))
