import os
import sys
import numpy as np
import renom as rm
from tqdm import tqdm
import inspect

from renom.layers.function.utils import col2im, im2col, tuplize
from renom.core import Node, Variable, to_value
from renom import precision
from renom.layers.function.parameterized import Parametrized
from renom_img import __version__
from renom_img.api import adddoc
from renom_img.api.utility.misc.download import download
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.load import load_img
from renom_img.api.utility.target import DataBuilderSegmentation
from renom_img.api.segmentation import SemanticSegmentation
from renom.utility.initializer import GlorotNormal, GlorotUniform
from renom_img.api.cnn import CnnBase

import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import GPUValue, get_gpu

def transpose_out_size(size, k, s, p, d=(1, 1), ceil_mode=False):
    if ceil_mode:
        return (np.array(s) * (np.array(size) - 1) + np.array(k) + (np.array(k) - 1) *
                (np.array(d) - 1) - 2 * np.array(p) + 1).astype(np.int)
    return (np.array(s) * (np.array(size) - 1) + np.array(k) + (np.array(k) - 1) *
            (np.array(d) - 1) - 2 * np.array(p)).astype(np.int)


class deconv2d(Node):

    def __new__(cls, x, w, b, filter=3, stride=1, padding=0, dilation=1, ceil_mode=False):
        filter, stride, padding, dilation = (tuplize(x)
                                             for x in (filter, stride, padding, dilation))

        in_shape = x.shape[1:]
        out_shape = [w.shape[1], ]
        out_shape.extend(transpose_out_size(
            in_shape[1:], filter, stride, padding, dilation, ceil_mode))
        return cls.calc_value(x, w, b, in_shape, out_shape, filter, stride, padding, dilation)

    @classmethod
    def _oper_cpu(cls, x, w, b, in_shape, out_shape, kernel, stride, padding, dilation):
        z = np.tensordot(w, x, (0, 1))
        z = np.rollaxis(z, 3)
        z = col2im(z, out_shape[1:], stride, padding, dilation)
        if b is not None:
            z += b
        ret = cls._create_node(z)
        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._b = b
        ret.attrs._in_shape = in_shape
        ret.attrs._kernel = kernel
        ret.attrs._stride = stride
        ret.attrs._padding = padding
        ret.attrs._dilation = dilation
        return ret

    @classmethod
    def _oper_gpu(cls, x, w, b, in_shape, out_shape, kernel, stride, padding, dilation):
        conv_desc = cu.ConvolutionDescriptor(padding, stride, dilation, precision)
        filter_desc = cu.FilterDescriptor(w.shape, precision)
        N = x.shape[0]
        z = GPUValue(shape=tuple([N, ] + list(out_shape)))

        with cu.cudnn_handler() as handle:
            cu.cuConvolutionBackwardData(handle, conv_desc, filter_desc, get_gpu(w), get_gpu(x), z)
        if b is not None:
            cu.cu_add_bias(get_gpu(b), z)

        ret = cls._create_node(z)
        ret.attrs._conv_desc = conv_desc
        ret.attrs._filter_desc = filter_desc
        ret.attrs._x = x
        ret.attrs._w = w
        ret.attrs._b = b
        return ret

    def _backward_cpu(self, context, dy, **kwargs):

        col = im2col(dy, self.attrs._in_shape[1:], self.attrs._kernel,
                     self.attrs._stride, self.attrs._padding, self.attrs._dilation)

        if isinstance(self.attrs._x, Node):
            dx = np.tensordot(col, self.attrs._w, ([1, 2, 3], [1, 2, 3]))
            dx = np.rollaxis(dx, 3, 1)
            self.attrs._x._update_diff(context, dx, **kwargs)

        if isinstance(self.attrs._w, Node):
            self.attrs._w._update_diff(context, np.tensordot(
                self.attrs._x, col, ([0, 2, 3], [0, 4, 5])), **kwargs)

        if isinstance(self.attrs._b, Node):
            self.attrs._b._update_diff(context, np.sum(dy, (0, 2, 3), keepdims=True), **kwargs)

    def _backward_gpu(self, context, dy, **kwargs):
        dw, db, dx = (get_gpu(g).empty_like_me() if g is not None else None
                      for g in (self.attrs._w, self.attrs._b, self.attrs._x))
        with cu.cudnn_handler() as handle:
            cu.cuConvolutionForward(handle, self.attrs._conv_desc,
                                    self.attrs._filter_desc, get_gpu(dy), get_gpu(self.attrs._w), dx)
            cu.cuConvolutionBackwardFilter(handle, self.attrs._conv_desc,
                                           self.attrs._filter_desc, get_gpu(dy), get_gpu(self.attrs._x), dw)
            if db is not None:
                cu.cuConvolutionBackwardBias(handle, get_gpu(dy), db)

        if isinstance(self.attrs._x, Node):
            self.attrs._x._update_diff(context, dx, **kwargs)

        if isinstance(self.attrs._w, Node):
            self.attrs._w._update_diff(context, dw, **kwargs)

        if isinstance(self.attrs._b, Node):
            self.attrs._b._update_diff(context, db, **kwargs)


class Deconv2d(Parametrized):
    '''2d convolution layer.

    This class creates a convolution filter to be convolved with
    the input tensor.
    The instance of this class only accepts and outputs 4d tensors.

    At instantiation, in the case of int input, filter, padding, and stride, the shape will be symmetric.

    If the argument `input_size` is passed, this layers' weight is initialized
    in the __init__ function.
    Otherwise, the weight is initialized in its first forward calculation.

    Args:
        channel (int): The dimensionality of the output.
        filter (tuple,int): Filter size to witch used as convolution kernel.
        padding (tuple,int): Pad around image by 0 according to this size.
        stride (tuple,int): Specifying the strides of the convolution.
        dilation (tuple, int): Dilation of the convolution.
        input_size (tuple): Input unit size. This must be a tuple like (Channel, Height, Width).
        ignore_bias (bool): If True is given, bias will not be added.
        initializer (Initializer): Initializer object for weight initialization.
        ceil_mode (bool): If True, choose larger output shape when two shapes are possible

    Example:
        >>> import numpy as np
        >>> import renom as rm
        >>> n, c, h, w = (10, 3, 32, 32)
        >>> x = np.random.rand(n, c, h, w)
        >>> x.shape
        (10, 3, 32, 32)
        >>> layer = rm.Deconv2d(channel=32)
        >>> z = layer(x)
        >>> z.shape
        (10, 32, 34, 34)

    '''

    def __init__(self,
                 channel=1,
                 filter=3,
                 padding=0,
                 stride=1,
                 dilation=1,
                 input_size=None,
                 ignore_bias=False,
                 initializer=GlorotNormal(),
                 weight_decay=0,
                 ceil_mode=False):

        self._padding, self._stride, self._kernel, self._dilation = (tuplize(x)
                                                                     for x in (padding, stride, filter, dilation))
        self._channel = channel
        self._initializer = initializer
        self._ignore_bias = ignore_bias
        self._weight_decay = weight_decay
        self._ceil_mode = ceil_mode
        super(Deconv2d, self).__init__(input_size)

    def weight_initiallize(self, input_size):
        size_f = (input_size[0], self._channel,
                  self._kernel[0], self._kernel[1])
        self.params = {"w": Variable(self._initializer(
            size_f), auto_update=True, weight_decay=self._weight_decay)}
        if not self._ignore_bias:
            self.params["b"] = Variable(
                np.zeros((1, self._channel, 1, 1), dtype=precision), auto_update=True)

    def forward(self, x):
        return deconv2d(x, self.params["w"], self.params.get("b"),
                        self._kernel, self._stride, self._padding, self._dilation, self._ceil_mode)


def layer_factory(channel_list=[64]):
    layers = []
    for i in range(len(channel_list)):
        layers.append(rm.Conv2d(channel=channel_list[i],
                                padding=1, filter=3, initializer=GlorotUniform()))
        layers.append(rm.Relu())
    return rm.Sequential(layers)


def layer_factory_deconv(channel_list=[512, 256]):
    layers = []
    layers.append(rm.Conv2d(channel=channel_list[0],
                            padding=1, filter=3, initializer=GlorotUniform()))
    layers.append(rm.Relu())
    if 'ceil_mode' in inspect.signature(rm.Deconv2d).parameters:
        layers.append(rm.Deconv2d(
            channel=channel_list[1], padding=1, filter=3, stride=2, initializer=GlorotUniform(), ceil_mode=True))
    else:
        layers.append(Deconv2d(
            channel=channel_list[1], padding=1, filter=3, stride=2, initializer=GlorotUniform(), ceil_mode=True))
    layers.append(rm.Relu())
    return rm.Sequential(layers)

class CNN_TernausNet(CnnBase):

    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/segmentation/TernausNet.h5".format(__version__)

    def __init__(self, num_class):
        super(CNN_TernausNet, self).__init__()
        self.block1 = layer_factory(channel_list=[64])
        self.block2 = layer_factory(channel_list=[128])
        self.block3 = layer_factory(channel_list=[256, 256])
        self.block4 = layer_factory(channel_list=[512, 512])
        self.block5 = layer_factory(channel_list=[512, 512])

        self.center = layer_factory_deconv(channel_list=[512, 256])

        self.decoder_block5 = layer_factory_deconv(channel_list=[512, 256])
        self.decoder_block4 = layer_factory_deconv(channel_list=[512, 128])
        self.decoder_block3 = layer_factory_deconv(channel_list=[256, 64])
        self.decoder_block2 = layer_factory_deconv(channel_list=[128, 32])
        self.decoder_block1 = layer_factory(channel_list=[32])

        self.final = rm.Conv2d(channel=num_class, filter=1, stride=1)

    def forward(self, x):
        self._freeze()
        c1 = self.block1(x)
        t = rm.max_pool2d(c1, filter=2, stride=2)
        c2 = self.block2(t)
        t = rm.max_pool2d(c2, filter=2, stride=2)
        c3 = self.block3(t)
        t = rm.max_pool2d(c3, filter=2, stride=2)
        c4 = self.block4(t)
        t = rm.max_pool2d(c4, filter=2, stride=2)
        c5 = self.block5(t)
        t = rm.max_pool2d(c5, filter=2, stride=2)

        t = self.center(t)
        t = rm.concat([t, c5])
        t = self.decoder_block5(t)
        t = rm.concat([t, c4])
        t = self.decoder_block4(t)
        t = rm.concat([t, c3])
        t = self.decoder_block3(t)
        t = rm.concat([t, c2])
        t = self.decoder_block2(t)
        t = rm.concat([t, c1])
        t = self.decoder_block1(t)

        t = self.final(t)

        return t

    def _freeze(self):
        self.block1.set_auto_update(self.train_whole)
        self.block2.set_auto_update(self.train_whole)
        self.block3.set_auto_update(self.train_whole)
        self.block4.set_auto_update(self.train_whole)
        self.block5.set_auto_update(self.train_whole)

    def set_output_size(self, output_size):
        self.output_size = output_size
        self.final._channel = output_size

    def load_pretrained_weight(self, path):
        self.load(path)

