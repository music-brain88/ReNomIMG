import os
import sys
import numpy as np
import renom as rm

from renom_img import __version__
from renom_img.api import adddoc
from renom_img.api.segmentation import SemanticSegmentation
from renom_img.api.cnn import CnnBase
from renom.config import precision
from renom.utility.initializer import Initializer, GlorotNormal, GlorotUniform
from renom.layers.function.utils import col2im, im2col, tuplize
from renom.layers.function.parameterized import Parametrized
from renom.core import Node, Variable, to_value
from renom import precision
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
            size_f), auto_update=False, weight_decay=self._weight_decay)}
        if not self._ignore_bias:
            self.params["b"] = Variable(
                np.zeros((1, self._channel, 1, 1), dtype=precision), auto_update=False)

    def forward(self, x):
        return deconv2d(x, self.params["w"], self.params.get("b"),
                        self._kernel, self._stride, self._padding, self._dilation, self._ceil_mode)


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


class XceptionBlock(rm.Model):
    def __init__(self, channels, stride=1, padding=1, dilation=1, downsample=None, residual=False):
        self.downsample=downsample
        self.residual=residual

        self.conv1_depth=rm.GroupConv2d(channel=channels[0], filter=3, stride=1, padding=padding, dilation=dilation, groups=channels[0], ignore_bias=True)
        self.bn1_depth = rm.BatchNormalize(mode='feature', epsilon=1e-3)
        self.conv1_point=rm.Conv2d(channel=channels[1], filter=1, ignore_bias=True)
        self.bn1_point = rm.BatchNormalize(mode='feature', epsilon=1e-3)
        
        self.conv2_depth=rm.GroupConv2d(channel=channels[1], filter=3, stride=1, padding=padding, dilation=dilation, groups=channels[1], ignore_bias=True)
        self.bn2_depth = rm.BatchNormalize(mode='feature', epsilon=1e-3)
        self.conv2_point=rm.Conv2d(channel=channels[2], filter=1, ignore_bias=True)
        self.bn2_point = rm.BatchNormalize(mode='feature', epsilon=1e-3)
        
        self.conv3_depth=rm.GroupConv2d(channel=channels[2], filter=3, stride=stride, padding=padding, dilation=dilation, groups=channels[2], ignore_bias=True)
        self.bn3_depth = rm.BatchNormalize(mode='feature', epsilon=1e-3)
        self.conv3_point=rm.Conv2d(channel=channels[3], filter=1, ignore_bias=True)
        self.bn3_point = rm.BatchNormalize(mode='feature', epsilon=1e-3)
        self.relu=rm.Relu()
            
    def forward(self, x):
        if self.residual:
            residual = x
            out = self.relu(x)
        else:
            out = x
        out = self.conv1_depth(out)
        out = self.bn1_depth(out)
        if not self.residual:
            out = self.relu(out)
        out = self.conv1_point(out)
        out = self.bn1_point(out)
        out = self.relu(out)
        out = self.conv2_depth(out)
        out = self.bn2_depth(out)
        if not self.residual:
            out = self.relu(out)
        out = self.conv2_point(out)
        out = self.bn2_point(out)
        out = self.relu(out)
        out = self.conv3_depth(out)
        out = self.bn3_depth(out)
        if not self.residual:
            out = self.relu(out)
        out = self.conv3_point(out)
        out = self.bn3_point(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        if self.residual:
            out += residual
        else:
            out = self.relu(out)

        return out

class AsppModule(rm.Model):
    def __init__(self, num_class, filter_size=(33,33), atrous_rates=[6,12,18]):
        
        # aspp0
        self.aspp0_conv = rm.Conv2d(channel=256, filter=1, ignore_bias=True)
        self.aspp0_bn = rm.BatchNormalize(mode='feature', epsilon=1e-5) 
        # aspp1
        self.aspp1_conv = rm.Conv2d(channel=256, filter=3, stride=1, padding=atrous_rates[0], dilation=atrous_rates[0], ignore_bias=True)
        self.aspp1_bn = rm.BatchNormalize(mode='feature', epsilon=1e-5)
        # aspp2
        self.aspp2_conv = rm.Conv2d(channel=256, filter=3, stride=1, padding=atrous_rates[1], dilation=atrous_rates[1], ignore_bias=True)
        self.aspp2_bn = rm.BatchNormalize(mode='feature', epsilon=1e-5) 
        # aspp3
        self.aspp3_conv = rm.Conv2d(channel=256, filter=3, stride=1, padding=atrous_rates[2], dilation=atrous_rates[2], ignore_bias=True)
        self.aspp3_bn = rm.BatchNormalize(mode='feature', epsilon=1e-5)        
        # Image Pooling
        self.avg_pool = rm.AveragePool2d(filter=filter_size)
        self.image_pool_conv = rm.Conv2d(channel=256, filter=1, ignore_bias=True)
        self.image_pool_bn = rm.BatchNormalize(mode='feature', epsilon=1e-5)
                
        self.image_pool_resize = rm.GroupConv2d(channel=256, filter=filter_size[0], stride=1, padding=int(filter_size[0]-1), ignore_bias=True, initializer=np.ones, groups=256)   
        
        # Common
        self.relu = rm.Relu()
    
    def forward(self, x):
        aspp0 = self.aspp0_conv(x)
        aspp0 = self.aspp0_bn(aspp0)
        aspp0 = self.relu(aspp0)

        aspp1 = self.aspp1_conv(x)
        aspp1 = self.aspp1_bn(aspp1)
        aspp1 = self.relu(aspp1)

        aspp2 = self.aspp2_conv(x)
        aspp2 = self.aspp2_bn(aspp2)
        aspp2 = self.relu(aspp2)

        aspp3 = self.aspp3_conv(x)
        aspp3 = self.aspp3_bn(aspp3)
        aspp3 = self.relu(aspp3)

        image_pool = self.avg_pool(x)
        image_pool = self.image_pool_conv(image_pool)
        image_pool = self.image_pool_bn(image_pool)
        image_pool = self.relu(image_pool)
        image_pool = self.image_pool_resize(image_pool)

        x = rm.concat(image_pool, aspp0, aspp1, aspp2, aspp3)
        
        return x


class CnnDeeplabv3plus(CnnBase):
            
    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/segmentation/Deeplabv3plus.h5".format(__version__)

    def __init__(self, num_class, imsize=(513,513), scale_factor=16, atrous_rates=[6,12,18], decoder=False):

        super(CnnDeeplabv3plus, self).__init__()
        
        self.conv1_1 = rm.Conv2d(32, filter=3, stride=2, padding=1, ignore_bias=True)
        self.bn1_1 = rm.BatchNormalize(mode='feature', epsilon=1e-3)
        self.conv1_2 = rm.Conv2d(64, filter=3, stride=1, padding=1, ignore_bias=True)
        self.bn1_2 = rm.BatchNormalize(mode='feature', epsilon=1e-3)
        self.relu = rm.Relu()
        self.dilation = int(32/scale_factor)
        
        self.flow1 = self._make_flow(XceptionBlock, units=3, channels=[64,128,256,728], stride=2, dilation =1, type='Entry')
        self.flow2 = self._make_flow(XceptionBlock, units=16, channels=[728, 728, 728, 728], stride=1, dilation = 1, type='Middle')
        self.flow3 = self._make_flow(XceptionBlock, units=2, channels=[728,1024,1536,2048], stride=1, dilation = 1, type='Exit')
        self.avg_pool_filter = (int(np.ceil(imsize[0]/scale_factor)), int(np.ceil(imsize[1]/scale_factor))) 
        self.aspp = AsppModule(num_class, self.avg_pool_filter, atrous_rates)
        
        self.concat_conv = rm.Conv2d(channel=256, filter=1, ignore_bias=True)
        self.concat_bn = rm.BatchNormalize(mode='feature', epsilon=1e-5)
        self.dropout = rm.Dropout(dropout_ratio=0.1)
        self.final_conv = rm.Conv2d(channel=num_class, filter=1, ignore_bias=False)
        self.final_resize = Deconv2d(num_class, filter=32, stride=16, padding=16, ignore_bias=True, initializer=DeconvInitializer(), ceil_mode=True)
        
    def _make_flow(self, block, units, channels, stride=1, padding=1, dilation=1, type=None):
        layers = []
        
        for i in range(units):
            downsample = None
            residual=True
            
            if type == 'Entry':
                c = list([channels[i], channels[i+1], channels[i+1], channels[i+1]])
                downsample = rm.Sequential([
                    rm.Conv2d(c[-1], filter=1, stride=stride, ignore_bias=True),
                    rm.BatchNormalize(epsilon=1e-3, mode='feature')])
                layers.append(block(c, stride, padding, dilation, downsample, residual))
                              
            elif type == 'Middle':
                c = channels
                layers.append(block(c, stride, padding, dilation, downsample, residual))
                              
            elif type == 'Exit':
                if i==0:
                    c = list([channels[i], channels[i], channels[i+1], channels[i+1]])
                    downsample = rm.Sequential([
                        rm.Conv2d(c[-1], filter=1, stride=stride, ignore_bias=True),
                        rm.BatchNormalize(epsilon=1e-3, mode='feature')])
                else:
                    c = list([channels[i], channels[i+1], channels[i+1], channels[i+2]])
                    residual=False
                    padding = self.dilation
                    dilation = self.dilation
                layers.append(block(c, stride, padding, dilation, downsample, residual))
                              
        return rm.Sequential(layers)

    
    def forward(self, x):
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu(x)

        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu(x)

        x = self.flow1(x)
        x = self.flow2(x)
        x = self.flow3(x)

        x = self.aspp(x)
        
        x = self.concat_conv(x)
        x = self.concat_bn(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.final_conv(x)
        x = self.final_resize(x)
        self._freeze()
        return x

    def set_output_size(self, output_size):
        self.output_size = output_size

    def _freeze(self):
        self.conv1_1.set_auto_update(self.train_whole)
        self.bn1_1.set_auto_update(self.train_whole)
        self.conv1_2.set_auto_update(self.train_whole)
        self.bn1_2.set_auto_update(self.train_whole)
        self.flow1.set_auto_update(self.train_whole)
        self.flow2.set_auto_update(self.train_whole)
        self.flow3.set_auto_update(self.train_whole)
        self.aspp.image_pool_resize.params.w._auto_update = False
        self.final_resize.params.w._auto_update = False

    def _freeze_bn(self):
        for l in self.iter_models():
           if 'BatchNormalize' in l.__class__.__name__ and l._epsilon == 0.001: 
               l.set_models(inference=True)
               l.params.w._auto_update = False
               l.params.b._auto_update = False

    def load_pretrained_weight(self, path):
        self.load(path)
