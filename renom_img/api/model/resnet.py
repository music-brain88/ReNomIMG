import os
import sys
import renom as rm
import numpy as np
from renom_img.api.utility.misc.download import download

DIR = os.path.split(os.path.abspath(__file__))[0]


def layer_block(channel, filter):
    layers = []
    if filter != (1, 1):
        layers.append(rm.Conv2d(filter=filter, channel=channel, padding=1))
    else:
        layers.append(rm.Conv2d(filter=filter, channel=channel))
    layers.append(rm.BatchNormalize(epsilon=0.001, mode='feature'))
    layers.append(rm.Relu())
    return layers


def downsample_block(channel, filter):
    return [
        rm.Conv2d(filter=filter, channel=channel, padding=1, stride=2),
        rm.BatchNormalize(epsilon=0.001, mode='feature'),
        rm.Relu(),
    ]


def build_block(channels):
    """
    A block without down-sampling (stride == 1)
    """
    layers = []
    if type(channels) == int:
        layers.extend(layer_block(channels, (3, 3)))
        layers.extend(layer_block(channels, (3, 3)))
    else:
        layers.extend(layer_block(channels[0], (1, 1)))
        layers.extend(layer_block(channels[1], (3, 3)))
        layers.extend(layer_block(channels[2], (1, 1)))
    return rm.Sequential(layers)


def build_downsample_block(channels):
    """
    A block including down-sample process
    """
    layers = []
    if type(channels) == int:
        layers.extend(downsample_block(channels, (3, 3)))
        layers.extend(layer_block(channels, (3, 3)))
    else:
        layers.extend(downsample_block(channels[0], (1, 1)))
        layers.extend(layer_block(channels[1], (3, 3)))
        layers.extend(layer_block(channels[2], (1, 1)))
    return rm.Sequential(layers)


class ResNet(rm.Sequential):
    def __init__(self, n_class, channels, num_layers):
        if type(num_layers) == int:
            num_layers = [num_layers] * len(channels)
        self.num_layers = num_layers
        self.channels = channels
        layers = []
        layers.append(rm.Conv2d(channel=16, padding=1))
        layers.append(rm.BatchNormalize(epsilon=0.001, mode='feature'))

        # First block which doesn't have down-sampling
        for _ in range(num_layers[0]):
            layers.append(build_block(channels[0]))

        # The rest of blocks which has down-sampling layer
        for i, num in enumerate(num_layers[1:]):
            for j in range(num):
                if j == 0:
                    layers.append(build_downsample_block(channels[i + 1]))
                else:
                    layers.append(build_block(channels[i + 1]))

        # Add the last dense layer
        layers.append(rm.Dense(n_class))
        super(ResNet, self).__init__(layers)

    def forward(self, x):
        index = 0
        t = self._layers[index](x)
        index += 1
        t = rm.relu(self._layers[index](t))  # Batch normalization
        index += 1

        # First block
        for _ in range(self.num_layers[0]):
            tmp = t
            t = self._layers[index](t)
            index += 1
            t = rm.concat([t, tmp])

        # the rest of block
        for num in self.num_layers[1:]:
            for i in range(num):
                if i == 0:
                    t = self._layers[index](t)
                    index += 1
                else:
                    tmp = t
                    t = self._layers[index](t)
                    index += 1
                    t = rm.concat([t, tmp])
        t = rm.flatten(rm.average_pool2d(t))
        t = self._layers[index](t)
        return t


class ResNet32(ResNet):
    """ResNet32 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        n_class(int):
        load_weight(bool):

    Note:
        6n + 2(The first conv and the last dense) = 32
        → n = 5
        5 sets of a layer block in each block

        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    """

    WEIGHT_URL = "https://app.box.com/shared/static/o81vwdp4qsm88zt93jvpskqfzobhfx6s.h5"
    WEIGHT_PATH = os.path.join(DIR, 'resnet32.h5')

    def __init__(self, n_class, load_weight=False):
        num_layers = 5
        CHANNELS = [16, 32, 64]
        super(ResNet32, self).__init__(n_class, CHANNELS, num_layers)
        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if n_class != 1000:
            self._layers[-1].params = {}


class ResNet44(ResNet):
    """ResNet44 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        n_class(int):
        load_weight(bool):

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    """

    WEIGHT_URL = "https://app.box.com/shared/static/o81vwdp4qsm88zt93jvpskqfzobhfx6s.h5"
    WEIGHT_PATH = os.path.join(DIR, 'resnet44.h5')

    def __init__(self, n_class, load_weight=False):
        num_layers = 7
        CHANNELS = [16, 32, 64]
        super(ResNet44, self).__init__(n_class, CHANNELS, num_layers)

        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if n_class != 1000:
            self._layers[-1].params = {}

class ResNet56(ResNet):
    """ResNet56 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        n_class(int):
        load_weight(bool):

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    """

    WEIGHT_URL = "https://app.box.com/shared/static/o81vwdp4qsm88zt93jvpskqfzobhfx6s.h5"
    WEIGHT_PATH = os.path.join(DIR, 'resnet56.h5')
    def __init__(self, n_class, load_weight=False):
        num_layers = 9
        CHANNELS = [16, 32, 64]
        super(ResNet56, self).__init__(n_class, CHANNELS, num_layers)

        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if n_class != 1000:
            self._layers[-1].params = {}


class ResNet110(ResNet):
    """ResNet110 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        n_class(int):
        load_weight(bool):

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    """

    WEIGHT_URL = "https://app.box.com/shared/static/o81vwdp4qsm88zt93jvpskqfzobhfx6s.h5"
    WEIGHT_PATH = os.path.join(DIR, 'resnet110.h5')
    def __init__(self, n_class, load_weight=False):
        num_layers = 18
        CHANNELS = [16, 32, 64]
        super(ResNet110, self).__init__(n_class, CHANNELS, num_layers)

        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if n_class != 1000:
            self._layers[-1].params = {}


class ResNet34(ResNet):
    """ResNet34 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        n_class(int):
        load_weight(bool):

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    """

    WEIGHT_URL = "https://app.box.com/shared/static/o81vwdp4qsm88zt93jvpskqfzobhfx6s.h5"
    WEIGHT_PATH = os.path.join(DIR, 'resnet34.h5')
    def __init__(self, n_class, load_weight=False):
        num_layers = [3, 4, 6, 3]
        CHANNELS = [64, 128, 256, 512]
        super(ResNet34, self).__init__(n_class, CHANNELS, num_layers)

        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if n_class != 1000:
            self._layers[-1].params = {}


class ResNet50(ResNet):
    """ResNet50 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        n_class(int):
        load_weight(bool):

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    """

    WEIGHT_URL = "https://app.box.com/shared/static/o81vwdp4qsm88zt93jvpskqfzobhfx6s.h5"
    WEIGHT_PATH = os.path.join(DIR, 'resnet50.h5')
    def __init__(self, n_class, load_weight=False):
        num_layers = [3, 4, 6, 3]
        CHANNELS = [64, 128, 256, 512]
        super(ResNet50, self).__init__(n_class, CHANNELS, num_layers)

        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if n_class != 1000:
            self._layers[-1].params = {}

class ResNet101(ResNet):
    """ResNet101 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        n_class(int):
        load_weight(bool):

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    """

    WEIGHT_URL = "https://app.box.com/shared/static/o81vwdp4qsm88zt93jvpskqfzobhfx6s.h5"
    WEIGHT_PATH = os.path.join(DIR, 'resnet101.h5')
    def __init__(self, n_class, load_weight=False):
        num_layers = [3, 4, 23, 3]
        CHANNELS = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]
        super(ResNet101, self).__init__(n_class, CHANNELS, num_layers)

        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if n_class != 1000:
            self._layers[-1].params = {}
