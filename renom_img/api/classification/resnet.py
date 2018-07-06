import os
import sys
import renom as rm
import numpy as np
from tqdm import tqdm

DIR = os.path.split(os.path.abspath(__file__))[0]
from renom_img.api.utility.misc.download import download
from renom_img.api.model.classification_base import ClassificationBase
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.target import DataBuilderClassification


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


class ResNetBase(ClassificationBase):

    def get_optimizer(self, current_epoch=None, total_epoch=None, current_batch=None, total_batch=None):
        """Returns an instance of Optimiser for training Yolov1 algorithm.

        Args:
            current_epoch:
            total_epoch:
            current_batch:
            total_epoch:
        """
        if any([num is None for num in [current_epoch, total_epoch, current_batch, total_batch]]):
            return self._opt
        else:
            avg_valid_loss_list = kwargs['avg_valid_loss_list']
            if avg_valid_loss[-1] > avg_valid_loss[-2]:
                self._opt._lr = lr / 10.
            return self._opt

class ResNet(ResNetBase):
    def __init__(self, class_map, channels, num_layers, imsize=(224, 224), train_whole_network=False):
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        if type(num_layers) == int:
            num_layers = [num_layers] * len(channels)

        n_class = len(class_map)
        self.num_layers = num_layers
        self.n_class = len(class_map)
        self.channels = channels
        self._train_whole_network = train_whole_network
        self.imsize = imsize
        super(ResNet, self).__init__(class_map)


class CNN_ResNet(rm.Model):
    def __init__(self, n_class, channels, num):
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

        self._freezed_network = rm.Sequential(layers)
        self._network = rm.Dense(n_class)


    def forward(self, x):
        self.freezed_network.set_auto_update(self._train_whole_network)
        index = 0
        t = self.freezed_network[index](x)
        index += 1
        t = rm.relu(self.freezed_network[index](t))  # Batch normalization
        index += 1

        # First block
        for _ in range(self.num_layers[0]):
            tmp = t
            t = self.freezed_network[index](t)
            index += 1
            t = rm.concat([t, tmp])

        # the rest of block
        for num in self.num_layers[1:]:
            for i in range(num):
                if i == 0:
                    t = self.freezed_network[index](t)
                    index += 1
                else:
                    tmp = t
                    t = self.freezed_network[index](t)
                    index += 1
                    t = rm.concat([t, tmp])
        t = rm.flatten(rm.average_pool2d(t))
        t = self.network(t)
        return t

class ResNet32(ResNetBase):
    """ResNet32 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        load_weight(bool):
        class_map: Array of class names
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        6n + 2(The first conv and the last dense) = 32
        → n = 5
        5 sets of a layer block in each block
)
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    """

    WEIGHT_URL = "https://app.box.com/shared/static/o81vwdp4qsm88zt93jvpskqfzobhfx6s.h5"
    WEIGHT_PATH = os.path.join(DIR, 'resnet32.h5')

    def __init__(self, class_map, load_weight=False, imsize=(224, 224), train_whole_network=False):
        num_layers = 5
        CHANNELS = [16, 32, 64]
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        self.n_class = len(class_map)
        self.class_map = class_map
        self.imsize = imsize

        self._model = CNN_ResNet(self.n_class, CHANNELS, num_layers)
        self._opt = rm.Sgd(0.1, 0.9)

        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self._model.load(self.WEIGHT_PATH)
        if n_class != 1000:
            self._model.params = {}


class ResNet44(ResNet):
    """ResNet44 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        load_weight(bool):
        class_map: Array of class names
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    """

    WEIGHT_URL = "https://app.box.com/shared/static/o81vwdp4qsm88zt93jvpskqfzobhfx6s.h5"
    WEIGHT_PATH = os.path.join(DIR, 'resnet44.h5')

    def __init__(self, class_map, load_weight=False, imsize=(224, 224), train_whole_network=False):
        num_layers = 7
        CHANNELS = [16, 32, 64]
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        self.n_class = len(class_map)
        self.class_map = class_map
        self.imsize = imsize

        self._model = CNN_ResNet(self.n_class, CHANNELS, num_layers)
        self._opt = rm.Sgd(0.1, 0.9)

        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self._model.load(self.WEIGHT_PATH)
        if n_class != 1000:
            self._model.params = {}

class ResNet56(ResNet):
    """ResNet56 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        load_weight(bool):
        class_map: Array of class names
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    """

    WEIGHT_URL = "https://app.box.com/shared/static/o81vwdp4qsm88zt93jvpskqfzobhfx6s.h5"
    WEIGHT_PATH = os.path.join(DIR, 'resnet56.h5')

    def __init__(self, class_map, load_weight=False, imsize=(224, 224), train_whole_network=False):
        num_layers = 9
        CHANNELS = [16, 32, 64]
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        self.n_class = len(class_map)
        self.class_map = class_map
        self.imsize = imsize

        self._model = CNN_ResNet(self.n_class, CHANNELS, num_layers)
        self._opt = rm.Sgd(0.1, 0.9)

        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self._model.load(self.WEIGHT_PATH)
        if n_class != 1000:
            self._model.params = {}

class ResNet110(ResNet):
    """ResNet110 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        load_weight(bool):
        class_map: Array of class names
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    """

    WEIGHT_URL = "https://app.box.com/shared/static/o81vwdp4qsm88zt93jvpskqfzobhfx6s.h5"
    WEIGHT_PATH = os.path.join(DIR, 'resnet110.h5')

    def __init__(self, class_map, load_weight=False, imsize=(224, 224), train_whole_network=False):
        num_layers = 18
        CHANNELS = [16, 32, 64]
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        self.n_class = len(class_map)
        self.class_map = class_map
        self.imsize = imsize

        self._model = CNN_ResNet(self.n_class, CHANNELS, num_layers)
        self._opt = rm.Sgd(0.1, 0.9)

        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self._model.load(self.WEIGHT_PATH)

class ResNet34(ResNet):
    """ResNet34 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        load_weight(bool):
        class_map: Array of class names
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    """

    WEIGHT_URL = "https://app.box.com/shared/static/o81vwdp4qsm88zt93jvpskqfzobhfx6s.h5"
    WEIGHT_PATH = os.path.join(DIR, 'resnet34.h5')

    def __init__(self, class_map, load_weight=False, imsize=(224, 224), train_whole_network=False):
        num_layers = [3, 4, 6, 3]
        CHANNELS = [64, 128, 256, 512]
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        self.n_class = len(class_map)
        self.class_map = class_map
        self.imsize = imsize

        self._model = CNN_ResNet(self.n_class, CHANNELS, num_layers)
        self._opt = rm.Sgd(0.1, 0.9)

        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self._model.load(self.WEIGHT_PATH)

class ResNet50(ResNet):
    """ResNet50 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        load_weight(bool):
        class_map: Array of class names
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    """

    WEIGHT_URL = "https://app.box.com/shared/static/o81vwdp4qsm88zt93jvpskqfzobhfx6s.h5"
    WEIGHT_PATH = os.path.join(DIR, 'resnet50.h5')

    def __init__(self, class_map, load_weight=False, imsize=(224, 224), train_whole_network=False):
        num_layers = [3, 4, 6, 3]
        CHANNELS = [64, 128, 256, 512]
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        self.n_class = len(class_map)
        self.class_map = class_map
        self.imsize = imsize

        self._model = CNN_ResNet(self.n_class, CHANNELS, num_layers)
        self._opt = rm.Sgd(0.1, 0.9)

        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self._model.load(self.WEIGHT_PATH)

class ResNet101(ResNet):
    """ResNet101 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        load_weight(bool):
        class_map: Array of class names
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    """

    WEIGHT_URL = "https://app.box.com/shared/static/o81vwdp4qsm88zt93jvpskqfzobhfx6s.h5"
    WEIGHT_PATH = os.path.join(DIR, 'resnet101.h5')

    def __init__(self, class_map, load_weight=False, imsize=(224, 224), train_whole_network=False):
        num_layers = [3, 4, 23, 3]
        CHANNELS = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        self.n_class = len(class_map)
        self.class_map = class_map
        self.imsize = imsize

        self._model = CNN_ResNet(self.n_class, CHANNELS, num_layers)
        self._opt = rm.Sgd(0.1, 0.9)

        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self._model.load(self.WEIGHT_PATH)
