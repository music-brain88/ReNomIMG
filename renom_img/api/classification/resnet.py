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

DIR = os.path.split(os.path.abspath(__file__))[0]


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


class ResNetBase(Classification):

    SERIALIZED = Base.SERIALIZED

    def get_optimizer(self, current_loss=None, current_epoch=None, total_epoch=None,
                      current_batch=None, total_batch=None, avg_valid_loss_list=None):
        """Returns an optimizer instance for training ResNet algorithm.

        Args:
            current_epoch:
            total_epoch:
            current_batch:
            total_epoch:
        """
        if any([num is None for num in [current_epoch, total_epoch, current_batch, total_batch, avg_valid_loss_list]]):
            return self._opt
        elif self.plateau:
            avg_valid_loss_list = avg_valid_loss_list
            if len(avg_valid_loss_list) >= 2 and current_batch == 0:
                if avg_valid_loss_list[-1] > min(avg_valid_loss_list):
                    self._counter += 1
                    new_lr = self._opt._lr * self._factor
                    if self._counter > self._patience and new_lr > self._min_lr:
                        self._opt._lr = new_lr
                        self._counter = 0
                else:
                    self._counter = 0

            return self._opt
        else:
            return self._opt

    def preprocess(self, x):
        # normalization
        x /= 255
        # mean=0.4914, 0.4822, 0.4465 and std=0.2023, 0.1994, 0.2010
        x[:, 0, :, :] -= 0.4914
        x[:, 1, :, :] -= 0.4822
        x[:, 2, :, :] -= 0.4465

        x[:, 0, :, :] /= 0.2023
        x[:, 1, :, :] /= 0.1994
        x[:, 2, :, :] /= 0.2010

        return x

    def _freeze(self):
        self._model.conv1.set_auto_update(self.train_whole_network)
        self._model.bn1.set_auto_update(self.train_whole_network)
        self._model.layer1.set_auto_update(self.train_whole_network)
        self._model.layer2.set_auto_update(self.train_whole_network)
        self._model.layer3.set_auto_update(self.train_whole_network)
        self._model.layer4.set_auto_update(self.train_whole_network)

    def set_last_layer_unit(self, unit_size):
        self._model.set_last_layer_unit(unit_size)


class ResNet(rm.Model):

    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/classification/ResNet.h5".format(
        __version__)

    def __init__(self, num_classes, block, layers):
        self.inplanes = 64
        self.num_class = num_classes
        super(ResNet, self).__init__()
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
        assert self.num_class > 0, \
            "Class map is empty. Please set the attribute class_map when instantiating a model. " +\
            "Or, please load a pre-trained model using the $B!F(Bload()$B!G(B method."
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

    def set_last_layer_unit(self, unit_size):
        self.fc._output_size = unit_size


class ResNet18(ResNetBase):
    """ResNet18 model.

    If the argument load_pretrained_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        load_pretrained_weight(bool):
        class_map: Array of class names
        imsize(int or tuple): Input image size.
        plateau: True if error plateau should be used
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument num_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    References:
        Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Deep Residual Learning for Image Recognition
        https://arxiv.org/abs/1512.03385
    """

    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/classification/ResNet18.h5".format(
        __version__)

    def __init__(self, class_map=None, imsize=(224, 224), plateau=False, load_pretrained_weight=False, train_whole_network=False):

        self._model = ResNet(1, BasicBlock, [2, 2, 2, 2])
        super(ResNet18, self).__init__(class_map, imsize,
                                       load_pretrained_weight, train_whole_network, self._model)

        self.decay_rate = 0.0001
        self._opt = rm.Sgd(0.1, 0.9)

        # for error plateau
        self.plateau = plateau
        self._patience = 15
        self._counter = 0
        self._min_lr = 1e-6
        self._factor = np.sqrt(0.1)
        self._model.fc.params = {}


class ResNet34(ResNetBase):
    """ResNet34 model.

    If the argument load_pretrained_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        load_pretrained_weight(bool):
        class_map: Array of class names
        imsize(int or tuple): Input image size.
        plateau: True if error plateau should be used
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument num_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    References:
        Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Deep Residual Learning for Image Recognition
        https://arxiv.org/abs/1512.03385
    """

    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/classification/ResNet34.h5".format(
        __version__)

    def __init__(self, class_map=None, imsize=(224, 224), plateau=False, load_pretrained_weight=False, train_whole_network=False):

        self._model = ResNet(1, BasicBlock, [3, 4, 6, 3])
        super(ResNet34, self).__init__(class_map, imsize,
                                       load_pretrained_weight, train_whole_network, self._model)

        self.decay_rate = 0.0001
        self._opt = rm.Sgd(0.1, 0.9)

        # for error plateau
        self.plateau = plateau
        self._patience = 15
        self._counter = 0
        self._min_lr = 1e-6
        self._factor = np.sqrt(0.1)
        self._model.fc.params = {}


class ResNet50(ResNetBase):
    """ResNet50 model.

    If the argument load_pretrained_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        load_pretrained_weight(bool):
        class_map: Array of class names
        imsize(int or tuple): Input image size.
        plateau: True if error plateau should be used
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument num_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    References:
        Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/abs/1603.05027
    """

    SERIALIZED = ("imsize", "class_map", "num_class")
    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/classification/ResNet50.h5".format(
        __version__)

    def __init__(self, class_map=None, imsize=(224, 224), plateau=False, load_pretrained_weight=False, train_whole_network=False):

        self._model = ResNet(1, Bottleneck, [3, 4, 6, 3])
        super(ResNet50, self).__init__(class_map, imsize,
                                       load_pretrained_weight, train_whole_network, self._model)

        self.decay_rate = 0.0001
        self._opt = rm.Sgd(0.1, 0.9)

        # for error plateau
        self.plateau = plateau
        self._patience = 15
        self._counter = 0
        self._min_lr = 1e-6
        self._factor = np.sqrt(0.1)
        self._model.fc.params = {}


class ResNet101(ResNetBase):
    """ResNet101 model.

    If the argument load_pretrained_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        load_pretrained_weight(bool):
        class_map: Array of class names
        imsize(int or tuple): Input image size.
        plateau: True if error plateau should be used
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument num_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    References:
        Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/abs/1603.05027
    """

    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/classification/ResNet101.h5".format(
        __version__)

    def __init__(self, class_map=None, imsize=(224, 224), plateau=False, load_pretrained_weight=False, train_whole_network=False):

        self._model = ResNet(1, Bottleneck, [3, 4, 23, 3])
        super(ResNet101, self).__init__(class_map, imsize,
                                        load_pretrained_weight, train_whole_network, self._model)

        self.decay_rate = 0.0001
        self._opt = rm.Sgd(0.1, 0.9)

        # for error plateau
        self.plateau = plateau
        self._patience = 15
        self._counter = 0
        self._min_lr = 1e-6
        self._factor = np.sqrt(0.1)
        self._model.fc.params = {}


class ResNet152(ResNetBase):
    """ResNet152 model.

    If the argument load_pretrained_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        load_pretrained_weight(bool):
        class_map: Array of class names
        imsize(int or tuple): Input image size.
        plateau: True if error plateau should be used
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument num_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    References:
        Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
        Identity Mappings in Deep Residual Networks
        https://arxiv.org/abs/1603.05027
    """

    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/classification/ResNet152.h5".format(
        __version__)

    def __init__(self, class_map=None, imsize=(224, 224), plateau=False, load_pretrained_weight=False, train_whole_network=False):

        self._model = ResNet(1, Bottleneck, [3, 8, 36, 3])
        super(ResNet152, self).__init__(class_map, imsize,
                                        load_pretrained_weight, train_whole_network, self._model)

        self.decay_rate = 0.0001
        self._opt = rm.Sgd(0.1, 0.9)

        # for error plateau
        self.plateau = plateau
        self._patience = 15
        self._counter = 0
        self._min_lr = 1e-6
        self._factor = np.sqrt(0.1)
        self._model.fc.params = {}
