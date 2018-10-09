import os
import sys
import renom as rm
import numpy as np
from tqdm import tqdm

from renom_img.api.utility.misc.download import download
from renom_img.api.classification import Classification
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.target import DataBuilderClassification

DIR = os.path.split(os.path.abspath(__file__))[0]


class Basic(rm.Model):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(Basic, self).__init__()
        self.conv1 = rm.Conv2d(planes, filter=3,stride=stride,padding=1)
        self.bn1 = rm.BatchNormalize(epsilon=0.001, mode='feature')
        self.conv2 = rm.Conv2d(planes, filter=3, stride=1, padding=1)
        self.bn2 = rm.BatchNormalize(epsilon=0.001, mode='feature')

        self.shortcut = rm.Sequential([])
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = rm.Sequential([
                rm.Conv2d(self.expansion*planes, filter=1, stride=stride),
                rm.BatchNormalize(epsilon=0.001, mode='feature')
            ])

    def forward(self, x):
        out = rm.relu(self.bn1(self.conv1(x)))
        out = rm.relu(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        out = rm.relu(out)
        return out

class Bottleneck(rm.Model):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = rm.Conv2d(planes, filter=1)
        self.bn1 = rm.BatchNormalize(epsilon=0.001, mode='feature')
        self.conv2 = rm.Conv2d(planes, filter=3, stride=stride, padding=1)
        self.bn2 = rm.BatchNormalize(epsilon=0.001, mode='feature')
        self.conv3 = rm.Conv2d(self.expansion*planes, filter=1)
        self.bn3 = rm.BatchNormalize(epsilon=0.001, mode='feature')

        self.shortcut = rm.Sequential([])
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = rm.Sequential([
                rm.Conv2d(self.expansion*planes, filter=1, stride=stride),
                rm.BatchNormalize(epsilon=0.001, mode='feature')
            ])

    def forward(self, x):
        out = rm.relu(self.bn1(self.conv1(x)))
        out = rm.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = rm.relu(out)
        return out


class ResNetBase(Classification):

    def get_optimizer(self, current_epoch=None, total_epoch=None, current_batch=None, total_batch=None, **kwargs):
        """Returns an instance of Optimiser for training Yolov1 algorithm.

        Args:
            current_epoch:
            total_epoch:
            current_batch:
            total_epoch:
        """
        if any([num is None for num in [current_epoch, total_epoch, current_batch, total_batch]]):
            return self._opt
        elif self.plateau:
            avg_valid_loss_list = kwargs['avg_valid_loss_list']
            if len(avg_valid_loss_list) >= 2 and current_batch==0:
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
        if not self._train_whole_network:
            self._model.set_auto_update(self._train_whole_network)
            self._model.dnse.set_auto_update(True)
        else:
            self._model.set_auto_update(self._train_whole_network)


class ResNet(rm.Model):
    def __init__(self, num_classes, block, num_blocks):
        self.in_planes = 64

        self.conv1 = rm.Conv2d(64, filter=3, stride=1, padding=1)
        self.bn1 = rm.BatchNormalize(epsilon=0.001, mode='feature')
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.flat = rm.Flatten()
        self.dnse = rm.Dense(num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return rm.Sequential(layers)

    def forward(self, x):
        out = rm.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = rm.average_pool2d(out,filter=(out.shape[2],out.shape[3]))
        out = self.flat(out)
        out = self.dnse(out)
        return out


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

    SERIALIZED = ("imsize", "class_map", "num_class")
    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/ResNet18.h5"

    def __init__(self, class_map=[], imsize=(224, 224), plateau= False, load_pretrained_weight=False, train_whole_network=False):

        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        self.num_class = len(class_map)
        self.class_map = [c.encode("ascii", "ignore") for c in class_map]
        self.imsize = imsize
        self._train_whole_network = train_whole_network
        self.decay_rate = 0.0001

        self._model = ResNet(self.num_class, Basic, [2, 2, 2, 2])
        self._opt = rm.Sgd(0.1, 0.9)

        # for error plateau
        self.plateau = plateau
        self._patience = 15
        self._counter = 0
        self._min_lr = 1e-6
        self._factor = np.sqrt(0.1)

        if load_pretrained_weight:
            if isinstance(load_pretrained_weight, bool):
                load_pretrained_weight = self.__class__.__name__ + '.h5'

            if not os.path.exists(load_pretrained_weight):
                download(self.WEIGHT_URL, load_pretrained_weight)

            self._model.load(load_pretrained_weight)
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
    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/ResNet50.h5"

    def __init__(self, class_map=[], imsize=(224, 224), plateau= False, load_pretrained_weight=False, train_whole_network=False):

        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        self.num_class = len(class_map)
        self.class_map = [c.encode("ascii", "ignore") for c in class_map]
        self.imsize = imsize
        self._train_whole_network = train_whole_network
        self.decay_rate = 0.0001

        self._model = ResNet(self.num_class, Bottleneck, [3, 4, 6, 3])
        self._opt = rm.Sgd(0.1, 0.9)

        # for error plateau
        self.plateau = plateau
        self._patience = 15
        self._counter = 0
        self._min_lr = 1e-6
        self._factor = np.sqrt(0.1)

        if load_pretrained_weight:
            if isinstance(load_pretrained_weight, bool):
                load_pretrained_weight = self.__class__.__name__ + '.h5'

            if not os.path.exists(load_pretrained_weight):
                download(self.WEIGHT_URL, load_pretrained_weight)

            self._model.load(load_pretrained_weight)
            self._model.fc.params = {}
