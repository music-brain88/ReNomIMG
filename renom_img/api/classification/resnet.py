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


def identity_block(f,c):

    layers = []
    layers.append(rm.Conv2d(channel=c,filter=(1,1),stride=1,padding=0))
    layers.append(rm.BatchNormalize(epsilon=0.001, mode='feature'))
    layers.append(rm.Relu())

    layers.append(rm.Conv2d(channel=c,filter=(f,f),stride=1,padding=1))
    layers.append(rm.BatchNormalize(epsilon=0.001, mode='feature'))
    layers.append(rm.Relu())

    layers.append(rm.Conv2d(channel=c*4,filter=(1,1),stride=1,padding=0))
    layers.append(rm.BatchNormalize(epsilon=0.001, mode='feature'))

    return rm.Sequential(layers)

def convolution_block(f,c,s, short_cut=0):

    layers = []
    if short_cut==0:
        layers.append(rm.Conv2d(channel=c,filter=(1,1),stride=s,padding=0))
        layers.append(rm.BatchNormalize(epsilon=0.001, mode='feature'))
        layers.append(rm.Relu())

        layers.append(rm.Conv2d(channel=c,filter=(f,f),stride=1,padding=1))
        layers.append(rm.BatchNormalize(epsilon=0.001, mode='feature'))
        layers.append(rm.Relu())

        layers.append(rm.Conv2d(channel=c*4,filter=(1,1),stride=1,padding=0))

        return rm.Sequential(layers)
    else:
        layers.append(rm.Conv2d(channel=c*4,filter=(1,1),stride=s,padding=0))
        return rm.Sequential(layers)


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
        else:
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
        self._model.conv1.set_auto_update(self._train_whole_network)
        self._model.bn.set_auto_update(self._train_whole_network)
        self._model.base.set_auto_update(self._train_whole_network)


class CNN_ResNet(rm.Model):

    def __init__(self, num_class, channels, num_layers):
        self.channels = channels
        self.num_layers = num_layers
        self.layers = []
#         stage 1
        # here padding is three to match the weight of keras weight file. original paper doesn't have it
        self.conv1 = rm.Conv2d(channel=64,filter=(7,7),stride=2,padding=3)
        self.bn = rm.BatchNormalize(epsilon=0.001, mode='feature')
        self.activation = rm.Relu()
        self.maxpool = rm.MaxPool2d(filter=(3,3),stride=2,padding=1)

#         rest of the stages
        for i,num in enumerate(self.num_layers):
            for j in range(num):
                if j ==0:
                    if i==0:
#                         may be this type of conv block should be added more
                        self.layers.append(convolution_block(f=3,c=self.channels[i],s=1))
                        self.layers.append(convolution_block(f=3,c=self.channels[i],s=1,short_cut=1))
                        self.layers.append(rm.Sequential([rm.BatchNormalize(epsilon=0.001, mode='feature')]))
                        self.layers.append(rm.Sequential([rm.BatchNormalize(epsilon=0.001, mode='feature')]))
                    else:
                        self.layers.append(convolution_block(f=3,c=self.channels[i],s=2))
                        self.layers.append(convolution_block(f=3,c=self.channels[i],s=2,short_cut=1))
                        self.layers.append(rm.Sequential([rm.BatchNormalize(epsilon=0.001, mode='feature')]))
                        self.layers.append(rm.Sequential([rm.BatchNormalize(epsilon=0.001, mode='feature')]))
                else:
                    self.layers.append(identity_block(f=3,c=self.channels[i]))

        self.base = rm.Sequential(self.layers)
        self.flat = rm.Flatten()
        self.fc = rm.Dense(num_class)

    def forward(self, x):
        t = self.conv1(x)
        t = self.bn(t)
        t = self.activation(t)
        t = self.maxpool(t)

        index = 0
        # the rest of the blocks
        for num in self.num_layers:
            for i in range(num):
                if i==0:
                    tmp = t
                    t = self.base[index](t) # Conv
                    index+=1
                    tmp = self.base[index](tmp) # Conv for shortcut
                    index+=1
                    t = self.base[index](t) # Batch
                    index+=1
                    tmp = self.base[index](tmp) # Batch for shortcut
                    index+=1
                    t = t+tmp
                    t = rm.relu(t)

                else:
                    tmp = t
                    t = self.base[index](t)
                    index+=1
                    t = t + tmp
                    t = rm.relu(t)

        t = rm.average_pool2d(t, filter=(t.shape[2], t.shape[3]))
        t = self.flat(t)
        t = self.fc(t)

        return t


class ResNet50(ResNetBase):
    """ResNet50 model.

    If the argument load_pretrained_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        load_pretrained_weight(bool):
        class_map: Array of class names
        imsize(int or tuple): Input image size.
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
    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/ResNet50.h5"

    def __init__(self, class_map=[], imsize=(224, 224), load_pretrained_weight=False, train_whole_network=False):
        num_layers = [3, 4, 6, 3]
        CHANNELS = [64, 128, 256, 512]

        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        self.num_class = len(class_map)
        self.class_map = [c.encode("ascii", "ignore") for c in class_map]
        self.imsize = imsize
        self._train_whole_network = train_whole_network
        self.decay_rate = 0.0001

        self._model = CNN_ResNet(self.num_class, CHANNELS, num_layers)
        self._opt = rm.Sgd(0.1, 0.9)

        # for error plateau
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
