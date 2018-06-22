import os
import sys
import numpy as np
import renom as rm


def conv_block(growth_rate):
    return rm.Sequential([
        rm.BatchNormalize(epsilon=0.001, mode='feature'),
        rm.Relu(),
        rm.Conv2d(growth_rate * 4, 1, padding=0),
        rm.BatchNormalize(epsilon=0.001, mode='feature'),
        rm.Relu(),
        rm.Conv2d(growth_rate, 3, padding=1),
    ])


def transition_layer(growth_rate):
    return rm.Sequential([
        rm.BatchNormalize(epsilon=0.001, mode='feature'),
        rm.Relu(),
        rm.Conv2d(growth_rate, filter=1, padding=0, stride=1),
        rm.AveragePool2d(filter=2, stride=2)
    ])


class DenseNet(rm.Sequential):

    def __init__(self, n_class, layer_per_block=[6, 12, 24, 16], growth_rate=32):
        self.layer_per_block = layer_per_block
        self.growth_rate = growth_rate
        self.classes = n_class

        layers = []
        layers.append(rm.Conv2d(64, 7, padding=3, stride=2))
        layers.append(rm.BatchNormalize(epsilon=0.001, mode='feature'))
        for i in layer_per_block[:-1]:
            for j in range(i):
                layers.append(conv_block(growth_rate))
            layers.append(transition_layer(growth_rate))
        for i in range(layer_per_block[-1]):
            layers.append(conv_block(growth_rate))
        layers.append(rm.Dense(n_class))

        super(DenseNet, self).__init__(layers)

    def forward(self, x):
        i = 0
        t = self._layers[i](x)
        i += 1
        t = rm.relu(self._layers[i](t))
        t = rm.max_pool2d(t, filter=3, stride=2, padding=1)
        i += 1
        for i in self.layer_per_block[:-1]:
            for j in range(i):
                tmp = t
                t = self._layers[i](t)
                i += 1
                t = rm.concat(tmp, t)
            t = self._layers[i](t)
            i += 1
        for i in range(self.layer_per_block[-1]):
            tmp = t
            t = self._layers[i](t)
            i += 1
            t = rm.concat(tmp, t)
        t = rm.average_pool2d(t, filter=7, stride=1)
        t = rm.flatten(t)
        t = self._layers[i](t)
        return t


class DenseNet121(DenseNet):
    """ DenseNet121 Model

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        n_class(int): The number of class
        layer_per_block: The number of layers in each block
        load_weight(bool): 

    Note:
        if the argument num_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
    Densely Connected Convolutional Network
    https://arxiv.org/pdf/1608.06993.pdf
    """

    WEIGHT_URL = "https://app.box.com/shared/static/eovmxxgzyh5vg2kpcukjj8ypnxng4j5v.h5"
    WEIGHT_PATH = os.path.join(DIR, 'densenet121.h5')

    def __init__(self, n_class, growth_rate=32, load_weight=False):
        layer_per_block = [6, 12, 24, 16]
        super(DenseNet121, self).__init__(n_class, layer_per_block, growth_rate=32)

        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if num_class != 1000:
            self._layers[-1].params = {}


class DenseNet169(DenseNet):
    """ DenseNet169 Model

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        n_class(int): The number of class
        layer_per_block: The number of layers in each block
        load_weight(bool): 

    Note:
        if the argument num_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
    Densely Connected Convolutional Network
    https://arxiv.org/pdf/1608.06993.pdf
    """
    def __init__(self, n_class, growth_rate=32, load_weight=False):
        layer_per_block = [6, 12, 32, 32]
        super(DenseNet121, self).__init__(n_class, layer_per_block, growth_rate=32)

        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if num_class != 1000:
            self._layers[-1].params = {}


class DenseNet201(DenseNet):
    """ DenseNet201 Model

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        n_class(int): The number of class
        layer_per_block: The number of layers in each block
        load_weight(bool): 

    Note:
        if the argument num_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
    Densely Connected Convolutional Network
    https://arxiv.org/pdf/1608.06993.pdf
    """

    WEIGHT_URL = "https://app.box.com/shared/static/eovmxxgzyh5vg2kpcukjj8ypnxng4j5v.h5"
    WEIGHT_PATH = os.path.join(DIR, 'densenet201.h5')

    def __init__(self, n_class, growth_rate=32, load_weight=False):
        layer_per_block = [6, 12, 48, 32]
        super(DenseNet121, self).__init__(n_class, layer_per_block, growth_rate=32)

        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if num_class != 1000:
            self._layers[-1].params = {}
