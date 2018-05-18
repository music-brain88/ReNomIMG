import os
import sys
import numpy as np
import renom as rm

class DenseNet(rm.Sequential):
    def __init__(self, num_classes, layer_per_block=[6, 12, 24, 16], growth_rate=32):
        """
        DenseNet (Densely Connected Convolutional Network) https://arxiv.org/pdf/1608.06993.pdf

        Input
            num_classes: Number of classes
            layer_per_block: array specifing number of layers in a block
            growth_rate: int1
        """

        self.layer_per_block = layer_per_block
        self.growth_rate = growth_rate
        self.classes = num_classes

        layers = []
        layers.append(rm.Conv2d(64, 7, padding=3, stride=2))
        layers.append(rm.BatchNormalize(epsilon=0.001, mode='feature'))
        for i in layer_per_block[:-1]:
            for j in range(i):
                layers.append(conv_block(growth_rate))
            layers.append(transition_layer(growth_rate))
        for i in range(layer_per_block[-1]):
            layers.append(conv_block(growth_rate))
        layers.append(rm.Dense(num_classes))

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
    def __init__(self, num_classes, growth_rate=32, load_weight=False):
        layer_per_block = [6, 12, 24, 16]
        super(DenseNet121, self).__init__(num_classes, layer_per_block, growth_rate=32)
        if load_weight:
            self.load('densenet121.h5')


class DenseNet169(DenseNet):
    def __init__(self, num_classes, growth_rate=32, load_weight=False):
        layer_per_block = [6, 12, 32, 32]
        super(DenseNet121, self).__init__(num_classes, layer_per_block, growth_rate=32)
        if load_weight:
            self.load('densenet169.h5')

class DenseNet201(DenseNet):
    def __init__(self, num_classes, growth_rate=32, load_weight=False):
        layer_per_block = [6, 12, 48, 32]
        super(DenseNet121, self).__init__(num_classes, layer_per_block, growth_rate=32)
        if load_weight:
            self.load('densenet201.h5')

