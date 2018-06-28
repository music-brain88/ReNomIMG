#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import sys

import numpy as np
import renom as rm
from tqdm import tqdm
from renom_img.api.utility.misc.download import download
from renom_img.api.model.classification_base import ClassificationBase
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.target import DataBuilderClassification
DIR = os.path.split(os.path.abspath(__file__))[0]

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

class DenseNetBase(ClassificationBase):
    def __init__(self, class_map):
        super(DenseNetBase, self).__init__(class_map)
        self._opt = rm.Sgd(0.1, 0.9)

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
            if current_epoch == 30:
                lr = self._opt._lr / 10
            elif current_epoch == 60:
                lr = self._opt._lr / 10
            self._opt._lr = lr
            return self._opt


    def preprocess(self, x):
        """Image preprocess for VGG.

        Args:
            x (ndarray):

        Returns:
            (ndarray): Preprocessed data.
        """
        return x / 255.

    def regularize(self, decay_rate=0.0001):
        """L2 Regularization term. You can use this function to add L2 regularization term to a loss function.

        In VGG16, weight decay of 0.0005 is used.

        Example:
            >>> import numpy as np
            >>> from renom_img.api.model.vgg import VGG16
            >>> x = np.random.rand(1, 3, 224, 224)
            >>> y = np.random.rand(1, (5*2+20)*7*7)
            >>> model = VGG16()
            >>> loss = model.loss(x, y)
            >>> reg_loss = loss + model.regularize() # Add weight decay term.

        """
        return super().regularize(decay_rate)

    def fit(self, train_img_path_list=None, train_annotation_list=None, augmentation=None, valid_img_path_list=None, valid_annotation_list=None,  epoch=90, batch_size=16, callback_end_epoch=None):
        if train_img_path_list is not None and train_annotation_list is not None:
            train_dist = ImageDistributor(train_img_path_list, train_annotation_list, augmentation=augmentation)
        else:
            train_dist = train_image_distributor

        assert train_dist is not None

        if valid_img_path_list is not None and valid_annotation_list is not None:
            valid_dist = ImageDistributor(valid_img_path_list, valid_annotation_list)
        else:
            valid_dist = valid_image_distributor

        opt_flag = False
        batch_loop = int(np.ceil(len(train_dist) / batch_size))
        avg_train_loss_list = []
        avg_valid_loss_list = []
        for e in range(epoch):
            bar = tqdm(range(batch_loop))
            display_loss = 0
            for i, (train_x, train_y) in enumerate(train_dist.batch(batch_size, target_builder=DataBuilderClassification(self.imsize, self.class_map))):
                self.set_models(inference=False)
                with self.train():
                    loss = self.loss(self(train_x), train_y)
                    reg_loss = loss + self.regularize()

                reg_loss.grad().update(self.get_optimizer(e, epoch, i, batch_loop))
                try:
                    loss = loss.as_ndarray()[0]
                except:
                    loss = loss.as_ndarray()
                display_loss += loss
                bar.set_description("Epoch:{:03d} Train Loss:{:5.3f}".format(e, loss))
                bar.update(1)
            avg_train_loss = display_loss / (i + 1)
            avg_train_loss_list.append(avg_train_loss)

            if valid_dist is not None:
                display_loss = 0
                for i, (valid_x, valid_y) in enumerate(valid_dist.batch(batch_size, target_builder=DataBuilderClassification(self.imsize, self.class_map))):
                    self.set_models(inference=True)
                    loss = self.loss(self(train_x), train_y)
                    try:
                        loss = loss.as_ndarray()[0]
                    except:
                        loss = loss.as_ndarray()
                    display_loss += loss
                    bar.set_description("Epoch:{:03d} Valid Loss:{:5.3f}".format(e, loss))
                    bar.update(1)
                avg_valid_loss = display_loss / (i + 1)
                avg_valid_loss_list.append(avg_train_loss)
                bar.set_description("Epoch:{:03d} Avg Train Loss:{:5.3f} Avg Valid Loss:{:5.3f}".format(
                    e, avg_train_loss, avg_valid_loss))
            else:
                bar.set_description("Epoch:{:03d} Avg Train Loss:{:5.3f}".format(e, avg_train_loss))
            bar.close()
            if callback_end_epoch is not None:
                callback_end_epoch(e, self, avg_train_loss_list, avg_valid_loss_list)
        return avg_train_loss_list, avg_valid_loss_list

class DenseNet(DenseNetBase):
    """
    DenseNet (Densely Connected Convolutional Network) https://arxiv.org/pdf/1608.06993.pdf

    Input
        class_map: Array of class names
        layer_per_block: array specifing number of layers in a block.
        growth_rate(int): Growth rate of the number of filters.
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.
    """
    def __init__(self, class_map, layer_per_block=[6, 12, 24, 16], growth_rate=32, imsize=(224, 224), train_whole_network=False):
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        self.layer_per_block = layer_per_block
        self.growth_rate = growth_rate
        self.n_class = len(class_map)

        layers = []
        layers.append(rm.Conv2d(64, 7, padding=3, stride=2))
        layers.append(rm.BatchNormalize(epsilon=0.001, mode='feature'))
        for i in layer_per_block[:-1]:
            for j in range(i):
                layers.append(conv_block(growth_rate))
            layers.append(transition_layer(growth_rate))
        for i in range(layer_per_block[-1]):
            layers.append(conv_block(growth_rate))

        self.freezed_network = rm.Sequential(layers)
        self._network = rm.Dense(self.n_class)
        self._train_whole_network = train_whole_network
        self.imsize = imsize

        super(DenseNet, self).__init__(class_map)

    @property
    def freezed_network(self):
        return self.freezed_network

    @property
    def network(self):
        return self._network

    def forward(self, x):
        self.freezed_network.set_auto_update(self._train_whole_network)
        i = 0
        t = self.freezed_network[i](x)
        i += 1
        t = rm.relu(self.freezed_network[i](t))
        i += 1
        t = rm.max_pool2d(t, filter=3, stride=2, padding=1)
        for j in self.layer_per_block[:-1]:
            for k in range(j):
                tmp = t
                t = self.freezed_network[i](t)
                i += 1
                t = rm.concat(tmp, t)
            t = self.freezed_network[i](t)
            i += 1
        for j in range(self.layer_per_block[-1]):
            tmp = t
            t = self.freezed_network[i](t)
            i += 1
            t = rm.concat(tmp, t)
        t = rm.average_pool2d(t, filter=7, stride=1)
        t = rm.flatten(t)
        t = self.network(t)
        return t


class DenseNet121(DenseNet):
    """ DenseNet121 Model

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        class_map: Array of class names
        growth_rate(int): Growth rate of the number of filters.
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
    Densely Connected Convolutional Network
    https://arxiv.org/pdf/1608.06993.pdf
    """

    WEIGHT_URL = "https://app.box.com/shared/static/eovmxxgzyh5vg2kpcukjj8ypnxng4j5v.h5"
    WEIGHT_PATH = os.path.join(DIR, 'densenet121.h5')

    def __init__(self, class_map, growth_rate=32, load_weight=False, imsize=(224, 224), train_whole_network=False):
        layer_per_block = [6, 12, 24, 16]
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        super(DenseNet121, self).__init__(class_map, layer_per_block, growth_rate=growth_rate, imsize=imsize, train_whole_network=train_whole_network)
        n_class = len(class_map)
        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if n_class != 1000:
            self.freezed_network.params = {}


class DenseNet169(DenseNet):
    """ DenseNet169 Model

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        class_map: Array of class names
        growth_rate(int): Growth rate of the number of filters.
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
    Densely Connected Convolutional Network
    https://arxiv.org/pdf/1608.06993.pdf
    """

    def __init__(self, class_map, growth_rate=32, load_weight=False, imsize=(224, 224), train_whole_network=False):
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        layer_per_block = [6, 12, 32, 32]
        super(DenseNet169, self).__init__(class_map, layer_per_block, growth_rate=growth_rate, imsize=imsize,  train_whole_network=train_whole_network)
        n_class = len(class_map)
        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if n_class != 1000:
            self.network.params = {}


class DenseNet201(DenseNet):
    """ DenseNet201 Model

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        class_map: Array of class names
        growth_rate(int): Growth rate of the number of filters.
        load_weight(bool): True if the pre-trained weight is loaded.
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
    Densely Connected Convolutional Network
    https://arxiv.org/pdf/1608.06993.pdf
    """

    WEIGHT_URL = "https://app.box.com/shared/static/eovmxxgzyh5vg2kpcukjj8ypnxng4j5v.h5"
    WEIGHT_PATH = os.path.join(DIR, 'densenet201.h5')

    def __init__(self, class_map, growth_rate=32, load_weight=False, imsize=(224, 224), train_whole_network=False):
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        layer_per_block = [6, 12, 48, 32]

        super(DenseNet201, self).__init__(class_map, layer_per_block, growth_rate=growth_rate, imsize=imsize, train_whole_network=train_whole_network)
        n_class = len(class_map)
        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if n_class != 1000:
            self.network.params = {}
