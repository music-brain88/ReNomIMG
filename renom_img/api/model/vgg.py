#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import numpy as np
import renom as rm
from tqdm import tqdm
from renom_img.api.utility.misc.download import download
from renom_img.api.model.classification_base import ClassificationBase
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.target import DataBuilderClassification

DIR = os.path.split(os.path.abspath(__file__))[0]


def layer_factory(channel=32, conv_layer_num=2):
    layers = []
    for _ in range(conv_layer_num):
        layers.append(rm.Conv2d(channel=channel, padding=1, filter=3))
        layers.append(rm.Relu())
    layers.append(rm.MaxPool2d(filter=2, stride=2))
    return rm.Sequential(layers)


class VGGBase(ClassificationBase):
    def __init__(self, class_map):
        super(VGGBase, self).__init__(class_map)
        self._opt = rm.Sgd(0.01, 0.9)

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
            self._opt._lr = lr / 10.
            return self._opt

    def preprocess(self, x):
        """Image preprocess for VGG.

        Args:
            x (ndarray):

        Returns:
            (ndarray): Preprocessed data.
        """
        x[:, 0, :, :] -= 123.68  # R
        x[:, 1, :, :] -= 116.779  # G
        x[:, 2, :, :] -= 103.939  # B
        return x

    def regularize(self, decay_rate=0.0005):
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

    def fit(self, train_img_path_list=None, train_annotation_list=None, augmentation=None, valid_img_path_list=None, valid_annotation_list=None,  epoch=200, batch_size=16, callback_end_epoch=None):
        if train_img_path_list is not None and train_annotation_list is not None:
            train_dist = ImageDistributor(
                train_img_path_list, train_annotation_list, augmentation=augmentation)
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

                if opt_flag:
                    reg_loss.grad().update(self.get_optimizer(e, epoch, i, batch_loop))
                    opt_flag = False
                else:
                    reg_loss.grad().update(self.opt)
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
                avg_valid_loss_list.append(avg_valid_loss)
                if avg_valid_loss[-1] > avg_valid_loss[-2]:
                    opt_flag = True
                bar.set_description("Epoch:{:03d} Avg Train Loss:{:5.3f} Avg Valid Loss:{:5.3f}".format(
                    e, avg_train_loss, avg_valid_loss))
            else:
                bar.set_description("Epoch:{:03d} Avg Train Loss:{:5.3f}".format(e, avg_train_loss))
            bar.close()
            if callback_end_epoch is not None:
                callback_end_epoch(e, self, avg_train_loss_list, avg_valid_loss_list)
        return avg_train_loss_list, avg_valid_loss_list


class VGG16(VGGBase):
    """VGG16 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        class_map(array): Array of class names
        load_weight(bool):
        imsize(int or tuple): Input image size
        train_whole_network(bool): True if the overall model is trained, otherwise False


    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition
    https://arxiv.org/abs/1409.1556
    """

    WEIGHT_URL = "http://docs.renom.jp/downloads/weights/Vgg16.h5"
    WEIGHT_PATH = os.path.join(DIR, 'vgg16.h5')

    def __init__(self, class_map, load_weight=False, imsize=(224, 224), train_whole_network=False):
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)

        self.n_class = len(class_map)

        model = [
            layer_factory(channel=64, conv_layer_num=2),
            layer_factory(channel=128, conv_layer_num=2),
            layer_factory(channel=256, conv_layer_num=3),
            layer_factory(channel=512, conv_layer_num=3),
            layer_factory(channel=512, conv_layer_num=3),
            rm.Flatten(),
            rm.Dense(4096),
            rm.Relu(),
            rm.Dropout(0.5),
            rm.Dense(4096),
            rm.Relu(),
            rm.Dropout(0.5),
            rm.Dense(self.n_class)
        ]
        self.class_map = class_map
        self._train_whole_network = train_whole_network
        self.imsize = imsize
        self._freezed_network = rm.Sequential(model[:5])
        self._network = rm.Sequential(model[5:])

        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if self.n_class != 1000:
            self._network.params = {}

        super(VGG16, self).__init__(class_map)

    @property
    def freezed_network(self):
        return self._freezed_network

    @property
    def network(self):
        return self._network

    def forward(self, x):
        self.freezed_network.set_auto_update(self._train_whole_network)
        return self.network(self.freezed_network(x))


class VGG19(VGGBase):
    """VGG19 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        class_map(array): Array of class names
        load_weight(bool):
        imsize(int or tuple): Input image size
        train_whole_network(bool): True if the overall model is trained, otherwise False


    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition
    https://arxiv.org/abs/1409.1556
    """

    WEIGHT_URL = "http://docs.renom.jp/downloads/weights/Vgg16.h5"
    WEIGHT_PATH = os.path.join(DIR, 'vgg19.h5')

    def __init__(self, class_map, imsize=(224, 224), load_weight=False, train_whole_network=False):
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)

        self.n_class = len(class_map)

        model = [
            layer_factory(channel=64, conv_layer_num=2),
            layer_factory(channel=128, conv_layer_num=2),
            layer_factory(channel=256, conv_layer_num=4),
            layer_factory(channel=512, conv_layer_num=4),
            layer_factory(channel=512, conv_layer_num=4),
            rm.Flatten(),
            rm.Dense(4096),
            rm.Relu(),
            rm.Dropout(0.5),
            rm.Dense(4096),
            rm.Relu(),
            rm.Dropout(0.5),
            rm.Dense(self.n_class)
        ]
        self.class_map = class_map
        self._train_whole_network = train_whole_network
        self.imsize = imsize
        self._freezed_network = rm.Sequential(model[:5])
        self._network = rm.Sequential(model[5:])

        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if self.n_class != 1000:
            self._network.params = {}

        super(VGG16, self).__init__(class_map)

    @property
    def freezed_network(self):
        return self._freezed_network

    @property
    def network(self):
        return self._network

    def forward(self, x):
        self.freezed_network.set_auto_update(self._train_whole_network)
        return self.network(self.freezed_network(x))
