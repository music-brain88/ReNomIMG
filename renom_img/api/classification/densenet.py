#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import sys

import numpy as np
import renom as rm
from tqdm import tqdm
from PIL import Image
from renom_img import __version__
from renom_img.api.utility.misc.download import download
from renom_img.api.classification import Classification
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.cnn.densenet import CNN_DenseNet
from renom_img.api.utility.optimizer import OptimizerDenseNet

RESIZE_METHOD = Image.BILINEAR

class TargetBuilderDenseNet():
    def __init__(self, class_map, imsize):
        self.class_map = class_map
        self.imsize = imsize

    def __call__(self, *args, **kwargs):
        return self.build(*args, **kwargs)

    def preprocess(self, x):
        return x/255.

    def resize_img(self, img_list, label_list):
        im_list = []

        for img in img_list:
            channel_last = img.transpose(1, 2, 0)
            img = Image.fromarray(np.uint8(channel_last))
            img = img.resize(self.imsize, RESIZE_METHOD).convert('RGB')
            im_list.append(np.asarray(img))

        return np.asarray(im_list).transpose(0, 3, 1, 2).astype(np.float32), np.asarray(label_list)

    def load_img(self, path):
        """ Loads an image

        Args:
            path(str): A path of an image

        Returns:
            (tuple): Returns image(numpy.array), the ratio of the given width to the actual image width,
                     and the ratio of the given height to the actual image height
        """
        img = Image.open(path)
        img.load()
        w, h = img.size
        img = img.convert('RGB')
        # img = img.resize(self.imsize, RESIZE_METHOD)
        img = np.asarray(img).transpose(2, 0, 1).astype(np.float32)
        return img, self.imsize[0] / float(w), self.imsize[1] / h


    def build(self, img_path_list, annotation_list, augmentation=None, **kwargs):
        """ Builds an array of images and corresponding labels

        Args:
            img_path_list(list): List of input image paths.
            annotation_list(list): List of class id
                                    [1, 4, 6 (int)]
            augmentation(Augmentation): Instance of the augmentation class.

        Returns:
            (tuple): Batch of images and corresponding one hot labels for each image in a batch
        """

        # Check the class mapping.
        n_class = len(self.class_map)

        img_list = []
        label_list = []
        for img_path, an_data in zip(img_path_list, annotation_list):
            one_hot = np.zeros(n_class)
            img, sw, sh = self.load_img(img_path)
            img_list.append(img)
            one_hot[an_data] = 1.
            label_list.append(one_hot)

        if augmentation is not None:
            img_list, label_list = augmentation(img_list, label_list, mode="classification")

        img_list, label_list = self.resize_img(img_list, label_list)

        return self.preprocess(np.array(img_list)), np.array(label_list)


class DenseNet121(Classification):
    """ DenseNet121 Model

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        class_map: Array of class names
        growth_rate(int): Growth rate of the number of filters.
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument num_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    References:
        Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
        Densely Connected Convolutional Network
        https://arxiv.org/pdf/1608.06993.pdf
    """

    SERIALIZED = ("imsize", "class_map", "num_class")
    WEIGHT_URL = "".format(
        __version__)

    def __init__(self, class_map=[], imsize=(224, 224), load_pretrained_weight=False, train_whole_network=False):
        assert not load_pretrained_weight, "In ReNomIMG version {}, pretained weight of {} is not prepared.".format(
            __version__, self.__class__.__name__)

        layer_per_block = [6, 12, 24, 16]
        growth_rate = 32
        self.model = CNN_DenseNet(1, layer_per_block,
                                   growth_rate)

        super(DenseNet121, self).__init__(class_map, imsize, load_pretrained_weight, train_whole_network, self.model)

        self.model.set_train_whole(train_whole_network)
        self.model.set_output_size(self.num_class)
        self.default_optimizer = OptimizerDenseNet()
        self.decay_rate = 0.0005

    def build_data(self):
        return TargetBuilderDenseNet(self.class_map, self.imsize)



class DenseNet169(Classification):
    """ DenseNet169 Model

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        class_map: Array of class names
        growth_rate(int): Growth rate of the number of filters.
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument num_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    References:
        Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
        Densely Connected Convolutional Network
        https://arxiv.org/pdf/1608.06993.pdf
    """

    SERIALIZED = ("imsize", "class_map", "num_class")
    WEIGHT_URL = ""

    def __init__(self, class_map=[], imsize=(224, 224), load_pretrained_weight=False, train_whole_network=False):
        assert not load_pretrained_weight, "In ReNomIMG version {}, pretained weight of {} is not prepared.".format(
            __version__, self.__class__.__name__)

        layer_per_block = [6, 12, 32, 32]
        growth_rate = 32

        self.model = CNN_DenseNet(1, layer_per_block,
                                   growth_rate)

        super(DenseNet169, self).__init__(class_map, imsize, load_pretrained_weight, train_whole_network, self.model)

        self.model.set_train_whole(train_whole_network)
        self.model.set_output_size(self.num_class)
        self.default_optimizer = OptimizerDenseNet()
        self.decay_rate = 0.0005

    def build_data(self):
        return TargetBuilderDenseNet(self.class_map, self.imsize)



class DenseNet201(Classification):
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
        if the argument num_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    References:
        Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger
        Densely Connected Convolutional Network
        https://arxiv.org/pdf/1608.06993.pdf
    """

    SERIALIZED = ("imsize", "class_map", "num_class")
    WEIGHT_URL = ""

    def __init__(self, class_map=[], imsize=(224, 224), load_pretrained_weight=False, train_whole_network=False):
        assert not load_pretrained_weight, "In ReNomIMG version {}, pretained weight of {} is not prepared.".format(
            __version__, self.__class__.__name__)

        layer_per_block = [6, 12, 48, 32]
        growth_rate = 32

        self.model = CNN_DenseNet(1, layer_per_block,
                                   growth_rate)

        super(DenseNet201, self).__init__(class_map, imsize, load_pretrained_weight, train_whole_network, self.model)

        self.model.set_train_whole(train_whole_network)
        self.model.set_output_size(self.num_class)
        self.default_optimizer = OptimizerDenseNet()
        self.decay_rate = 0.0005

    def build_data(self):
        return TargetBuilderDenseNet(self.class_map, self.imsize)

