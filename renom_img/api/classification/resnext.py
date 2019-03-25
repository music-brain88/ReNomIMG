import os
import sys
import renom as rm
import numpy as np
from tqdm import tqdm
from PIL import Image
from renom_img import __version__
from renom_img.api.utility.misc.download import download
from renom_img.api.classification import Classification
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.target import DataBuilderClassification
from renom_img.api.cnn.resnext import CnnResNeXt, Bottleneck
from renom_img.api.utility.optimizer import OptimizerResNeXt

RESIZE_METHOD = Image.BILINEAR

class TargetBuilderResNeXt():
    def __init__(self, class_map, imsize):
        self.class_map = class_map
        self.imsize = imsize

    def __call__(self, *args, **kwargs):
        return self.build(*args, **kwargs)

    def preprocess(self, x):
        # normalization
        x /= 255
        # mean=0.485, 0.456, 0.406 and std=0.229, 0.224, 0.225
        # these values taken from official facebook torch implementation
        # https://github.com/facebook/fb.resnet.torch/blob/master/pretrained/classify.lua
        x[:, 0, :, :] -= 0.485
        x[:, 1, :, :] -= 0.456
        x[:, 2, :, :] -= 0.406

        x[:, 0, :, :] /= 0.229
        x[:, 1, :, :] /= 0.224
        x[:, 2, :, :] /= 0.225

        return x

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


class ResNeXt50(Classification):
    """ResNeXt50 model.

    If the argument load_pretrained_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        load_pretrained_weight(bool):
        class_map: Array of class names
        imsize(int or tuple): Input image size.
        plateau: True if error plateau should be used
        train_whole_network(bool): True if the overal model is trained.
        cardinality: number of groups in group convolution layers (default = 32)

    Note:
        if the argument num_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    References:
        Aggregated Residual Transformations for Deep Neural Networks
        Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, Kaiming H
        https://arxiv.org/abs/1611.05431
    """

    SERIALIZED = ("imsize", "class_map", "num_class")
    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/classification/ResNeXt50.h5".format(
        __version__)

    def __init__(self, class_map=[], imsize=(224, 224), cardinality=32, plateau=False, load_pretrained_weight=False, train_whole_network=False):


        self.cardinality = cardinality

        self.model = CnnResNeXt(1, Bottleneck, [3, 4, 6, 3], self.cardinality)
        super(ResNeXt50, self).__init__(class_map, imsize, load_pretrained_weight, train_whole_network, self.model)

        self.model.set_output_size(self.num_class)
        self.model.set_train_whole(train_whole_network)
        
        self.default_optimizer = OptimizerResNeXt()
        self.decay_rate = 0.0001
        self.model.fc.params = {}

    def build_data(self):
        return TargetBuilderResNeXt(self.class_map, self.imsize)

class ResNeXt101(Classification):
    """ResNeXt101 model.

    If the argument load_pretrained_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        load_pretrained_weight(bool):
        class_map: Array of class names
        imsize(int or tuple): Input image size.
        plateau: True if error plateau should be used
        train_whole_network(bool): True if the overal model is trained.
        cardinality: number of groups in group convolution layers (default = 32)

    Note:
        if the argument num_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    References:
        Aggregated Residual Transformations for Deep Neural Networks
        Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, Kaiming H
        https://arxiv.org/abs/1611.05431
    """

    SERIALIZED = ("imsize", "class_map", "num_class")
    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/classification/ResNeXt101.h5".format(
        __version__)

    def __init__(self, class_map=[], imsize=(224, 224), cardinality=32, plateau=False, load_pretrained_weight=False, train_whole_network=False):

        self.cardinality = cardinality

        self.model = CnnResNeXt(1, Bottleneck, [3, 4, 23, 3], self.cardinality)
        super(ResNeXt50, self).__init__(class_map, imsize, load_pretrained_weight, train_whole_network, self.model)

        self.default_optimizer = OptimizerResNeXt()
        self.decay_rate = 0.0001
        self.model.fc.params = {}

    def build_data(self):
        return TargetBuilderResNeXt(self.class_map, self.imsize)



