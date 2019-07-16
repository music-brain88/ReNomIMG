import os
import sys
import renom as rm
import numpy as np
from tqdm import tqdm
from PIL import Image
from renom_img import __version__
from renom_img.api import Base, adddoc
from renom_img.api.utility.misc.download import download
from renom_img.api.classification import Classification
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.target import DataBuilderClassification
from renom_img.api.cnn.resnext import CnnResNeXt, Bottleneck
from renom_img.api.utility.optimizer import OptimizerResNeXt
from renom_img.api.utility.exceptions.check_exceptions import *

RESIZE_METHOD = Image.BILINEAR


class TargetBuilderResNeXt():
    '''
    Target Builder for ResNet

    '''

    def __init__(self, class_map, imsize):
        self.class_map = class_map
        self.imsize = imsize

    def __call__(self, *args, **kwargs):
        return self.build(*args, **kwargs)

    def preprocess(self, x):
        """
        Returns:
            (ndarray): Preprocessed data.

        Preprocessing for ResNeXt is as follows.
        These values are taken from the official facebook torch implementation:
        https://github.com/facebook/fb.resnet.torch/blob/master/pretrained/classify.lua

        .. math::

            x /= 255
            x[:, 0, :, :] -= 0.485
            x[:, 1, :, :] -= 0.456
            x[:, 2, :, :] -= 0.406

            x[:, 0, :, :] /= 0.229
            x[:, 1, :, :] /= 0.224
            x[:, 2, :, :] /= 0.225
        """
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

    def _load(self, path):
        """ Loads an image

        Args:
            path(str): Path to an image

        Returns:
            (tuple): Returns image data (numpy.array), the ratio of the given width to the actual image width,
                     and the ratio of the given height to the actual image height
        """
        img = Image.open(path)
        img.load()
        w, h = img.size
        img = img.convert('RGB')
        # img = img.resize(self.imsize, RESIZE_METHOD)
        img = np.asarray(img).transpose(2, 0, 1).astype(np.float32)
        return img, self.imsize[0] / float(w), self.imsize[1] / h

    def build(self, img_path_list, annotation_list=None, augmentation=None, **kwargs):
        """ Builds an array of images and corresponding labels

        Args:
            img_path_list(list): List of input image paths.
            annotation_list(list): List of class id
                                    [1, 4, 6 (int)]
            augmentation(Augmentation): Instance of the augmentation class.

        Returns:
            (tuple): Batch of images and corresponding one hot labels for each image in a batch
        """
        check_missing_param(self.class_map)
        if annotation_list is None:
            img_array = np.vstack([load_img(path, self.imsize)[None]
                                   for path in img_path_list])
            img_array = self.preprocess(img_array)
            return img_array

        # Check the class mapping.
        n_class = len(self.class_map)

        img_list = []
        label_list = []
        for img_path, an_data in zip(img_path_list, annotation_list):
            one_hot = np.zeros(n_class)
            img, sw, sh = self._load(img_path)
            img_list.append(img)
            one_hot[an_data] = 1.
            label_list.append(one_hot)

        if augmentation is not None:
            img_list, label_list = augmentation(img_list, label_list, mode="classification")

        img_list, label_list = self.resize_img(img_list, label_list)

        return self.preprocess(np.array(img_list)), np.array(label_list)


@adddoc
class ResNeXt50(Classification):
    """ResNeXt50 model.

    If the argument load_pretrained_weight is True, pretrained weights will be downloaded.
    The pretrained weights were trained using ILSVRC2012.

    Args:
        class_map (list, dict): List of class names.
        imsize (int, tuple): Input image size.
        cardinality (int): Number of groups in group convolution layers (default = 32)
        plateau (bool): Specifies whether or not error plateau learning rate adjustment should be used. If True, learning rate is
          automatically decreased when training loss reaches a plateau.
        load_pretrained_weight (bool, str): Argument specifying whether or not to load pretrained weight values.
          If True, pretrained weights will be downloaded to the current directory and loaded as the initial weight values.
          If a string is given, weight values will be loaded and initialized from the weights in the given file name.
        train_whole_network (bool): Flag specifying whether to freeze or train the base layers of the model during training.
          If True, trains all layers of the model. If False, the convolutional base is frozen during training.

    Example:
        >>> from renom_img.api.classification.resnext import ResNeXt50
        >>>
        >>> class_map = ["dog", "cat"]
        >>> model = ResNeXt50(class_map, imsize=(224,224), plateau=True, load_pretrained_weight=True, train_whole_network=True)

    Note:
        If the argument num_class is not equal to 1000, the last dense layer will be reset because
        the pretrained weight was trained on a 1000-class dataset.

    References:
        | Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, Kaiming He
        | **Aggregated Residual Transformations for Deep Neural Networks**
        | https://arxiv.org/abs/1611.05431
        |

    """

    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/classification/ResNeXt50.h5".format(
        __version__)

    def __init__(self, class_map=[], imsize=(224, 224), cardinality=32, plateau=False, load_pretrained_weight=False, train_whole_network=False):
        # exceptions checking
        check_resnext_init(plateau, cardinality)

        self.cardinality = cardinality

        self._model = CnnResNeXt(1, Bottleneck, [3, 4, 6, 3], self.cardinality)
        super(ResNeXt50, self).__init__(class_map, imsize,
                                        load_pretrained_weight, train_whole_network, self._model)

        self._model.set_output_size(self.num_class)
        self._model.set_train_whole(train_whole_network)

        self.default_optimizer = OptimizerResNeXt(plateau)
        self.decay_rate = 0.0001
        self._model.fc.params = {}

    def build_data(self):
        return TargetBuilderResNeXt(self.class_map, self.imsize)


@adddoc
class ResNeXt101(Classification):
    """ResNeXt101 model.

    If the argument load_pretrained_weight is True, pretrained weights will be downloaded.
    The pretrained weights were trained using ILSVRC2012.

    Args:
        class_map (list, dict): List of class names.
        imsize (int, tuple): Input image size.
        cardinality (int): Number of groups in group convolution layers (default = 32)
        plateau (bool): Specifies whether or not error plateau learning rate adjustment should be used. If True, learning rate is
          automatically decreased when training loss reaches a plateau.
        load_pretrained_weight (bool, str): Argument specifying whether or not to load pretrained weight values.
          If True, pretrained weights will be downloaded to the current directory and loaded as the initial weight values.
          If a string is given, weight values will be loaded and initialized from the weights in the given file name.
        train_whole_network (bool): Flag specifying whether to freeze or train the base layers of the model during training.
          If True, trains all layers of the model. If False, the convolutional base is frozen during training.

    Example:
        >>> from renom_img.api.classification.resnext import ResNeXt101
        >>>
        >>> class_map = ["dog", "cat"]
        >>> model = ResNeXt101(class_map, imsize=(224,224), plateau=True, load_pretrained_weight=True, train_whole_network=True)

    Note:
        If the argument num_class is not equal to 1000, the last dense layer will be reset because
        the pretrained weight was trained on a 1000-class dataset.

    References:
        | Saining Xie, Ross Girshick, Piotr Dollar, Zhuowen Tu, Kaiming He
        | **Aggregated Residual Transformations for Deep Neural Networks**
        | https://arxiv.org/abs/1611.05431
        |

    """

    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/classification/ResNeXt101.h5".format(
        __version__)

    def __init__(self, class_map=[], imsize=(224, 224), cardinality=32, plateau=False, load_pretrained_weight=False, train_whole_network=False):
        # exceptions checking
        check_resnext_init(plateau, cardinality)

        self.cardinality = cardinality

        self._model = CnnResNeXt(1, Bottleneck, [3, 4, 23, 3], self.cardinality)
        super(ResNeXt101, self).__init__(class_map, imsize,
                                         load_pretrained_weight, train_whole_network, self._model)

        self._model.set_output_size(self.num_class)
        self._model.set_train_whole(train_whole_network)

        self.default_optimizer = OptimizerResNeXt(plateau)
        self.decay_rate = 0.0001
        self._model.fc.params = {}

    def build_data(self):
        return TargetBuilderResNeXt(self.class_map, self.imsize)
