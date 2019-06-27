from __future__ import print_function, division
import os
import numpy as np
import renom as rm
from tqdm import tqdm
from PIL import Image
from renom_img import __version__
from renom_img.api import Base, adddoc
from renom_img.api.classification import Classification
from renom_img.api.utility.misc.download import download
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.cnn.vgg import CNN_VGG19, CNN_VGG16, CNN_VGG16_NODENSE, CNN_VGG11
from renom_img.api.utility.optimizer import OptimizerVGG
from renom_img.api.utility.exceptions.check_exceptions import *
from renom_img.api.utility.exceptions.exceptions import FunctionNotImplementedError

RESIZE_METHOD = Image.BILINEAR


class TargetBuilderVGG():
    '''
    Target Builder for VGG

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

        Preprocessing for VGG is as follows.

        .. math::

            x_{red} -= 123.68 \\
            x_{green} -= 116.779 \\
            x_{blue} -= 103.939 \\

        """
        x[:, 0, :, :] -= 123.68  # R
        x[:, 1, :, :] -= 116.779  # G
        x[:, 2, :, :] -= 103.939  # B
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

        Returns:            (tuple): Batch of images and corresponding one hot labels for each image in a batch
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
class VGG11(Classification):
    """VGG11 model.

    Args:
        class_map (list, dict): List of class names.
        imsize (int, tuple): Input image size.
        load_pretrained_weight (bool, str): Argument specifying whether or not to load pretrained weight values.
          Pretrained weights have not been prepared for VGG-11 yet, so this must either be set to False, or to a
          user-defined string that specifies a local pretrained weights filename.
        train_whole_network (bool): Flag specifying whether to freeze or train the base layers of the model during training.
          If true, trains all layers of the model. If False, the convolutional base is frozen during training.

    Note:
        If the argument num_class is not equal to 1000, the last dense layer will be reset because
        the pretrained weight was trained on a 1000-class dataset.

    References:
        | Karen Simonyan, Andrew Zisserman
        | **Very Deep Convolutional Networks for Large-Scale Image Recognition**
        | https://arxiv.org/abs/1409.1556
        |

    """

    def __init__(self, class_map=None, imsize=(224, 224),
                 load_pretrained_weight=False, train_whole_network=False):

        self._model = CNN_VGG11()
        if load_pretrained_weight:
            raise FunctionNotImplementedError("The pretrained weights for %s are not currently available. Please set `load_pretrained_weight` to False.".format(self.__class__.__name__))
        super(VGG11, self).__init__(class_map, imsize, load_pretrained_weight,
                                    train_whole_network, self._model)

        self._model.set_output_size(self.num_class)
        self._model.set_train_whole(train_whole_network)

        self.default_optimizer = OptimizerVGG()
        self.decay_rate = 0.0005

        self._model.fc1.params = {}
        self._model.fc2.params = {}
        self._model.fc3.params = {}

    def build_data(self):
        return TargetBuilderVGG(self.class_map, self.imsize)


@adddoc
class VGG16(Classification):
    """VGG16 model.

    If the argument load_pretrained_weight is True, pretrained weights will be downloaded.
    The pretrained weights were trained using ILSVRC2012.

    Args:
        class_map (list, dict): List of class names.
        imsize (int or tuple): Input image size.
        load_pretrained_weight (bool, str): Argument specifying whether or not to load pretrained weight values.
          If True, pretrained weights will be downloaded to the current directory and loaded as the initial weight values.
          If a string is given, weight values will be loaded and initialized from the weights in the given file name.
        train_whole_network (bool): Flag specifying whether to freeze or train the base layers of the model during training.
          If True, trains all layers of the model. If False, the convolutional base is frozen during training.

    Note:
        If the argument num_class is not equal to 1000, the last dense layer will be reset because
        the pretrained weight was trained on a 1000-class dataset.

    References:
        | Karen Simonyan, Andrew Zisserman
        | **Very Deep Convolutional Networks for Large-Scale Image Recognition**
        | https://arxiv.org/abs/1409.1556
        |

    """
    WEIGHT_URL = CNN_VGG16.WEIGHT_URL

    def __init__(self, class_map=None, imsize=(224, 224),
                 load_pretrained_weight=False, train_whole_network=False):

        self._model = CNN_VGG16()
        super(VGG16, self).__init__(class_map, imsize, load_pretrained_weight,
                                    train_whole_network, self._model)

        self._model.set_output_size(self.num_class)
        self._model.set_train_whole(train_whole_network)

        self.default_optimizer = OptimizerVGG()
        self.decay_rate = 0.0005

        self._model.fc1.params = {}
        self._model.fc2.params = {}
        self._model.fc3.params = {}

    def build_data(self):
        return TargetBuilderVGG(self.class_map, self.imsize)


class VGG16_NODENSE(Classification):

    def __init__(self, class_map=None, imsize=(224, 224),
                 load_pretrained_weight=False, train_whole_network=False):

        self._model = CNN_VGG16_NODENSE()
        super(VGG16, self).__init__(class_map, imsize, load_pretrained_weight,
                                    train_whole_network, self._model, load_target=self._model)

        self._model.set_output_size(self.num_class)
        self._model.set_train_whole(train_whole_network)

        self.default_optimizer = OptimizerVGG()
        self.decay_rate = 0.0005

        self._model.fc1.params = {}
        self._model.fc2.params = {}
        self._model.fc3.params = {}

    def build_data(self):
        return TargetBuilderVGG(self.class_map, self.imsize)


@adddoc
class VGG19(Classification):
    """VGG19 model.

    If the argument load_pretrained_weight is True, pretrained weights will be downloaded.
    The pretrained weights were trained using ILSVRC2012.

    Args:
        class_map (list, dict): List of class names.
        imsize (int, tuple): Input image size.
        load_pretrained_weight (bool, str): Argument specifying whether or not to load pretrained weight values.
          If True, pretrained weights will be downloaded to the current directory and loaded as the initial weight values.
          If a string is given, weight values will be loaded and initialized from the weights in the given file name.
        train_whole_network (bool): Flag specifying whether to freeze or train the base layers of the model during training.
          If True, trains all layers of the model. If False, the convolutional base is frozen during training.

    Note:
        If the argument num_class is not equal to 1000, the last dense layer will be reset because
        the pretrained weight was trained on a 1000-class dataset.

    References:
        | Karen Simonyan, Andrew Zisserman
        | **Very Deep Convolutional Networks for Large-Scale Image Recognition**
        | https://arxiv.org/abs/1409.1556
        |

    """
    WEIGHT_URL = CNN_VGG19.WEIGHT_URL

    def __init__(self, class_map=None, imsize=(224, 224),
                 load_pretrained_weight=False, train_whole_network=False):

        self._model = CNN_VGG19()
        super(VGG19, self).__init__(class_map, imsize,
                                    load_pretrained_weight, train_whole_network, self._model)

        self._model.set_output_size(self.num_class)
        self._model.set_train_whole(train_whole_network)

        self.default_optimizer = OptimizerVGG()
        self.decay_rate = 0.0005

        self._model.fc1.params = {}
        self._model.fc2.params = {}
        self._model.fc3.params = {}

    def build_data(self):
        return TargetBuilderVGG(self.class_map, self.imsize)
