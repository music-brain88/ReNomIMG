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
from renom_img.api.cnn.resnet import CnnResNet, BasicBlock, Bottleneck
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.target import DataBuilderClassification
from renom_img.api.utility.optimizer import OptimizerResNet

RESIZE_METHOD = Image.BILINEAR

class TargetBuilderResNet():
    '''
    Target Builder for ResNet
    
    '''

    def __init__(self, class_map, imsize):
        self.class_map = class_map
        self.imsize = imsize

    def __call__(self, *args, **kwargs):
        return self.build(*args, **kwargs)


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


class ResNet18(Classification):
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
    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/classification/ResNet18.h5".format(
        __version__)

    def __init__(self, class_map=None, imsize=(224, 224), plateau=False, load_pretrained_weight=False, train_whole_network=False):

        self.model = CnnResNet(1, BasicBlock, [2, 2, 2, 2])
        super(ResNet18, self).__init__(class_map, imsize,
                                       load_pretrained_weight, train_whole_network, self.model)
        
        self.model.set_output_size(self.num_class)
        self.model.set_train_whole(train_whole_network)

        self.decay_rate = 0.0001
        self.default_optimizer = OptimizerResNet()
        self.model.fc.params = {}

    def build_data(self):
        return TargetBuilderResNet(self.class_map, self.imsize)


class ResNet34(Classification):
    """ResNet34 model.

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
    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/classification/ResNet34.h5".format(
        __version__)

    def __init__(self, class_map=None, imsize=(224, 224), plateau=False, load_pretrained_weight=False, train_whole_network=False):

        self.model = CnnResNet(1, BasicBlock, [3, 4, 6, 3])
        super(ResNet34, self).__init__(class_map, imsize,
                                       load_pretrained_weight, train_whole_network, self.model)
        self.model.set_output_size(self.num_class)
        self.model.set_train_whole(train_whole_network)

        self.decay_rate = 0.0001
        self.default_optimizer = OptimizerResNet()
        self.model.fc.params = {}


    def build_data(self):
        return TargetBuilderResNet(self.class_map, self.imsize)


class ResNet50(Classification):
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
    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/classification/ResNet50.h5".format(
        __version__)

    def __init__(self, class_map=None, imsize=(224, 224), plateau=False, load_pretrained_weight=False, train_whole_network=False):

        self.model = CnnResNet(1, Bottleneck, [3, 4, 6, 3])
        super(ResNet50, self).__init__(class_map, imsize,
                                       load_pretrained_weight, train_whole_network, self.model)
        self.model.set_output_size(self.num_class)
        self.model.set_train_whole(train_whole_network)

        self.decay_rate = 0.0001
        self.default_optimizer = OptimizerResNet()
        self.model.fc.params = {}

    def build_data(self):
        return TargetBuilderResNet(self.class_map, self.imsize)

class ResNet101(Classification):
    """ResNet101 model.

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
    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/classification/ResNet101.h5".format(
        __version__)

    def __init__(self, class_map=None, imsize=(224, 224), plateau=False, load_pretrained_weight=False, train_whole_network=False):

        self.model = CnnResNet(1, Bottleneck, [3, 4, 23, 3])
        super(ResNet101, self).__init__(class_map, imsize,
                                        load_pretrained_weight, train_whole_network, self.model)
        self.model.set_output_size(self.num_class)
        self.model.set_train_whole(train_whole_network)

        self.decay_rate = 0.0001
        self.default_optimizer = OptimizerResNet()
        self.model.fc.params = {}


    def build_data(self):
        return TargetBuilderResNet(self.class_map, self.imsize)


class ResNet152(Classification):
    """ResNet152 model.

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
    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/classification/ResNet152.h5".format(
        __version__)

    def __init__(self, class_map=None, imsize=(224, 224), plateau=False, load_pretrained_weight=False, train_whole_network=False):

        self.model = CnnResNet(1, Bottleneck, [3, 8, 36, 3])
        super(ResNet152, self).__init__(class_map, imsize,
                                        load_pretrained_weight, train_whole_network, self.model)
        self.model.set_output_size(self.num_class)
        self.model.set_train_whole(train_whole_network)

        self.decay_rate = 0.0001
        self.default_optimizer = OptimizerResNet()
        self.model.fc.params = {}

    def build_data(self):
        return TargetBuilderResNet(self.class_map, self.imsize)

