import os
import sys
import numpy as np
import renom as rm
from tqdm import tqdm

from PIL import Image
from renom_img.api.cnn.fcn import CNN_FCN8s, CNN_FCN16s, CNN_FCN32s
from renom_img import __version__
from renom_img.api import adddoc
from renom_img.api.segmentation import SemanticSegmentation
from renom.utility.initializer import Initializer
from renom.config import precision
from renom.layers.function.pool2d import pool_base, max_pool2d
from renom.layers.function.utils import tuplize
from renom_img.api.utility.optimizer import FCN_Optimizer

MEAN_BGR = np.array([104.00698793, 116.66876762, 122.67891434])
RESIZE_METHOD = Image.BILINEAR


class TargetBuilderFCN():

    def __init__(self, class_map, imsize):
        self.class_map = class_map
        self.imsize = imsize

    def __call__(self, *args, **kwargs):
        return self.build(*args, **kwargs)

    def preprocess(self, x):
        """
        Preprocessing for FCN is follows.

        .. math::

            x_{red} -= 123.68 \\\\
            x_{green} -= 116.779 \\\\
            x_{blue} -= 103.939

        """
        x = x[:, ::-1, :, :]  # RGB -> BGR
        x[:, 0, :, :] -= MEAN_BGR[0]  # B
        x[:, 1, :, :] -= MEAN_BGR[1]  # G
        x[:, 2, :, :] -= MEAN_BGR[2]  # R

        return x

    def resize(self, img_list, label_list):
        x_list = []
        y_list = []
        for i, (img, label) in enumerate(zip(img_list, label_list)):
            channel_last = img.transpose(1, 2, 0)
            img = Image.fromarray(np.uint8(channel_last))
            img = img.resize(self.imsize, RESIZE_METHOD).convert('RGB')
            x_list.append(np.asarray(img))
            c, h, w = label.shape
            l = []
            for z in range(c):
                select = label[z, :, :]
                im = Image.fromarray(select)
                resized = im.resize(self.imsize, RESIZE_METHOD)
                l.append(np.array(resized))
            y_list.append(np.array(l))

        return np.asarray(x_list).transpose(0, 3, 1, 2).astype(np.float32), np.asarray(y_list)

    def load_annotation(self, path):
        """ Loads annotation data

        Args:
            path: A path of annotation file

        Returns:
            (tuple): Returns annotation data(numpy.array), the ratio of the given width to the actual image width,
        """
        N = len(self.class_map)
        img = Image.open(path)
        img.load()
        w, h = img.size
        # img = np.array(img.resize(self.imsize, RESIZE_METHOD))
        img = np.array(img)
        assert np.sum(np.histogram(img, bins=list(range(256)))[0][N:-1]) == 0
        assert img.ndim == 2
        return img, img.shape[0], img.shape[1]

    def load_img(self, path):
        img = Image.open(path)
        img.load()
        w, h = img.size
        img = img.convert('RGB')
        # img = img.resize(self.imsize, RESIZE_METHOD)
        img = np.asarray(img).transpose(2, 0, 1).astype(np.float32)
        return img, w, h

    def crop_to_square(self, image):
        size = min(image.size)
        left, upper = (image.width - size) // 2, (image.height - size) // 2
        right, bottom = (image.width + size) // 2, (image.height + size) // 2
        return image.crop((left, upper, right, bottom))

    def build(self, img_path_list, annotation_list, augmentation=None, **kwargs):
        """
        Args:
            img_path_list(list): List of input image paths.
            annotation_list(list): The format of annotation list is as follows.
            augmentation(Augmentation): Instance of the augmentation class.

        Returns:
            (tuple): Batch of images and ndarray whose shape is **(batch size, #classes, width, height)**

        """
        # Check the class mapping.
        n_class = len(self.class_map)

        img_list = []
        label_list = []
        for img_path, an_path in zip(img_path_list, annotation_list):
            img, sw, sh = self.load_img(img_path)
            labels, asw, ash = self.load_annotation(an_path)
            annot = np.zeros((n_class, asw, ash))
            img_list.append(img)
            for i in range(labels.shape[0]):
                for j in range(labels.shape[1]):
                    if int(labels[i][j]) >= n_class:
                        annot[n_class - 1, i, j] = 1
                    else:
                        annot[int(labels[i][j]), i, j] = 1
            label_list.append(annot)
        if augmentation is not None:
            img_list, label_list = augmentation(img_list, label_list, mode="segmentation")
            data,label = self.resize(img_list, label_list)
            return self.preprocess(data),label
        else:
            data,label = self.resize(img_list, label_list)   
            return self.preprocess(data),label

@adddoc
class FCN32s(SemanticSegmentation):
    """ Fully convolutional network (32s) for semantic segmentation

    Args:
        class_map(array): Array of class names
        imsize(int or tuple): Input image size
        load_pretrained_weight(bool, str): True if pre-trained weight is used, otherwise False.
        train_whole_network(bool): True if the overall model is trained, otherwise False
        train_final_upscore(bool): True if final upscore layer is trainable, otherwise False

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>> from renom_img.api.segmentation.fcn import FCN32s
        >>> n, c, h, w = (2, 12, 64, 64)
        >>> x = rm.Variable(np.random.rand(n, c, h, w))
        >>> model = FCN32s()
        >>> t = model(x)
        >>> t.shape
        (2, 12, 64, 64)

    References:
        | Jonathan Long, Evan Shelhamer, Trevor Darrell
        | **Fully Convolutional Networks for Semantic Segmentation**
        | https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
        |

    """

    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/segmentation/FCN32s.h5".format(
        __version__)

    def __init__(self, class_map=None, train_final_upscore=False, imsize=(224, 224), load_pretrained_weight=False, train_whole_network=False):

        self.model = CNN_FCN32s(1)

        super(FCN32s, self).__init__(class_map, imsize,
                                     load_pretrained_weight, train_whole_network, load_target=self.model)
        self.model.set_output_size(self.num_class)
        self.model.set_train_whole(train_whole_network, train_final_upscore)
        self.decay_rate = 5e-4
        self.default_optimizer = FCN_Optimizer()

    def build_data(self):
        return TargetBuilderFCN(self.class_map, self.imsize)

class FCN16s(SemanticSegmentation):
    """ Fully convolutional network (16s) for semantic segmentation

    Args:
        class_map(array): Array of class names
        imsize(int or tuple): Input image size
        load_pretrained_weight(bool, str): True if pre-trained weight is used, otherwise False.
        train_whole_network(bool): True if the overall model is trained, otherwise False

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>> from renom_img.api.segmentation.fcn import FCN16s
        >>> n, c, h, w = (2, 12, 64, 64)
        >>> x = rm.Variable(np.random.rand(n, c, h, w))
        >>> model = FCN16s()
        >>> t = model(x)
        >>> t.shape
        (2, 12, 64, 64)

    References:
        | Jonathan Long, Evan Shelhamer, Trevor Darrell
        | **Fully Convolutional Networks for Semantic Segmentation**
        | https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
        |

    """

    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/segmentation/FCN16s.h5".format(
        __version__)

    def __init__(self, class_map=None, train_final_upscore=False, imsize=(224, 224), load_pretrained_weight=False, train_whole_network=False):

        self.model = CNN_FCN16s(1)

        super(FCN16s, self).__init__(class_map, imsize,
                                     load_pretrained_weight, train_whole_network, load_target=self.model)

        self.model.set_train_whole(train_whole_network, train_final_upscore)
        self.model.set_output_size(self.num_class)
        self.decay_rate = 5e-4
        self.default_optimizer = FCN_Optimizer()

    def build_data(self):
        return TargetBuilderFCN(self.class_map, self.imsize)


class FCN8s(SemanticSegmentation):
    """ Fully convolutional network (8s) for semantic segmentation

    Args:
        class_map(array): Array of class names
        imsize(int or tuple): Input image size
        load_pretrained_weight(bool, str): True if pre-trained weight is used, otherwise False.
        train_whole_network(bool): True if the overall model is trained, otherwise False

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>> from renom_img.api.segmentation.fcn import FCN8s
        >>> n, c, h, w = (2, 12, 64, 64)
        >>> x = rm.Variable(np.random.rand(n, c, h, w))
        >>> model = FCN8s()
        >>> t = model(x)
        >>> t.shape
        (2, 12, 64, 64)

    References:
        | Jonathan Long, Evan Shelhamer, Trevor Darrell
        | **Fully Convolutional Networks for Semantic Segmentation**
        | https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
        |

    """

    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/segmentation/FCN8s.h5".format(
        __version__)

    def __init__(self, class_map=None, train_final_upscore=False, imsize=(224, 224), load_pretrained_weight=False, train_whole_network=False):

        self.model = CNN_FCN8s(1)

        super(FCN8s, self).__init__(class_map, imsize,
                                    load_pretrained_weight, train_whole_network, load_target=self.model)
        self.model.set_output_size(self.num_class)
        self.model.set_train_whole(train_whole_network, train_final_upscore)
        self.decay_rate = 5e-4
        self.default_optimizer = FCN_Optimizer()

    def build_data(self):
        return TargetBuilderFCN(self.class_map, self.imsize)


