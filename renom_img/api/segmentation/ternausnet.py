import os
import sys
import numpy as np
import renom as rm
from tqdm import tqdm
import inspect

from renom.layers.function.utils import col2im, im2col, tuplize
from renom.core import Node, Variable, to_value
from renom import precision
from renom.layers.function.parameterized import Parametrized
from renom_img import __version__
from renom_img.api import adddoc
from renom_img.api.utility.misc.download import download
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.load import load_img
from renom_img.api.utility.target import DataBuilderSegmentation
from renom_img.api.utility.exceptions.check_exceptions import check_segmentation_label, check_ternausnet_init
from renom_img.api.segmentation import SemanticSegmentation
from renom.utility.initializer import GlorotNormal, GlorotUniform
from PIL import Image
from renom_img.api.cnn.ternausnet import CNN_TernausNet
from renom_img.api.utility.optimizer import OptimizerTernausNet
import renom.cuda as cu
if cu.has_cuda():
    from renom.cuda.gpuvalue import GPUValue, get_gpu

RESIZE_METHOD = Image.BILINEAR


class TargetBuilderTernausNet():
    def __init__(self, class_map, imsize):
        self.class_map = class_map
        self.imsize = imsize

    def __call__(self, *args, **kwargs):
        return self.build(*args, **kwargs)

    def preprocess(self, x):
        """Image preprocessing for TernausNet training.

        :math:`new_x = (x-mean)/(255*std).`

        Returns:
            (ndarray): Preprocessed data.
        """
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        x = x / 255.
        x[:, 0, :, :] -= mean[0]  # R
        x[:, 1, :, :] -= mean[1]  # G
        x[:, 2, :, :] -= mean[2]  # B
        x[:, 0, :, :] /= std[0]  # R
        x[:, 1, :, :] /= std[1]  # G
        x[:, 2, :, :] /= std[2]  # B
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
        check_segmentation_label(img, N)
        return img, img.shape[0], img.shape[1]

    def _load(self, path):
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

    def build(self, img_path_list, annotation_list=None, augmentation=None, **kwargs):
        """
        Args:
            img_path_list(list): List of input image paths.
            annotation_list(list): The format of annotation list is as follows.
            augmentation(Augmentation): Instance of the augmentation class.

        Returns:
            (tuple): Batch of images and ndarray whose shape is **(batch size, #classes, width, height)**

        """
        if annotation_list is None:
            img_array = np.vstack([load_img(path, self.imsize)[None]
                                   for path in img_path_list])
            img_array = self.preprocess(img_array)

            return img_array

        # Check the class mapping.
        n_class = len(self.class_map)

        img_list = []
        label_list = []
        for img_path, an_path in zip(img_path_list, annotation_list):
            img, sw, sh = self._load(img_path)
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
            data, label = self.resize(img_list, label_list)
            return self.preprocess(data), label
        else:
            data, label = self.resize(img_list, label_list)
            return self.preprocess(data), label


@adddoc
class TernausNet(SemanticSegmentation):
    """ TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation

    Args:
        class_map (list, dict): List of class names.
        imsize (int, tuple): Input image size.
        load_pretrained_weight (bool, str): Argument specifying whether or not to load pretrained weight values.
          If True, pretrained weights will be downloaded to the current directory and loaded as the initial weight values.
          If a string is given, weight values will be loaded and initialized from the weights in the given file name.
        train_whole_network (bool): Flag specifying whether to freeze or train the base encoder layers of the model during training.
          If True, trains all layers of the model. If False, the convolutional encoder base is frozen during training.

    Example:
        >>> from renom_img.api.segmentation.ternausnet import TernausNet
        >>>
        >>> class_map = ['background', 'object']
        >>> model = TernausNet(class_map, imsize=(224,224), load_pretrained_weight=True, train_whole_network=True)

    References:
        | Vladimir Iglovikov, Alexey Shvets
        | **TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation**
        | https://arxiv.org/pdf/1801.05746.pdf
        |

    """

    WEIGHT_URL = CNN_TernausNet.WEIGHT_URL

    def __init__(self, class_map=None, imsize=(224, 224), load_pretrained_weight=False, train_whole_network=False):

        # make int into array
        if isinstance(imsize, int):
            imsize = (imsize, imsize)
        check_ternausnet_init(imsize)
        self._model = CNN_TernausNet(1)

        super(TernausNet, self).__init__(class_map, imsize,
                                         load_pretrained_weight, train_whole_network, load_target=self._model)
        self._model.set_output_size(self.num_class)
        self._model.set_train_whole(train_whole_network)
        self.decay_rate = 0
        self.default_optimizer = OptimizerTernausNet()

    def forward(self, x):
        self._model.set_output_size(self.num_class)
        return self._model(x)

    def build_data(self):
        return TargetBuilderTernausNet(self.class_map, self.imsize)
