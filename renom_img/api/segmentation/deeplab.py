import os
import sys
import numpy as np
import renom as rm
from tqdm import tqdm

from PIL import Image
from renom_img.api.cnn.deeplab import CnnDeeplabv3plus
from renom_img import __version__
from renom_img.api import adddoc
from renom_img.api.segmentation import SemanticSegmentation
from renom_img.api.utility.optimizer import OptimizerDeeplab
from renom_img.api.utility.load import load_img
from renom_img.api.utility.exceptions.check_exceptions import *

RESIZE_METHOD = Image.BILINEAR

class TargetBuilderDeeplab():
    def __init__(self, class_map, imsize):
        self.class_map = class_map
        self.imsize = imsize

    def __call__(self, *args, **kwargs):
        return self.build(*args, **kwargs)

    def preprocess(self, x):
        """Image preprocessing for Deeplabv3+ training.

        :math:`new_x = 2*x/255 - 1.`

        Returns:
            (ndarray): Preprocessed data.
        """
        x = 2.*x/255. - 1.0
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
        img = np.array(img)
        check_segmentation_label(img, N)
        return img, img.shape[0], img.shape[1]

    def _load(self, path):
        img = Image.open(path)
        img.load()
        w, h = img.size
        img = img.convert('RGB')
        img = np.asarray(img).transpose(2, 0, 1).astype(np.float32)
        return img, w, h

    def build(self, img_path_list, annotation_list=None, augmentation=None, **kwargs):
        """
        Args:
            img_path_list(list): List of input image paths.
            annotation_list(list): The format of annotation list is as follows.
            augmentation(Augmentation): Instance of the augmentation class.

        Returns:
            (tuple): Batch of images and ndarray whose shape is **(batch size, #classes, width, height)**

        """
        check_missing_param(self.class_map)
        if annotation_list is None:
            img_array = np.vstack([load_img(path,self.imsize)[None]
                                    for path in img_path_list])
            img_array = self.preprocess(img_array)

            return img_array

        # Check the class mapping.
        n_class = len(self.class_map)

        img_list = []
        label_list = []
        for img_path, an_path in zip(img_path_list, annotation_list):
            img, sw, sh = self._load(img_path)
            label, asw, ash = self.load_annotation(an_path)
            annot = np.zeros((n_class+1, label.shape[0], label.shape[1]))
            for i in range(label.shape[0]):
                for j in range(label.shape[1]):
                    if int(label[i][j]) >= n_class:
                        annot[n_class, i, j] = 1
                    else:
                        annot[int(label[i][j]), i, j] = 1
            img_list.append(img)
            label_list.append(annot[:n_class])
        if augmentation is not None:
            img_list, label_list = augmentation(img_list, label_list, mode="segmentation")
            data,label = self.resize(img_list, label_list)
            return self.preprocess(data),label
        else:
            data,label = self.resize(img_list, label_list)   
            return self.preprocess(data),label


@adddoc
class Deeplabv3plus(SemanticSegmentation):
    """Deeplabv3+ model with modified aligned Xception65 backbone.

    Args:
        class_map (list, dict): List of class names.
        imsize (int, tuple): Image size after rescaling. Must be set to (321,321) in current implementation, which only supports a fixed rescaled size of 321x321.
        scale_factor (int): Reduction factor for output feature maps before upsampling. Current implementation only supports a value of 16.
        atrous_rates (list): List of dilation factors in ASPP module atrous convolution layers. Current implementation only supports values of [6,12,18].
        lr_initial (float): Initial learning rate for poly learning rate schedule. The default value is 1e-4.
        lr_power (float): Exponential factor for poly learning rate schedule. The default value is 0.9
        load_pretrained_weight (bool, str): Argument specifying whether or not to load pretrained weight values.
          If True, pretrained weights will be downloaded to the current directory and loaded as the initial weight values.
          If a string is given, weight values will be loaded and initialized from the weights in the given file name.
        train_whole_network (bool): Flag specifying whether to freeze or train the base encoder layers of the model during training.
          If True, trains all layers of the model. If False, the convolutional encoder base is frozen during training.

    References:
        | Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam
        | **Rethinking Atrous Convolution for Semantic Image Segmentation**
        | https://arxiv.org/abs/1706.05587
        | 
        | Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam
        | **Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation**
        | https://arxiv.org/abs/1802.02611
        |

    """
    WEIGHT_URL = CnnDeeplabv3plus.WEIGHT_URL

    def __init__(self, class_map=[], imsize=(321, 321), scale_factor=16, atrous_rates = [6,12,18], lr_initial=1e-4, lr_power=0.9, load_pretrained_weight=False, train_whole_network=False):
        check_deeplabv3plus_init(imsize, scale_factor, atrous_rates, lr_initial, lr_power)
        self.lr_initial = lr_initial
        self.lr_power = lr_power
        self.scale_factor = scale_factor
        self.imsize = imsize
        self.atrous_rates = atrous_rates
        self._model = CnnDeeplabv3plus(21, self.imsize, self.scale_factor, self.atrous_rates)
        super(Deeplabv3plus, self).__init__(class_map, imsize, load_pretrained_weight, train_whole_network, self._model)

        self._model.set_output_size(self.num_class)
        self._model.set_train_whole(train_whole_network)
        
        self.default_optimizer = OptimizerDeeplab(self.lr_initial, self.lr_power)
        self.decay_rate = 4e-5

    def build_data(self):
        return TargetBuilderDeeplab(self.class_map, self.imsize)

    def loss(self, x, y, class_weight=None):
        if class_weight is not None and class_weight:
            mask = np.concatenate(
                [np.ones((y.shape[0], 1, y.shape[2], y.shape[3])) * c for c in class_weight], axis=1)
            loss = rm.softmax_cross_entropy(x, y, reduce_sum=False)
            loss *= mask.astype(y.dtype)
            loss = rm.sum(loss)
        else:
            loss = rm.softmax_cross_entropy(x, y) * float(len(x))
        non_void_pixels = rm.sum(y[y==1])
        return loss/non_void_pixels
