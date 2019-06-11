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
        # img = np.array(img.resize(self.imsize, RESIZE_METHOD))
        img = np.array(img)
        assert np.sum(np.histogram(img, bins=list(range(256)))[0][N:-1]) == 0
        assert img.ndim == 2
        return img, img.shape[0], img.shape[1]

    def _load(self, path):
        img = Image.open(path)
        img.load()
        w, h = img.size
        img = img.convert('RGB')
        # img = img.resize(self.imsize, RESIZE_METHOD)
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
            labels, asw, ash = self.load_annotation(an_path)
            annot = np.zeros((n_class+1, asw, ash))
            img_list.append(img)
            for i in range(labels.shape[0]):
                for j in range(labels.shape[1]):
                    if int(labels[i][j]) >= n_class:
                        annot[n_class, i, j] = 1
                    else:
                        annot[int(labels[i][j]), i, j] = 1
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

    If the argument load_pretrained_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using Imagenet.

    Args:
        class_map: Array of class names.
        imsize(int or tuple): Input image size.
        load_pretrained_weight(bool): Loads pretrained Xception65 backbone if True.
        train_whole_network(bool): Trains all layers in model if True.

    References:
        Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
        Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam
        https://arxiv.org/abs/1802.02611
    """

    #WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/segmentation/Deeplabv3plus.h5".format(
    #    __version__)

    def __init__(self, class_map=[], imsize=(513, 513), scale_factor=16, atrous_rates = [6,12,18], lr_initial=1e-4, lr_power=0.9, load_pretrained_weight=False, train_whole_network=False):

        self.lr_initial = lr_initial
        self.lr_power = lr_power
        #if isinstance(imsize, tuple):
        #    assert imsize[0] == imsize[1], "Current Deeplabv3+ implementation only accepts square image sizes"
        self.scale_factor = scale_factor
        self.imsize = imsize
        self.atrous_rates = atrous_rates
        self.model = CnnDeeplabv3plus(21, self.imsize, self.scale_factor, self.atrous_rates)
        super(Deeplabv3plus, self).__init__(class_map, imsize, load_pretrained_weight, train_whole_network, self.model)

        self.model.set_output_size(self.num_class)
        self.model.set_train_whole(train_whole_network)
        
        self.default_optimizer = OptimizerDeeplab(self.lr_initial, self.lr_power)
        self.decay_rate = 4e-5

    def build_data(self):
        return TargetBuilderDeeplab(self.class_map, self.imsize)

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.load(filename)

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
