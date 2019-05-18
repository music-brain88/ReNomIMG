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
    pass

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

    def __init__(self, class_map=[], imsize=(513, 513), dilation=2, atrous_rates = [6,12,18], load_pretrained_weight=False, train_whole_network=False):

        self.model = CnnDeeplabv3plus(21, dilation, atrous_rates, imsize)
        super(Deeplabv3plus, self).__init__(class_map, imsize, load_pretrained_weight, train_whole_network, self.model)

        self.model.set_output_size(self.num_class)
        self.model.set_train_whole(train_whole_network)
        
        self.default_optimizer = OptimizerDeeplab()
        self.decay_rate = 4e-5

    def build_data(self):
        return TargetBuilderDeeplabv3plus(self.class_map, self.imsize)

    def save(self, filename):
        self.model.save(filename)

    def load(self, filename):
        self.model.load(filename)

