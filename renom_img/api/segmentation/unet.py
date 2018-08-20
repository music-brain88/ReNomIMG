import os
import sys
import numpy as np
import renom as rm
from tqdm import tqdm

from renom_img.api import adddoc
from renom_img.api.utility.misc.download import download
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.load import load_img
from renom_img.api.utility.target import DataBuilderSegmentation
from renom_img.api.segmentation import SemanticSegmentation

DIR = os.path.split(os.path.abspath(__file__))[0]


@adddoc
class UNet(SemanticSegmentation):
    """ U-Net: Convolutional Networks for Biomedical Image Segmentation

    Args:
        class_map(array): Array of class names
        imsize(int or tuple): Input image size
        load_pretrained_weight(bool, str): True if pre-trained weight is used, otherwise False.
        train_whole_network(bool): True if the overall model is trained, otherwise False

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>> from renom_img.api.segmentation.unet import UNet
        >>> n, c, h, w = (2, 12, 64, 64)
        >>> x = rm.Variable(np.random.rand(n, c, h, w))
        >>> model = UNet()
        >>> t = model(x)
        >>> t.shape
        (2, 12, 64, 64)

    References:
        | Olaf Ronneberger, Philipp Fischer, Thomas Brox
        | **U-Net: Convolutional Networks for Biomedical Image Segmentation**
        | https://arxiv.org/pdf/1505.04597.pdf
        |

    """

    def __init__(self, class_map=[], imsize=(512, 512), load_pretrained_weight=False, train_whole_network=False):
        assert not load_pretrained_weight, "Currently pretrained weight of %s is not prepared. Please set False to `load_pretrained_weight` flag." % self.__class__.__name__
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        self.imsize = imsize
        self.num_class = len(class_map)
        self.class_map = [c.encode("ascii", "ignore") for c in class_map]
        self._model = CNN_UNet(self.num_class)
        self._train_whole_network = train_whole_network
        self._opt = rm.Adam(lr=1e-3)
        self._freeze()

    def preprocess(self, x):
        """Image preprocess for U-Net.

        :math:`new_x = x/255`

        Returns:
            (ndarray): Preprocessed data.
        """
        return x / 255.

    def get_optimizer(self, current_epoch=None, total_epoch=None, current_batch=None, total_batch=None, **kwargs):
        if any([num is None for num in [current_epoch, total_epoch, current_batch, total_batch]]):
            return self._opt
        else:
            ind1 = int(total_epoch * 0.5)
            ind2 = int(total_epoch * 0.3)
            ind3 = total_epoch - (ind1 + ind2 + 1)
            lr_list = [0] + [0.01] * ind1 + [0.001] * ind2 + [0.0001] * ind3
            if current_epoch == 0:
                lr = 0.0001
            else:
                lr = lr_list[current_epoch]
            self._opt._lr = lr
            return self._opt

    def regularize(self):
        reg = 0
        for layer in self.iter_models():
            if hasattr(layer, "params") and hasattr(layer.params, "w"):
                reg += rm.sum(layer.params.w * layer.params.w)
        return 0.0004 * reg

    def _freeze(self):
        self._model.conv1_1.set_auto_update(self._train_whole_network)
        self._model.conv1_2.set_auto_update(self._train_whole_network)
        self._model.conv2_1.set_auto_update(self._train_whole_network)
        self._model.conv2_2.set_auto_update(self._train_whole_network)
        self._model.conv3_1.set_auto_update(self._train_whole_network)
        self._model.conv3_2.set_auto_update(self._train_whole_network)
        self._model.conv4_1.set_auto_update(self._train_whole_network)
        self._model.conv4_2.set_auto_update(self._train_whole_network)
        self._model.conv5_1.set_auto_update(self._train_whole_network)
        self._model.conv5_2.set_auto_update(self._train_whole_network)


class CNN_UNet(rm.Model):
    def __init__(self, num_class):
        self.conv1_1 = rm.Conv2d(64, padding=1, filter=3)
        self.conv1_2 = rm.Conv2d(64, padding=1, filter=3)
        self.conv2_1 = rm.Conv2d(128, padding=1, filter=3)
        self.conv2_2 = rm.Conv2d(128, padding=1, filter=3)
        self.conv3_1 = rm.Conv2d(256, padding=1, filter=3)
        self.conv3_2 = rm.Conv2d(256, padding=1, filter=3)
        self.conv4_1 = rm.Conv2d(512, padding=1, filter=3)
        self.conv4_2 = rm.Conv2d(512, padding=1, filter=3)
        self.conv5_1 = rm.Conv2d(1024, padding=1, filter=3)
        self.conv5_2 = rm.Conv2d(1024, padding=1, filter=3)

        self.deconv1 = rm.Deconv2d(512, stride=2)
        self.conv6_1 = rm.Conv2d(256, padding=1)
        self.conv6_2 = rm.Conv2d(256, padding=1)
        self.deconv2 = rm.Deconv2d(256, stride=2)
        self.conv7_1 = rm.Conv2d(128, padding=1)
        self.conv7_2 = rm.Conv2d(128, padding=1)
        self.deconv3 = rm.Deconv2d(128, stride=2)
        self.conv8_1 = rm.Conv2d(64, padding=1)
        self.conv8_2 = rm.Conv2d(64, padding=1)
        self.deconv4 = rm.Deconv2d(64, stride=2)
        self.conv9 = rm.Conv2d(num_class, filter=1)

    def forward(self, x):
        t = rm.relu(self.conv1_1(x))
        c1 = rm.relu(self.conv1_2(t))
        t = rm.max_pool2d(c1, filter=2, stride=2)
        t = rm.relu(self.conv2_1(t))
        c2 = rm.relu(self.conv2_2(t))
        t = rm.max_pool2d(c2, filter=2, stride=2)
        t = rm.relu(self.conv3_1(t))
        c3 = rm.relu(self.conv3_2(t))
        t = rm.max_pool2d(c3, filter=2, stride=2)
        t = rm.relu(self.conv4_1(t))
        c4 = rm.relu(self.conv4_2(t))
        t = rm.max_pool2d(c4, filter=2, stride=2)
        t = rm.relu(self.conv5_1(t))
        t = rm.relu(self.conv5_2(t))

        t = self.deconv1(t)[:, :, :c4.shape[2], :c4.shape[3]]
        t = rm.concat([c4, t])
        t = rm.relu(self.conv6_1(t))
        t = rm.relu(self.conv6_2(t))
        t = self.deconv2(t)[:, :, :c3.shape[2], :c3.shape[3]]
        t = rm.concat([c3, t])

        t = rm.relu(self.conv7_1(t))
        t = rm.relu(self.conv7_2(t))
        t = self.deconv3(t)[:, :, :c2.shape[2], :c2.shape[3]]
        t = rm.concat([c2, t])

        t = rm.relu(self.conv8_1(t))
        t = rm.relu(self.conv8_2(t))
        t = self.deconv4(t)[:, :, :c1.shape[2], :c1.shape[3]]
        t = rm.concat([c1, t])

        t = self.conv9(t)

        return t
