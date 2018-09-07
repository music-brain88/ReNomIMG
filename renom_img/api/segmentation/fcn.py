import os
import sys
import numpy as np
import renom as rm
from tqdm import tqdm
from renom_img.api import adddoc
from renom_img.api.segmentation import SemanticSegmentation
from renom_img.api.classification.vgg import VGG16

DIR = os.path.split(os.path.abspath(__file__))[0]


def layer_factory(channel=32, conv_layer_num=2):
    layers = []
    for _ in range(conv_layer_num):
        layers.append(rm.Conv2d(channel=channel, padding=1, filter=3))
        layers.append(rm.Relu())
    layers.append(rm.MaxPool2d(filter=2, stride=2))
    return rm.Sequential(layers)


@adddoc
class FCN_Base(SemanticSegmentation):
    def get_optimizer(self, current_epoch=None, total_epoch=None, current_batch=None, total_batch=None, **kwargs):
        if current_epoch == 100:
            self._opt._lr = 1e-4
        elif current_epoch == 150:
            self._opt._lr = 1e-5
        return self._opt

    def preprocess(self, x):
        """
        Preprocessing for FCN is follows.

        .. math::

            x_{red} -= 123.68 \\\\
            x_{green} -= 116.779 \\\\
            x_{blue} -= 103.939

        """

        x[:, 0, :, :] -= 123.68  # R
        x[:, 1, :, :] -= 116.779  # G
        x[:, 2, :, :] -= 103.939  # B
        return x


class FCN32s(FCN_Base):
    """ Fully convolutional network (21s) for semantic segmentation

    Args:
        class_map(array): Array of class names
        imsize(int or tuple): Input image size
        load_pretrained_weight(bool, str): True if pre-trained weight is used, otherwise False.
        train_whole_network(bool): True if the overall model is trained, otherwise False

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

    SERIALIZED = ("imsize", "class_map", "num_class")
    WEIGHT_URL = "http://docs.renom.jp/downloads/weights/VGG16.h5"

    def __init__(self, class_map=[], imsize=(224, 224), load_pretrained_weight=False, train_whole_network=False):
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        self.imsize = imsize
        self.num_class = len(class_map)
        self.class_map = [c.encode("ascii", "ignore") for c in class_map]
        self._model = CNN_FCN32s(self.num_class)
        self._train_whole_network = train_whole_network
        self.decay_rate = 2e-4
        self._opt = rm.Sgd(0.001, 0.9)

        if load_pretrained_weight:
            vgg16 = VGG16(class_map, load_pretrained_weight=load_pretrained_weight,
                          train_whole_network=train_whole_network)
            self._model.block1 = vgg16._model.block1
            self._model.block2 = vgg16._model.block2
            self._model.block3 = vgg16._model.block3
            self._model.block4 = vgg16._model.block4
            self._model.block5 = vgg16._model.block5

    def _freeze(self):
        self._model.block1.set_auto_update(self._train_whole_network)
        self._model.block2.set_auto_update(self._train_whole_network)
        self._model.block3.set_auto_update(self._train_whole_network)
        self._model.block4.set_auto_update(self._train_whole_network)
        self._model.block5.set_auto_update(self._train_whole_network)


class FCN16s(FCN_Base):
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

    SERIALIZED = ("imsize", "class_map", "num_class")
    WEIGHT_URL = "http://docs.renom.jp/downloads/weights/VGG16.h5"

    def __init__(self, class_map=[], imsize=(224, 224), load_pretrained_weight=False, train_whole_network=False):
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        self.imsize = imsize
        self.num_class = len(class_map)
        self.class_map = [c.encode("ascii", "ignore") for c in class_map]
        self._model = CNN_FCN16s(self.num_class)
        self._train_whole_network = train_whole_network
        self._opt = rm.Sgd(0.001, 0.9)

        if load_pretrained_weight:
            vgg16 = VGG16(class_map, load_pretrained_weight=load_pretrained_weight,
                          train_whole_network=train_whole_network)
            self._model.block1 = vgg16._model.block1
            self._model.block2 = vgg16._model.block2
            self._model.block3 = vgg16._model.block3
            self._model.block4 = vgg16._model.block4
            self._model.block5 = vgg16._model.block5

    def _freeze(self):
        self._model.block1.set_auto_update(self._train_whole_network)
        self._model.block2.set_auto_update(self._train_whole_network)
        self._model.block3.set_auto_update(self._train_whole_network)
        self._model.block4.set_auto_update(self._train_whole_network)
        self._model.block5.set_auto_update(self._train_whole_network)


class FCN8s(FCN_Base):
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

    SERIALIZED = ("imsize", "class_map", "num_class")
    WEIGHT_URL = "http://docs.renom.jp/downloads/weights/VGG16.h5"

    def __init__(self, class_map=[], imsize=(224, 224), load_pretrained_weight=False, train_whole_network=False):
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        self.imsize = imsize
        self.num_class = len(class_map)
        self.class_map = [c.encode("ascii", "ignore") for c in class_map]
        self._model = CNN_FCN8s(self.num_class)
        self._train_whole_network = train_whole_network
        self._opt = rm.Sgd(0.001, 0.9)

        if load_pretrained_weight:
            vgg16 = VGG16(class_map, load_pretrained_weight=load_pretrained_weight,
                          train_whole_network=train_whole_network)
            self._model.block1 = vgg16._model.block1
            self._model.block2 = vgg16._model.block2
            self._model.block3 = vgg16._model.block3
            self._model.block4 = vgg16._model.block4
            self._model.block5 = vgg16._model.block5

    def _freeze(self):
        self._model.block1.set_auto_update(self._train_whole_network)
        self._model.block2.set_auto_update(self._train_whole_network)
        self._model.block3.set_auto_update(self._train_whole_network)
        self._model.block4.set_auto_update(self._train_whole_network)
        self._model.block5.set_auto_update(self._train_whole_network)


class CNN_FCN8s(rm.Model):
    def __init__(self, num_class):

        self.block1 = layer_factory(channel=64, conv_layer_num=2)
        self.block2 = layer_factory(channel=128, conv_layer_num=2)
        self.block3 = layer_factory(channel=256, conv_layer_num=3)
        self.block4 = layer_factory(channel=512, conv_layer_num=3)
        self.block5 = layer_factory(channel=512, conv_layer_num=3)

        self.fc6 = rm.Conv2d(4096, filter=7, padding=3)
        self.dr1 = rm.Dropout(0.5)
        self.fc7 = rm.Conv2d(4096, filter=1)
        self.dr2 = rm.Dropout(0.5)

        self.score_fr = rm.Conv2d(num_class, filter=1)
        self.upscore2 = rm.Deconv2d(num_class, filter=2, stride=2, padding=0)
        self.upscore8 = rm.Deconv2d(num_class, filter=8, stride=8, padding=0)

        self.score_pool3 = rm.Conv2d(num_class, filter=1)
        self.score_pool4 = rm.Conv2d(num_class, filter=1)

        self.upscore_pool4 = rm.Deconv2d(num_class, filter=2, stride=2, padding=0)

    def forward(self, x):
        t = x
        t = self.block1(t)
        t = self.block2(t)
        t = self.block3(t)
        pool3 = t
        t = self.block4(t)
        pool4 = t
        t = self.block5(t)

        t = rm.relu(self.fc6(t))
        t = self.dr1(t)
        fc6 = t

        t = rm.relu(self.fc7(t))
        t = self.dr2(t)
        fc7 = t

        t = self.score_fr(t)
        score_fr = t

        t = self.upscore2(t)
        upscore2 = t

        t = self.score_pool4(pool4)
        score_pool4 = t

        t = upscore2 + score_pool4
        fuse_pool4 = t

        t = self.score_pool3(pool3)
        score_pool3 = t

        t = self.upscore_pool4(fuse_pool4)
        upscore_pool4 = t
        t = upscore_pool4 + score_pool3

        t = self.upscore8(t)
        return t


class CNN_FCN16s(rm.Model):
    def __init__(self, num_class):
        self.block1 = layer_factory(channel=64, conv_layer_num=2)
        self.block2 = layer_factory(channel=128, conv_layer_num=2)
        self.block3 = layer_factory(channel=256, conv_layer_num=3)
        self.block4 = layer_factory(channel=512, conv_layer_num=3)
        self.block5 = layer_factory(channel=512, conv_layer_num=3)

        self.fc6 = rm.Conv2d(4096, filter=7, padding=3)
        self.fc7 = rm.Conv2d(4096, filter=1)

        self.score_fr = rm.Conv2d(num_class, filter=1)
        self.score_pool4 = rm.Conv2d(num_class, filter=1)

        self.upscore2 = rm.Deconv2d(num_class, filter=2, stride=2, padding=0)
        self.upscore16 = rm.Deconv2d(num_class, filter=16, stride=16, padding=0)

    def forward(self, x):
        t = x
        t = self.block1(t)
        t = self.block2(t)
        t = self.block3(t)
        t = self.block4(t)
        pool4 = t
        t = self.block5(t)

        t = rm.relu(self.fc6(t))

        t = rm.relu(self.fc7(t))

        t = self.score_fr(t)

        t = self.upscore2(t)
        upscore2 = t

        t = self.score_pool4(pool4)
        score_pool4 = t

        t = upscore2 + score_pool4
        fuse_pool4 = t
        t = self.upscore16(fuse_pool4)
        upscore16 = t

        return t


class CNN_FCN32s(rm.Model):
    def __init__(self, num_class):
        self.block1 = layer_factory(channel=64, conv_layer_num=2)
        self.block2 = layer_factory(channel=128, conv_layer_num=2)
        self.block3 = layer_factory(channel=256, conv_layer_num=3)
        self.block4 = layer_factory(channel=512, conv_layer_num=3)
        self.block5 = layer_factory(channel=512, conv_layer_num=3)

        self.fc6 = rm.Conv2d(4096, filter=7, padding=3)
        self.fc7 = rm.Conv2d(4096, filter=1)

        self.score_fr = rm.Conv2d(num_class, filter=1)  # n_classes
        self.upscore = rm.Deconv2d(num_class, stride=32, padding=0, filter=32)  # n_classes

    def forward(self, x):
        t = x
        t = self.block1(t)
        t = self.block2(t)
        t = self.block3(t)
        t = self.block4(t)
        t = self.block5(t)

        t = rm.relu(self.fc6(t))
        fc6 = t

        t = rm.relu(self.fc7(t))
        fc7 = t

        t = self.score_fr(t)
        score_fr = t
        t = self.upscore(t)
        return t
