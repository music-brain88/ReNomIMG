from __future__ import print_function, division
import os
import numpy as np
import renom as rm
from tqdm import tqdm

from renom_img import __version__
from renom_img.api import Base, adddoc
from renom_img.api.classification import Classification
from renom_img.api.utility.misc.download import download
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.utility.distributor.distributor import ImageDistributor

DIR = os.path.split(os.path.abspath(__file__))[0]


def layer_factory(channel=32, conv_layer_num=2):
    layers = []
    for _ in range(conv_layer_num):
        layers.append(rm.Conv2d(channel=channel, padding=1, filter=3))
        layers.append(rm.Relu())
    layers.append(rm.MaxPool2d(filter=2, stride=2))
    return rm.Sequential(layers)


@adddoc
class VGGBase(Classification):

    SERIALIZED = Base.SERIALIZED

    def set_last_layer_unit(self, unit_size):
        self._model.fc3._output_size = unit_size

    def get_optimizer(self, current_loss=None, current_epoch=None,
                      total_epoch=None, current_batch=None, total_batch=None, avg_valid_loss_list=None):
        if any([num is None for num in [current_loss, current_epoch, total_epoch, current_batch, total_batch]]):
            return self._opt
        else:
            if current_epoch == 0:
                self._opt._lr = 0.00001 + (0.001 - 0.00001) * current_batch / total_batch
            return self._opt

    def preprocess(self, x):
        """
        Returns:
            (ndarray): Preprocessed data.

        Preprocessing for VGG is follows.

        .. math::

            x_{red} -= 123.68 \\\\
            x_{green} -= 116.779 \\\\
            x_{blue} -= 103.939

        """
        x[:, 0, :, :] -= 123.68  # R
        x[:, 1, :, :] -= 116.779  # G
        x[:, 2, :, :] -= 103.939  # B
        return x

    def _freeze(self):
        self._model.block1.set_auto_update(self.train_whole_network)
        self._model.block2.set_auto_update(self.train_whole_network)
        self._model.block3.set_auto_update(self.train_whole_network)
        self._model.block4.set_auto_update(self.train_whole_network)
        self._model.block5.set_auto_update(self.train_whole_network)

    def forward(self, x):
        """
        Returns:
            (Node): Returns raw output of ${class}.

        Example:
            >>> import numpy as np
            >>> from renom_img.api.classification.vgg import ${class}
            >>>
            >>> x = np.random.rand(1, 3, 224, 224)
            >>> class_map = ["dog", "cat"]
            >>> model = ${class}(class_map)
            >>> y = model.forward(x) # Forward propagation.
            >>> y = model(x)  # Same as above result.
        """
        self._freeze()
        return self._model(x)


@adddoc
class VGG11(VGGBase):
    """VGG11 model.

    Args:
        class_map(array): Array of class names
        imsize(int or tuple): Input image size
        train_whole_network(bool): True if the overall model is trained, otherwise False
        load_pretrained_weight (bool, str): If true, pretrained weight will be
          downloaded to current directory. If string is given, pretrained weight
          will be saved as given name.

    Note:
        if the argument num_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    References:
        | Karen Simonyan, Andrew Zisserman
        | Very Deep Convolutional Networks for Large-Scale Image Recognition
        | https://arxiv.org/abs/1409.1556
        |

    """

    WEIGHT_URL = None

    def __init__(self, class_map=None, imsize=(224, 224),
                 load_pretrained_weight=False, train_whole_network=False):

        self._model = CNN_VGG11()
        super(VGG11, self).__init__(class_map, imsize, load_pretrained_weight,
                                    train_whole_network, self._model)

        self._opt = rm.Sgd(0.01, 0.9)
        self.decay_rate = 0.0005

        self._model.fc1.params = {}
        self._model.fc2.params = {}
        self._model.fc3.params = {}


@adddoc
class VGG16(VGGBase):
    """VGG16 model.
    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        class_map(array): Array of class names
        imsize(int or tuple): Input image size
        train_whole_network(bool): True if the overall model is trained, otherwise False
        load_pretrained_weight (bool, str): If true, pretrained weight will be
          downloaded to current directory. If string is given, pretrained weight
          will be saved as given name.


    Note:
        if the argument num_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    References:
        | Karen Simonyan, Andrew Zisserman
        | Very Deep Convolutional Networks for Large-Scale Image Recognition
        | https://arxiv.org/abs/1409.1556
        |

    """

    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/classification/VGG16.h5".format(
        __version__)

    def __init__(self, class_map=None, imsize=(224, 224),
                 load_pretrained_weight=False, train_whole_network=False):

        self._model = CNN_VGG16()
        super(VGG16, self).__init__(class_map, imsize, load_pretrained_weight,
                                    train_whole_network, self._model)

        self._opt = rm.Sgd(0.01, 0.9)
        self.decay_rate = 0.0005

        self._model.fc1.params = {}
        self._model.fc2.params = {}
        self._model.fc3.params = {}


class VGG16_NODENSE(VGGBase):

    def __init__(self, class_map=None, imsize=(224, 224),
                 load_pretrained_weight=False, train_whole_network=False):

        self._model = CNN_VGG16_NODENSE()
        super(VGG16, self).__init__(class_map, imsize, load_pretrained_weight,
                                    train_whole_network, self._model, load_target=self._model)

        self.train_whole_network = train_whole_network
        self._opt = rm.Sgd(0.001, 0.9)
        self.decay_rate = 0.0005

        self._model.fc1.params = {}
        self._model.fc2.params = {}
        self._model.fc3.params = {}


@adddoc
class VGG19(VGGBase):
    """VGG19 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        class_map(array): Array of class names
        imsize(int or tuple): Input image size
        train_whole_network(bool): True if the overall model is trained, otherwise False
        load_pretrained_weight (bool, str): If true, pretrained weight will be
          downloaded to current directory. If string is given, pretrained weight
          will be saved as given name.

    Note:
        if the argument num_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    References:
        | Karen Simonyan, Andrew Zisserman
        | Very Deep Convolutional Networks for Large-Scale Image Recognition
        | https://arxiv.org/abs/1409.1556
        |

    """

    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/classification/VGG19.h5".format(
        __version__)

    def __init__(self, class_map=None, imsize=(224, 224),
                 load_pretrained_weight=False, train_whole_network=False):

        self._model = CNN_VGG19()
        super(VGG19, self).__init__(class_map, imsize,
                                    load_pretrained_weight, train_whole_network, self._model)

        self._opt = rm.Sgd(0.01, 0.9)
        self.decay_rate = 0.0005

        self._model.fc1.params = {}
        self._model.fc2.params = {}
        self._model.fc3.params = {}


class CNN_VGG19(rm.Model):

    def __init__(self, num_class=1000):
        self.num_class = num_class
        self.block1 = layer_factory(channel=64, conv_layer_num=2)
        self.block2 = layer_factory(channel=128, conv_layer_num=2)
        self.block3 = layer_factory(channel=256, conv_layer_num=4)
        self.block4 = layer_factory(channel=512, conv_layer_num=4)
        self.block5 = layer_factory(channel=512, conv_layer_num=4)
        self.fc1 = rm.Dense(4096)
        self.dropout1 = rm.Dropout(dropout_ratio=0.5)
        self.fc2 = rm.Dense(4096)
        self.dropout2 = rm.Dropout(dropout_ratio=0.5)
        self.fc3 = rm.Dense(num_class)

    def forward(self, x):
        assert self.num_class > 0, \
            "Class map is empty. Please set the attribute class_map when instantiating a model. " +\
            "Or, please load a pre-trained model using the ‘load()’ method."
        t = self.block1(x)
        t = self.block2(t)
        t = self.block3(t)
        t = self.block4(t)
        t = self.block5(t)
        t = rm.flatten(t)
        t = rm.relu(self.fc1(t))
        t = self.dropout1(t)
        t = rm.relu(self.fc2(t))
        t = self.dropout2(t)
        t = self.fc3(t)
        return t


class CNN_VGG16(rm.Model):

    def __init__(self, num_class=1000):
        self.num_class = num_class
        self.block1 = layer_factory(channel=64, conv_layer_num=2)
        self.block2 = layer_factory(channel=128, conv_layer_num=2)
        self.block3 = layer_factory(channel=256, conv_layer_num=3)
        self.block4 = layer_factory(channel=512, conv_layer_num=3)
        self.block5 = layer_factory(channel=512, conv_layer_num=3)
        self.fc1 = rm.Dense(4096)
        self.dropout1 = rm.Dropout(dropout_ratio=0.5)
        self.fc2 = rm.Dense(4096)
        self.dropout2 = rm.Dropout(dropout_ratio=0.5)
        self.fc3 = rm.Dense(num_class)

    def forward(self, x):
        assert self.num_class > 0, \
            "Class map is empty. Please set the attribute class_map when instantiating a model. " +\
            "Or, please load a pre-trained model using the ‘load()’ method."
        t = self.block1(x)
        t = self.block2(t)
        t = self.block3(t)
        t = self.block4(t)
        t = self.block5(t)
        t = rm.flatten(t)
        t = rm.relu(self.fc1(t))
        t = self.dropout1(t)
        t = rm.relu(self.fc2(t))
        t = self.dropout2(t)
        t = self.fc3(t)
        return t


class CNN_VGG16_NODENSE(rm.Model):

    def __init__(self, num_class=1000):
        self.conv1_1 = rm.Conv2d(64, padding=1, filter=3)
        self.conv1_2 = rm.Conv2d(64, padding=1, filter=3)
        self.conv2_1 = rm.Conv2d(128, padding=1, filter=3)
        self.conv2_2 = rm.Conv2d(128, padding=1, filter=3)
        self.conv3_1 = rm.Conv2d(256, padding=1, filter=3)
        self.conv3_2 = rm.Conv2d(256, padding=1, filter=3)
        self.conv3_3 = rm.Conv2d(256, padding=1, filter=3)
        self.conv4_1 = rm.Conv2d(512, padding=1, filter=3)
        self.conv4_2 = rm.Conv2d(512, padding=1, filter=3)
        self.conv4_3 = rm.Conv2d(512, padding=1, filter=3)
        self.conv5_1 = rm.Conv2d(512, padding=1, filter=3)
        self.conv5_2 = rm.Conv2d(512, padding=1, filter=3)
        self.conv5_3 = rm.Conv2d(512, padding=1, filter=3)

    def forward(self, x):
        t = rm.relu(self.conv1_1(x))
        t = rm.relu(self.conv1_2(t))
        t = rm.max_pool2d(t, filter=2, stride=2)

        t = rm.relu(self.conv2_1(t))
        t = rm.relu(self.conv2_2(t))
        t = rm.max_pool2d(t, filter=2, stride=2)

        t = rm.relu(self.conv3_1(t))
        t = rm.relu(self.conv3_2(t))
        t = rm.relu(self.conv3_3(t))
        t = rm.max_pool2d(t, filter=2, stride=2)

        t = rm.relu(self.conv4_1(t))
        t = rm.relu(self.conv4_2(t))
        t = rm.relu(self.conv4_3(t))
        t = rm.max_pool2d(t, filter=2, stride=2)

        t = rm.relu(self.conv5_1(t))
        t = rm.relu(self.conv5_2(t))
        t = rm.relu(self.conv5_3(t))
        t = rm.max_pool2d(t, filter=2, stride=2)

        return t


class CNN_VGG11(rm.Model):

    def __init__(self, num_class=1000):
        self.num_class = num_class
        self.block1 = layer_factory(channel=64, conv_layer_num=1)
        self.block2 = layer_factory(channel=128, conv_layer_num=1)
        self.block3 = layer_factory(channel=256, conv_layer_num=2)
        self.block4 = layer_factory(channel=512, conv_layer_num=2)
        self.block5 = layer_factory(channel=512, conv_layer_num=2)
        self.fc1 = rm.Dense(4096)
        self.dropout1 = rm.Dropout(dropout_ratio=0.5)
        self.fc2 = rm.Dense(4096)
        self.dropout2 = rm.Dropout(dropout_ratio=0.5)
        self.fc3 = rm.Dense(num_class)

    def forward(self, x):
        assert self.num_class > 0, \
            "Class map is empty. Please set the attribute class_map when instantiating a model. " +\
            "Or, please load a pre-trained model using the ‘load()’ method."
        t = self.block1(x)
        t = self.block2(t)
        t = self.block3(t)
        t = self.block4(t)
        t = self.block5(t)
        t = rm.flatten(t)
        t = rm.relu(self.fc1(t))
        t = self.dropout1(t)
        t = rm.relu(self.fc2(t))
        t = self.dropout2(t)
        t = self.fc3(t)
        return t
