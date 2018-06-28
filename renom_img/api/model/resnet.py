import os
import sys
import renom as rm
import numpy as np
from tqdm import tqdm

DIR = os.path.split(os.path.abspath(__file__))[0]
from renom_img.api.utility.misc.download import download
from renom_img.api.model.classification_base import ClassificationBase
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.target import DataBuilderClassification


def layer_block(channel, filter):
    layers = []
    if filter != (1, 1):
        layers.append(rm.Conv2d(filter=filter, channel=channel, padding=1))
    else:
        layers.append(rm.Conv2d(filter=filter, channel=channel))
    layers.append(rm.BatchNormalize(epsilon=0.001, mode='feature'))
    layers.append(rm.Relu())
    return layers


def downsample_block(channel, filter):
    return [
        rm.Conv2d(filter=filter, channel=channel, padding=1, stride=2),
        rm.BatchNormalize(epsilon=0.001, mode='feature'),
        rm.Relu(),
    ]


def build_block(channels):
    """
    A block without down-sampling (stride == 1)
    """
    layers = []
    if type(channels) == int:
        layers.extend(layer_block(channels, (3, 3)))
        layers.extend(layer_block(channels, (3, 3)))
    else:
        layers.extend(layer_block(channels[0], (1, 1)))
        layers.extend(layer_block(channels[1], (3, 3)))
        layers.extend(layer_block(channels[2], (1, 1)))
    return rm.Sequential(layers)


def build_downsample_block(channels):
    """
    A block including down-sample process
    """
    layers = []
    if type(channels) == int:
        layers.extend(downsample_block(channels, (3, 3)))
        layers.extend(layer_block(channels, (3, 3)))
    else:
        layers.extend(downsample_block(channels[0], (1, 1)))
        layers.extend(layer_block(channels[1], (3, 3)))
        layers.extend(layer_block(channels[2], (1, 1)))
    return rm.Sequential(layers)

class ResNetBase(ClassificationBase):
    def __init__(self, class_map):
        super(ResNetBase, self).__init__(class_map)
        self._opt = rm.Sgd(0.1, 0.9)

    def get_optimizer(self, current_epoch=None, total_epoch=None, current_batch=None, total_batch=None):
        """Returns an instance of Optimiser for training Yolov1 algorithm.

        Args:
            current_epoch:
            total_epoch:
            current_batch:
            total_epoch:
        """
        if any([num is None for num in [current_epoch, total_epoch, current_batch, total_batch]]):
            return self._opt
        else:
            self._opt._lr = lr / 10.
            return self._opt


    def preprocess(self, x):
        """Image preprocess for VGG.

        Args:
            x (ndarray):

        Returns:
            (ndarray): Preprocessed data.
        """
        return x / 255.

    def regularize(self, decay_rate=0.0001):
        """L2 Regularization term. You can use this function to add L2 regularization term to a loss function.

        In ResNet, weight decay of 0.0001 is used.

        Example:
            >>> import numpy as np
            >>> from renom_img.api.model.vgg import VGG16
            >>> x = np.random.rand(1, 3, 224, 224)
            >>> y = np.random.rand(1, (5*2+20)*7*7)
            >>> model = VGG16()
            >>> loss = model.loss(x, y)
            >>> reg_loss = loss + model.regularize() # Add weight decay term.

        """
        return super().regularize(decay_rate)

    def fit(self, train_img_path_list=None, train_annotation_list=None, augmentation=None, valid_img_path_list=None, valid_annotation_list=None,  epoch=200, batch_size=16, callback_end_epoch=None):
        if train_img_path_list is not None and train_annotation_list is not None:
            train_dist = ImageDistributor(train_img_path_list, train_annotation_list, augmentation=augmentation)
        else:
            train_dist = train_image_distributor

        assert train_dist is not None

        if valid_img_path_list is not None and valid_annotation_list is not None:
            valid_dist = ImageDistributor(valid_img_path_list, valid_annotation_list)
        else:
            valid_dist = valid_image_distributor

        opt_flag = False
        batch_loop = int(np.ceil(len(train_dist) / batch_size))
        avg_train_loss_list = []
        avg_valid_loss_list = []
        for e in range(epoch):
            bar = tqdm(range(batch_loop))
            display_loss = 0
            for i, (train_x, train_y) in enumerate(train_dist.batch(batch_size, target_builder=DataBuilderClassification(self.imsize, self.class_map))):
                self.set_models(inference=False)
                with self.train():
                    loss = self.loss(self(train_x), train_y)
                    reg_loss = loss + self.regularize()

                if opt_flag:
                    reg_loss.grad().update(self.get_optimizer(e, epoch, i, batch_loop))
                    opt_flag = False
                else:
                    reg_loss.grad().update(self.opt)
                try:
                    loss = loss.as_ndarray()[0]
                except:
                    loss = loss.as_ndarray()
                display_loss += loss
                bar.set_description("Epoch:{:03d} Train Loss:{:5.3f}".format(e, loss))
                bar.update(1)
            avg_train_loss = display_loss / (i + 1)
            avg_train_loss_list.append(avg_train_loss)
            if avg_train_loss[-1] > avg_train_loss[-2]:
                opt_flag=True

            if valid_dist is not None:
                display_loss = 0
                for i, (valid_x, valid_y) in enumerate(valid_dist.batch(batch_size, target_builder=DataBuilderClassification(self.imsize, self.class_map))):
                    self.set_models(inference=True)
                    loss = self.loss(self(train_x), train_y)
                    try:
                        loss = loss.as_ndarray()[0]
                    except:
                        loss = loss.as_ndarray()
                    display_loss += loss
                    bar.set_description("Epoch:{:03d} Valid Loss:{:5.3f}".format(e, loss))
                    bar.update(1)
                avg_valid_loss = display_loss / (i + 1)
                avg_valid_loss_list.append(avg_train_loss)
                bar.set_description("Epoch:{:03d} Avg Train Loss:{:5.3f} Avg Valid Loss:{:5.3f}".format(
                    e, avg_train_loss, avg_valid_loss))
            else:
                bar.set_description("Epoch:{:03d} Avg Train Loss:{:5.3f}".format(e, avg_train_loss))
            bar.close()
            if callback_end_epoch is not None:
                callback_end_epoch(e, self, avg_train_loss_list, avg_valid_loss_list)
        return avg_train_loss_list, avg_valid_loss_list


class ResNet(ResNetBase):
    def __init__(self, class_map, channels, num_layers, imsize=(224, 224), train_whole_network=False):
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        if type(num_layers) == int:
            num_layers = [num_layers] * len(channels)

        n_class = len(class_map)
        self.num_layers = num_layers
        self.n_class = len(class_map)
        self.channels = channels
        layers = []
        layers.append(rm.Conv2d(channel=16, padding=1))
        layers.append(rm.BatchNormalize(epsilon=0.001, mode='feature'))

        # First block which doesn't have down-sampling
        for _ in range(num_layers[0]):
            layers.append(build_block(channels[0]))

        # The rest of blocks which has down-sampling layer
        for i, num in enumerate(num_layers[1:]):
            for j in range(num):
                if j == 0:
                    layers.append(build_downsample_block(channels[i + 1]))
                else:
                    layers.append(build_block(channels[i + 1]))

        # Add the last dense layer
        self._freezed_network = rm.Sequential(layers)
        self._network = rm.Dense(self.n_class)
        self._train_whole_network = train_whole_network
        self.imsize = imsize
        super(ResNet, self).__init__(class_map)

    @property
    def freezed_network(self):
        return self._freezed_network

    @property
    def network(self):
        return self._network

    def forward(self, x):
        self.freezed_network.set_auto_update(self._train_whole_network)
        index = 0
        t = self.freezed_network[index](x)
        index += 1
        t = rm.relu(self.freezed_network[index](t))  # Batch normalization
        index += 1

        # First block
        for _ in range(self.num_layers[0]):
            tmp = t
            t = self.freezed_network[index](t)
            index += 1
            t = rm.concat([t, tmp])

        # the rest of block
        for num in self.num_layers[1:]:
            for i in range(num):
                if i == 0:
                    t = self.freezed_network[index](t)
                    index += 1
                else:
                    tmp = t
                    t = self.freezed_network[index](t)
                    index += 1
                    t = rm.concat([t, tmp])
        t = rm.flatten(rm.average_pool2d(t))
        t = self.network(t)
        return t


class ResNet32(ResNet):
    """ResNet32 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        load_weight(bool):
        class_map: Array of class names
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        6n + 2(The first conv and the last dense) = 32
        → n = 5
        5 sets of a layer block in each block

        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    """

    WEIGHT_URL = "https://app.box.com/shared/static/o81vwdp4qsm88zt93jvpskqfzobhfx6s.h5"
    WEIGHT_PATH = os.path.join(DIR, 'resnet32.h5')

    def __init__(self, class_map, load_weight=False, imsize=(224, 224), train_whole_network=False):
        num_layers = 5
        CHANNELS = [16, 32, 64]
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        super(ResNet32, self).__init__(class_map, CHANNELS, num_layers, imsize=imsize, train_whole_network=train_whole_network)
        n_class = len(class_map)
        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if n_class != 1000:
            self.network.params = {}


class ResNet44(ResNet):
    """ResNet44 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        load_weight(bool):
        class_map: Array of class names
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    """

    WEIGHT_URL = "https://app.box.com/shared/static/o81vwdp4qsm88zt93jvpskqfzobhfx6s.h5"
    WEIGHT_PATH = os.path.join(DIR, 'resnet44.h5')

    def __init__(self, class_map, load_weight=False, imsize=(224, 224), train_whole_network=False):
        num_layers = 7
        CHANNELS = [16, 32, 64]
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        super(ResNet44, self).__init__(class_map, CHANNELS, num_layers, imsize=imsize, train_whole_network=train_whole_network)
        n_class = len(class_map)

        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if n_class != 1000:
            self.freezed_network.params = {}


class ResNet56(ResNet):
    """ResNet56 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        load_weight(bool):
        class_map: Array of class names
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    """

    WEIGHT_URL = "https://app.box.com/shared/static/o81vwdp4qsm88zt93jvpskqfzobhfx6s.h5"
    WEIGHT_PATH = os.path.join(DIR, 'resnet56.h5')

    def __init__(self, class_map, load_weight=False, imsize=(224, 224), train_whole_network=False):
        num_layers = 9
        CHANNELS = [16, 32, 64]
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        super(ResNet56, self).__init__(class_map, CHANNELS, num_layers, imsize=imsize, train_whole_network=train_whole_network)
        n_class = len(class_map)

        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if n_class != 1000:
            self.network.params = {}


class ResNet110(ResNet):
    """ResNet110 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        load_weight(bool):
        class_map: Array of class names
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    """

    WEIGHT_URL = "https://app.box.com/shared/static/o81vwdp4qsm88zt93jvpskqfzobhfx6s.h5"
    WEIGHT_PATH = os.path.join(DIR, 'resnet110.h5')

    def __init__(self, class_map, load_weight=False, imsize=(224, 224), train_whole_network=False):
        num_layers = 18
        CHANNELS = [16, 32, 64]
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        super(ResNet110, self).__init__(class_map, CHANNELS, num_layers, imsize=imsize, train_whole_network=train_whole_network)
        n_class = len(class_map)

        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if n_class != 1000:
            self.network.params = {}


class ResNet34(ResNet):
    """ResNet34 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        load_weight(bool):
        class_map: Array of class names
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    """

    WEIGHT_URL = "https://app.box.com/shared/static/o81vwdp4qsm88zt93jvpskqfzobhfx6s.h5"
    WEIGHT_PATH = os.path.join(DIR, 'resnet34.h5')

    def __init__(self, class_map, load_weight=False, imsize=(224, 224), train_whole_network=False):
        num_layers = [3, 4, 6, 3]
        CHANNELS = [64, 128, 256, 512]
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        super(ResNet34, self).__init__(class_map, CHANNELS, num_layers, imsize=imsize, train_whole_network=train_whole_network)
        n_class = len(class_map)
        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if n_class != 1000:
            self.network.params = {}


class ResNet50(ResNet):
    """ResNet50 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        load_weight(bool):
        class_map: Array of class names
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    """

    WEIGHT_URL = "https://app.box.com/shared/static/o81vwdp4qsm88zt93jvpskqfzobhfx6s.h5"
    WEIGHT_PATH = os.path.join(DIR, 'resnet50.h5')

    def __init__(self, class_map, load_weight=False, imsize=(224, 224), train_whole_network=False):
        num_layers = [3, 4, 6, 3]
        CHANNELS = [64, 128, 256, 512]
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)

        super(ResNet50, self).__init__(class_map, CHANNELS, num_layers, imsize=imsize, train_whole_network=train_whole_network)
        n_class = len(class_map)
        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if n_class != 1000:
            self.network.params = {}


class ResNet101(ResNet):
    """ResNet101 model.

    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        load_weight(bool):
        class_map: Array of class names
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385
    """

    WEIGHT_URL = "https://app.box.com/shared/static/o81vwdp4qsm88zt93jvpskqfzobhfx6s.h5"
    WEIGHT_PATH = os.path.join(DIR, 'resnet101.h5')

    def __init__(self, class_map, load_weight=False, imsize=(224, 224), train_whole_network=False):
        num_layers = [3, 4, 23, 3]
        CHANNELS = [[64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        super(ResNet101, self).__init__(class_map, CHANNELS, num_layers, imsize=imsize, train_whole_network=train_whole_network)
        n_class = len(class_map)
        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if n_class != 1000:
            self.network.params = {}
