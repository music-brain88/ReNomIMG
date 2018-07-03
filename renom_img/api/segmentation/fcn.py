import os
import sys
import numpy as np
import renom as rm
from tqdm import tqdm
from renom_img.api.utility.misc.download import download
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.load import load_img
from renom_img.api.utility.target import DataBuilderSegmentation

DIR = os.path.split(os.path.abspath(__file__))[0]


def layer_factory(channel=32, conv_layer_num=2):
    layers = []
    for _ in range(conv_layer_num):
        layers.append(rm.Conv2d(channel=channel, padding=1, filter=3))
        layers.append(rm.Relu())
    layers.append(rm.MaxPool2d(filter=2, stride=2))
    return rm.Sequential(layers)


class FCN_Base(rm.Model):
    def __init__(self, class_map, load_weight=False, imsize=(224, 224), train_whole_network=False):
        n_class = len(class_map)
        self.conv_block1 = layer_factory(channel=64, conv_layer_num=2)
        self.conv_block2 = layer_factory(channel=128, conv_layer_num=2)
        self.conv_block3 = layer_factory(channel=256, conv_layer_num=3)
        self.conv_block4 = layer_factory(channel=512, conv_layer_num=3)
        self.conv_block5 = layer_factory(channel=512, conv_layer_num=3)

        self.fc6 = rm.Conv2d(4096, filter=7, padding=3)
        self.fc7 = rm.Conv2d(4096, filter=1)
        self._train_whole_network = train_whole_network
        self.imsize = imsize
        self.class_map = class_map
        self._opt = rm.Sgd(0.0001, 0.9)

    def _freeze_network(self):
        self.conv_block1.set_auto_update(self._train_whole_network)
        self.conv_block2.set_auto_update(self._train_whole_network)
        self.conv_block3.set_auto_update(self._train_whole_network)
        self.conv_block4.set_auto_update(self._train_whole_network)
        self.conv_block5.set_auto_update(self._train_whole_network)

    def loss(self, x, y):
        return rm.softmax_cross_entropy(x, y) / (self.imsize[0] * self.imsize[1])

    def get_optimizer(self, current_epoch=None, total_epoch=None, current_batch=None, total_batch=None):
        """Returns an instance of Optimiser for training Yolov1 algorithm.

        Args:
            current_epoch:
            total_epoch:
            current_batch:
            total_epoch:

        Note: In FCN, the learning rate is fixed.
        """
        return self._opt

    def regularize(self, decay_rate=2e-4):
        """Regularize term. You can use this function to add regularize term to loss function.

        In FCN, weight decay of 2e-4 is applied.

        Example:
            >>> import numpy as np
            >>> from renom_img.api.segmentation.fcn import FCN32s
            >>> x = np.random.rand(1, 3, 224, 224)
            >>> y = np.random.rand(1, (5*2+20)*7*7)
            >>> model = FCN32s()
            >>> loss = model.loss(x, y)
            >>> reg_loss = loss + model.regularize() # Add weight decay term.

        """

        reg = 0
        for layer in self.iter_models():
            if hasattr(layer, "params") and hasattr(layer.params, "w"):
                reg += rm.sum(layer.params.w * layer.params.w)
        return decay_rate * reg

    def preprocess(self, x):
        """Image preprocess for VGG.

        Args:
            x (ndarray):

        Returns:
            (ndarray): Preprocessed data.
        """
        x[:, 0, :, :] -= 123.68  # R
        x[:, 1, :, :] -= 116.779  # G
        x[:, 2, :, :] -= 103.939  # B
        return x

    def predict(self, img_list):
        self.set_models(inference=True)
        if isinstance(img_list, (list, str)):
            if isinstance(img_list, (tuple, list)):
                img_array = np.vstack([load_img(path, self.imsize)[None] for path in img_list])
                img_array = self.preprocess(img_array)
                img_array = load_img(img_list, self.imsize)[None]
                img_array = self.preprocess(img_array)
                return np.argmax(rm.softmax(self(img_array)).as_ndarray(), axis=1)[0]
        else:
            img_array = img_list
        return np.argmax(self(img_array).as_ndarray(), axis=1)

    def fit(self, train_img_path_list=None, train_annotation_path_list=None, augmentation=None, valid_img_path_list=None, valid_annotation_path_list=None,  epoch=200, batch_size=16, callback_end_epoch=None):
        if train_img_path_list is not None and train_annotation_path_list is not None:
            train_dist = ImageDistributor(
                train_img_path_list, train_annotation_path_list, augmentation=augmentation)
        else:
            train_dist = train_image_distributor

        assert train_dist is not None

        if valid_img_path_list is not None and valid_annotation_path_list is not None:
            valid_dist = ImageDistributor(valid_img_path_list, valid_annotation_path_list)
        else:
            valid_dist = valid_image_distributor

        batch_loop = int(np.ceil(len(train_dist) / batch_size))
        avg_train_loss_list = []
        avg_valid_loss_list = []
        for e in range(epoch):
            bar = tqdm(range(batch_loop))
            display_loss = 0
            for i, (train_x, train_y) in enumerate(train_dist.batch(batch_size, target_builder=DataBuilderSegmentation(self.imsize, self.class_map))):
                self.set_models(inference=False)
                with self.train():
                    loss = self.loss(self(train_x), train_y)
                    reg_loss = loss + self.regularize()
                reg_loss.grad().update(self.get_optimizer(e, epoch, i, batch_loop))
                try:
                    loss = loss.as_ndarray()[0]
                except:
                    loss = loss.as_ndarray()
                display_loss += loss
                bar.set_description("Epoch:{:03d} Train Loss:{:5.3f}".format(e, loss))
                bar.update(1)
            avg_train_loss = display_loss / (i + 1)
            avg_train_loss_list.append(avg_train_loss)

            if valid_dist is not None:
                display_loss = 0
                for i, (valid_x, valid_y) in enumerate(valid_dist.batch(batch_size, target_builder=DataBuilderSegmentation(self.imsize, self.class_map))):
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


class FCN32s(FCN_Base):
    """ Fully convolutional network (21s) for semantic segmentation

    Args:
        n_class (int): The number of classes
        load_weight (Bool): If True, the pre-trained weight of ImageNet is loaded.

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>> from renom_img.api.segmentation.fcn import FCN32s
        >>> n, c, h, w = (2, 12, 64, 64)
        >>> x = rm.Variable(np.random.rand(n, c, h, w))
        >>> model = FCN32s(12)
        >>> t = model(x)
        >>> t.shape
        (2, 12, 64, 64)

    Note:
        Jonathan Long, Evan Shelhamer, Trevor Darrell
        Fully Convolutional Networks for Semantic Segmentation
        https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
    """

    def __init__(self, class_map, load_weight=False, imsize=(224, 224), train_whole_network=False):
        n_class = len(class_map)
        super(FCN32s, self).__init__(class_map, load_weight,
                                     imsize=imsize, train_whole_network=train_whole_network)
        self.score_fr = rm.Conv2d(n_class, filter=1)  # n_classes
        self.upscore = rm.Deconv2d(n_class, stride=32, padding=0, filter=32)  # n_classes

    def forward(self, x):
        t = x
        t = self.conv_block1(t)
        t = self.conv_block2(t)
        t = self.conv_block3(t)
        t = self.conv_block4(t)
        t = self.conv_block5(t)

        t = rm.relu(self.fc6(t))
        fc6 = t

        t = rm.relu(self.fc7(t))
        fc7 = t

        t = self.score_fr(t)
        score_fr = t
        t = self.upscore(t)
        return t


class FCN16s(FCN_Base):
    """ Fully convolutional network (16s) for semantic segmentation
    Reference: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf

    Args:
        n_class (int): The number of classes
        load_weight (Bool): If True, the pre-trained weight of ImageNet is loaded.

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>> from renom_img.api.segmentation.fcn import FCN16s
        >>> n, c, h, w = (2, 12, 64, 64)
        >>> x = rm.Variable(np.random.rand(n, c, h, w))
        >>> model = FCN16s(12)
        >>> t = model(x)
        >>> t.shape
        (2, 12, 64, 64)

    Note:
        Jonathan Long, Evan Shelhamer, Trevor Darrell
        Fully Convolutional Networks for Semantic Segmentation
        https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
    """

    def __init__(self, class_map, load_weight=False, imsize=(224, 224), train_whole_network=False):
        n_class = len(class_map)
        super(FCN16s, self).__init__(class_map, load_weight,
                                     imsize=imsize, train_whole_network=train_whole_network)
        self.score_fr = rm.Conv2d(n_class, filter=1)
        self.score_pool4 = rm.Conv2d(n_class, filter=1)

        self.upscore2 = rm.Deconv2d(n_class, filter=2, stride=2, padding=0)
        self.upscore16 = rm.Deconv2d(n_class, filter=16, stride=16, padding=0)

    def forward(self, x):
        t = x
        t = self.conv_block1(t)
        t = self.conv_block2(t)
        t = self.conv_block3(t)
        t = self.conv_block4(t)
        pool4 = t
        t = self.conv_block5(t)

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


class FCN8s(FCN_Base):
    """ Fully convolutional network (8s) for semantic segmentation
    Reference: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf

    Args:
        n_class (int): The number of classes
        load_weight (Bool): If True, the pre-trained weight of ImageNet is loaded.

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>> from renom_img.api.segmentation.fcn import FCN8s
        >>> n, c, h, w = (2, 12, 64, 64)
        >>> x = rm.Variable(np.random.rand(n, c, h, w))
        >>> model = FCN8s(12)
        >>> t = model(x)
        >>> t.shape
        (2, 12, 64, 64)

    Note:
        Jonathan Long, Evan Shelhamer, Trevor Darrell
        Fully Convolutional Networks for Semantic Segmentation
        https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
    """

    def __init__(self, class_map, load_weight=False, imsize=(224, 224), train_whole_network=False):
        n_class = len(class_map)
        super(FCN8s, self).__init__(class_map, load_weight,
                                    imsize=imsize, train_whole_network=train_whole_network)

        self.drop_out = rm.Dropout(0.5)

        self.score_fr = rm.Conv2d(n_class, filter=1)
        self.upscore2 = rm.Deconv2d(n_class, filter=2, stride=2, padding=0)
        self.upscore8 = rm.Deconv2d(n_class, filter=8, stride=8, padding=0)

        self.score_pool3 = rm.Conv2d(n_class, filter=1)
        self.score_pool4 = rm.Conv2d(n_class, filter=1)

        self.upscore_pool4 = rm.Deconv2d(n_class, filter=2, stride=2, padding=0)

    def forward(self, x):
        t = x
        t = self.conv_block1(t)
        t = self.conv_block2(t)
        t = self.conv_block3(t)
        pool3 = t
        t = self.conv_block4(t)
        pool4 = t
        t = self.conv_block5(t)

        t = rm.relu(self.fc6(t))
        t = self.drop_out(t)
        fc6 = t

        t = rm.relu(self.fc7(t))
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
