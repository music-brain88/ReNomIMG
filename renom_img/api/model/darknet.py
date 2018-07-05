import os
import sys
import numpy as np
import renom as rm
from tqdm import tqdm
from renom_img.api.utility.misc.download import download
from renom_img.api.model.classification_base import ClassificationBase
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.target import DataBuilderClassification



class DarknetBase(ClassificationBase):
    def __init__(self, class_map):
        super(Darknet, self).__init__(class_map)
        self._opt = rm.Sgd(0.01, 0.9)

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
        return (x / 255.) *2 - 1

    def regularize(self, decay_rate=0.0005):
        """L2 Regularization term. You can use this function to add L2 regularization term to a loss function.

        In VGG16, weight decay of 0.0005 is used.

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
                avg_valid_loss_list.append(avg_valid_loss)
                bar.set_description("Epoch:{:03d} Avg Train Loss:{:5.3f} Avg Valid Loss:{:5.3f}".format(
                    e, avg_train_loss, avg_valid_loss))
            else:
                bar.set_description("Epoch:{:03d} Avg Train Loss:{:5.3f}".format(e, avg_train_loss))
            bar.close()
            if callback_end_epoch is not None:
                callback_end_epoch(e, self, avg_train_loss_list, avg_valid_loss_list)
        return avg_train_loss_list, avg_valid_loss_list

class Darknet(DarknetBase):
    WEIGHT_URL = "Darknet"

    def __init__(self, class_map, load_weight_path=None, imsize=(224, 224), train_whole_network=False):
        # TODO: Passing last_unit_size is not good.
        assert load_weight_path is None or isinstance(load_weight_path, str)
        n_class = len(class_map)

        super(Darknet, self).__init__([
            # 1st Block
            rm.Conv2d(channel=64, filter=7, stride=2, padding=3),
            rm.LeakyRelu(slope=0.1),
            rm.MaxPool2d(stride=2, filter=2),

            # 2nd Block
            rm.Conv2d(channel=192, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.MaxPool2d(stride=2, filter=2),

            # 3rd Block
            rm.Conv2d(channel=128, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=256, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=256, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.MaxPool2d(stride=2, filter=2),

            # 4th Block
            rm.Conv2d(channel=256, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=256, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=256, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=256, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.MaxPool2d(stride=2, filter=2),

            # 5th Block
            rm.Conv2d(channel=512, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, stride=2, padding=1),
            rm.LeakyRelu(slope=0.1),

            # 6th Block
            rm.Conv2d(channel=1024, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),

            # 7th Block
            rm.Flatten(),
            rm.Dense(1024),
            rm.LeakyRelu(slope=0.1),
            rm.Dense(4096),
            rm.LeakyRelu(slope=0.1),
            rm.Dropout(0.5),

            # 8th Block
            rm.Dense(n_class),
        ])

        if load_weight_path is not None:
            # Call download method.
            path, ext = os.path.splitext(load_weight_path)
            if ext:
                self.load(load_weight_path)
            else:
                self.load(path + '.h5')

# Darknet19


class DarknetConv2dBN(rm.Model):

    def __init__(self, channel, filter=3, prev_ch=None):
        pad = int((filter - 1) / 2)
        if prev_ch is not None:
            self._conv = rm.Conv2d(channel=channel, filter=filter, padding=pad)
            self._conv.params = {
                "w": rm.Variable(self._conv._initializer((channel, prev_ch, filter, filter)), auto_update=True),
                "b": rm.Variable(self._conv._initializer((1, channel, 1, 1)), auto_update=False),
            }
            self._bn = rm.BatchNormalize(mode='feature', momentum=0.01)
        else:
            self._conv = rm.Conv2d(channel=channel, filter=filter, padding=pad)
            self._bn = rm.BatchNormalize(mode='feature', momentum=0.01)

    def forward(self, x):
        return rm.leaky_relu(self._bn(self._conv(x)), 0.1)


class Darknet19Base(rm.Model):

    def __init__(self):
        self.block1 = rm.Sequential([
            DarknetConv2dBN(32, prev_ch=3),
            rm.MaxPool2d(filter=2, stride=2)
        ])
        self.block2 = rm.Sequential([
            DarknetConv2dBN(64, prev_ch=32),
            rm.MaxPool2d(filter=2, stride=2)
        ])
        self.block3 = rm.Sequential([
            DarknetConv2dBN(128, prev_ch=64),
            DarknetConv2dBN(64, filter=1, prev_ch=128),
            DarknetConv2dBN(128, prev_ch=64),
            rm.MaxPool2d(filter=2, stride=2)
        ])
        self.block4 = rm.Sequential([
            DarknetConv2dBN(256, prev_ch=128),
            DarknetConv2dBN(128, filter=1, prev_ch=256),
            DarknetConv2dBN(256, prev_ch=128),
            rm.MaxPool2d(filter=2, stride=2)
        ])
        self.block5 = rm.Sequential([
            DarknetConv2dBN(512, prev_ch=256),
            DarknetConv2dBN(256, filter=1, prev_ch=512),
            DarknetConv2dBN(512, prev_ch=256),
            DarknetConv2dBN(256, filter=1, prev_ch=512),
            DarknetConv2dBN(512, prev_ch=256),
        ])
        self.block6 = rm.Sequential([
            # For concatenation.
            rm.MaxPool2d(filter=2, stride=2),
            DarknetConv2dBN(1024, prev_ch=512),
            DarknetConv2dBN(512, filter=1, prev_ch=1024),
            DarknetConv2dBN(1024, prev_ch=512),
            DarknetConv2dBN(512, filter=1, prev_ch=1024),
            DarknetConv2dBN(1024, prev_ch=512),
        ])

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        f = self.block5(h)
        h = self.block6(f)
        return h, f


class Darknet19(rm.Model):

    WEIGHT_URL = "Darknet19"

    def __init__(self, num_class=1000):
        self._num_class = num_class
        self._base = Darknet19Base()
        self._last = rm.Conv2d(num_class, filter=1)
        self._last.params = {
            "w": rm.Variable(self._conv._initializer((num_class, 1024, filter, filter)), auto_update=True),
            "b": rm.Variable(self._conv._initializer((1, num_class, 1, 1)), auto_update=False),
        }

    def forward(self, x):
        N = len(x)
        h, _ = self._base(x)
        D = h.shape[2] * h.shape[3]
        h = rm.sum(self._last(h).reshape(N, self._num_class, -1), axis=2)
        h /= D
        return h
