import os
import sys
import renom as rm
import numpy as np
from tqdm import tqdm
from renom_img.api.utility.misc.download import download
from renom_img.api.model.classification_base import ClassificationBase
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.target import DataBuilderClassification

DIR = os.path.split(os.path.abspath(__file__))[0]

class InceptionBase(ClassificationBase):
    def __init__(self, class_map):
        super(InceptionBase, self).__init__(class_map)

    def get_optimizer(self, current_epoch=None, total_epoch=None, current_batch=None, total_batch=None):
        pass


    def preprocess(self, x):
        """Image preprocess for VGG.

        Args:
            x (ndarray):

        Returns:
            (ndarray): Preprocessed data.
        """
        return x / 255.

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

    def fit(self, train_img_path_list=None, train_annotation_list=None, augmentation=None, valid_img_path_list=None, valid_annotation_list=None,  epoch=90, batch_size=16, callback_end_epoch=None):
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
                avg_valid_loss_list.append(avg_train_loss)
                bar.set_description("Epoch:{:03d} Avg Train Loss:{:5.3f} Avg Valid Loss:{:5.3f}".format(
                    e, avg_train_loss, avg_valid_loss))
            else:
                bar.set_description("Epoch:{:03d} Avg Train Loss:{:5.3f}".format(e, avg_train_loss))
            bar.close()
            if callback_end_epoch is not None:
                callback_end_epoch(e, self, avg_train_loss_list, avg_valid_loss_list)
        return avg_train_loss_list, avg_valid_loss_list

class InceptionV1Block(rm.Model):
    def __init__(self, channels=[64, 96, 128, 16, 32]):
        self.conv1 = rm.Conv2d(channels[0], filter=1)
        self.conv2_reduced = rm.Conv2d(channels[1], filter=1)
        self.conv2 = rm.Conv2d(channels[2], filter=3, padding=1)
        self.conv3_reduced = rm.Conv2d(channels[1], filter=1)
        self.conv3 = rm.Conv2d(channels[2], filter=5, padding=2)
        self.conv4 = rm.Conv2d(channels[1], filter=1)

    def forward(self, x):
        t1 = rm.relu(self.conv1(x))
        t2 = rm.relu(self.conv2_reduced(x))
        t2 = rm.relu(self.conv2(t2))
        t3 = rm.relu(self.conv3_reduced(x))
        t3 = rm.relu(self.conv3(t3))
        t4 = rm.max_pool2d(x, filter=3, stride=1, padding=1)
        t4 = rm.relu(self.conv4(t4))

        return rm.concat([t1, t2, t3, t4])


class InceptionV1(InceptionBase):
    """ Inception V1 model
    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        class_map: Array of class names
        load_weight(bool): True if the pre-trained weight is loaded.
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Christian Szegedy, Wei Liu, Yangqing Jia , Pierre Sermanet, Scott Reed ,Dragomir Anguelov,
    Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich
    Going Deeper with Convolutions
    https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf
    """

    WEIGHT_URL = "https://app.box.com/shared/static/eovmxxgzyh5vg2kpcukjj8ypnxng4j5v.h5"
    WEIGHT_PATH = os.path.join(DIR, 'inceptionv4.h5')

    def __init__(self, class_map, load_weight=False, imsize=(224, 224), opt=rm.Sgd(0.045, 0.9), train_whole_network=False):
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        n_class = len(class_map)
        self.imsize = imsize
        self._opt = opt
        self._train_whole_network = train_whole_network
        self.base1 = rm.Sequential([rm.Conv2d(64, filter=7, padding=3, stride=2),
                              rm.Relu(),
                              rm.MaxPool2d(filter=3, stride=2, padding=1),
                              rm.BatchNormalize(mode='feature'),
                              rm.Conv2d(64, filter=1, stride=1),
                              rm.Relu(),
                              rm.Conv2d(192, filter=3, padding=1, stride=1),
                              rm.Relu(),
                              rm.BatchNormalize(mode='feature'),
                              rm.MaxPool2d(filter=3, stride=2, padding=1),
                              InceptionV1Block(),
                              InceptionV1Block([128, 128, 192, 32, 96, 64]),
                              rm.MaxPool2d(filter=3, stride=2),
                              InceptionV1Block([192, 96, 208, 16, 48, 64]),
                              ])

        self.aux1 = rm.Sequential([rm.AveragePool2d(filter=5, stride=3),
                              rm.Flatten(),
                              rm.Dense(1024),
                              rm.Dense(n_class)])

        self.base2 = rm.Sequential([InceptionV1Block([160, 112, 224, 24, 64, 64]),
                                    InceptionV1Block([128, 128, 256, 24, 64, 64]),
                                    InceptionV1Block([112, 144, 288, 32, 64, 64])])

        self.aux2 = rm.Sequential([rm.AveragePool2d(filter=5, stride=3),
                                rm.Flatten(),
                                rm.Dense(1024),
                                rm.Dense(n_class)])

        self.base3 = rm.Sequential([InceptionV1Block([256, 160, 320, 32, 128, 128]),
                        InceptionV1Block([256, 160, 320, 32, 128, 128]),
                        InceptionV1Block([192, 384, 320, 48, 128, 128]),
                        rm.AveragePool2d(filter=7, stride=1),
                        rm.Flatten()])
        self.aux3 = rm.Dense(n_class)
        super(InceptionV1, self).__init__(class_map)

        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if n_class != 1000:
            self.aux1.params = {}
            self.aux2.params = {}
            self.aux3.params = {}

    def forward(self, x):
        self.base1.set_auto_update(self._train_whole_network)
        self.base2.set_auto_update(self._train_whole_network)
        self.base3.set_auto_update(self._train_whole_network)

        t = self.base1(x)
        out1 = self.aux1(t)
        t = self.base2(t)
        out2 = self.aux2(t)
        t = self.base3(t)
        out3 = self.aux3(t)
        return out1, out2, out3

    def loss(self, x, y):
        return 0.3 * rm.softmax_cross_entropy(x[0], y) + 0.3*rm.softmax_cross_entropy(x[1], y) + rm.softmax_cross_entropy(x[2], y)

    def predict(self, img_list):
        self.set_models(inference=True)
        if isinstance(img_list, (list, str)):
            if isinstance(img_list, (tuple, list)):
                img_array = np.vstack([load_img(path, self.imsize)[None] for path in img_list])
                img_array = self.preprocess(img_array)
            else:
                img_array = load_img(img_list, self.imsize)[None]
                img_array = self.preprocess(img_array)
                return np.argmax(rm.softmax(self(img_array)[2]).as_ndarray(), axis=1)[0]
        else:
            img_array = img_list
        return np.argmax(rm.softmax(self(img_array)[2]).as_ndarray(), axis=1)

class InceptionV2BlockA(rm.Model):
    def __init__(self, channels=[64, 48, 64, 64, 96, 64]):
        self.conv1 = rm.Conv2d(channels[0], filter=1)
        self.batch_norm1 = rm.BatchNormalize(mode='feature')

        self.conv2_reduced = rm.Conv2d(channels[1], filter=1)
        self.batch_norm2_reduced = rm.BatchNormalize(mode='feature')
        self.conv2 = rm.Conv2d(channels[2], filter=3, padding=1)
        self.batch_norm2 = rm.BatchNormalize(mode='feature')

        self.conv3_reduced = rm.Conv2d(channels[3], filter=1)
        self.batch_norm3_reduced = rm.BatchNormalize(mode='feature')
        self.conv3_1 = rm.Conv2d(channels[4], filter=3, padding=1)
        self.batch_norm3_1 = rm.BatchNormalize(mode='feature')
        self.conv3_2 = rm.Conv2d(channels[4], filter=3, padding=1)
        self.batch_norm3_2 = rm.BatchNormalize(mode='feature')

        self.conv4 = rm.Conv2d(channels[5], filter=1)
        self.batch_norm4 = rm.BatchNormalize(mode='feature')

    def forward(self, x):
        t1 = rm.relu(self.batch_norm1(self.conv1(x)))

        t2 = rm.relu(self.batch_norm2_reduced(self.conv2_reduced(x)))
        t2 = rm.relu(self.batch_norm2(self.conv2(t2)))
        t3 = rm.relu(self.batch_norm3_reduced(self.conv3_reduced(x)))
        t3 = rm.relu(self.batch_norm3_1(self.conv3_1(t3)))
        t3 = rm.relu(self.batch_norm3_2(self.conv3_2(t3)))

        t4 = rm.max_pool2d(x, filter=3, stride=1, padding=1)
        t4 = rm.relu(self.batch_norm4(self.conv4(t4)))

        return rm.concat([
            t1, t2, t3, t4
        ])


class InceptionV2BlockB(rm.Model):
    def __init__(self, channels=[64, 96, 384]):
        self.conv1_reduced = rm.Conv2d(channels[0], filter=1)
        self.batch_norm1_reduced = rm.BatchNormalize(mode='feature')
        self.conv1_1 = rm.Conv2d(channels[1], filter=3, padding=1)
        self.batch_norm1_1 = rm.BatchNormalize(mode='feature')
        self.conv1_2 = rm.Conv2d(channels[1], filter=3, stride=2)
        self.batch_norm1_2 = rm.BatchNormalize(mode='feature')

        self.conv2 = rm.Conv2d(channels[2], filter=3, stride=2)
        self.batch_norm2 = rm.BatchNormalize(mode='feature')

    def forward(self, x):
        t1 = rm.relu(self.batch_norm1_reduced(self.conv1_reduced(x)))
        t1 = rm.relu(self.batch_norm1_1(self.conv1_1(t1)))
        t1 = rm.relu(self.batch_norm1_2(self.conv1_2(t1)))

        t2 = rm.relu(self.batch_norm2(self.conv2(x)))

        t3 = rm.max_pool2d(x, filter=3, stride=2)
        return rm.concat([t1, t2, t3])


class InceptionV2BlockC(rm.Model):
    def __init__(self, channels=[192, 128, 192, 128, 192, 192]):
        self.conv1 = rm.Conv2d(channels[0], filter=1)
        self.batch_norm1 = rm.BatchNormalize(mode='feature')

        self.conv2_reduced = rm.Conv2d(channels[1], filter=1)
        self.batch_norm2_reduced = rm.BatchNormalize(mode='feature')
        self.conv2_1 = rm.Conv2d(channels[1], filter=(3, 1), padding=(1, 0))
        self.batch_norm2_1 = rm.BatchNormalize(mode='feature')
        self.conv2_2 = rm.Conv2d(channels[2], filter=(1, 3), padding=(0, 1))
        self.batch_norm2_2 = rm.BatchNormalize(mode='feature')

        self.conv3_reduced = rm.Conv2d(channels[3], filter=1)
        self.batch_norm3_reduced = rm.BatchNormalize(mode='feature')
        self.conv3_1 = rm.Conv2d(channels[3], filter=(3, 1), padding=(1, 0))
        self.batch_norm3_1 = rm.BatchNormalize(mode='feature')
        self.conv3_2 = rm.Conv2d(channels[3], filter=(1, 3), padding=(0, 1))
        self.batch_norm3_2 = rm.BatchNormalize(mode='feature')
        self.conv3_3 = rm.Conv2d(channels[3], filter=(3, 1), padding=(1, 0))
        self.batch_norm3_3 = rm.BatchNormalize(mode='feature')
        self.conv3_4 = rm.Conv2d(channels[4], filter=(1, 3), padding=(0, 1))
        self.batch_norm3_4 = rm.BatchNormalize(mode='feature')

        self.conv4 = rm.Conv2d(channels[5], filter=1)
        self.batch_norm4 = rm.BatchNormalize(mode='feature')

    def forward(self, x):
        t1 = rm.relu(self.batch_norm1(self.conv1(x)))

        t2 = rm.relu(self.batch_norm2_reduced(self.conv2_reduced(x)))
        t2 = rm.relu(self.batch_norm2_1(self.conv2_1(t2)))
        t2 = rm.relu(self.batch_norm2_2(self.conv2_2(t2)))

        t3 = rm.relu(self.batch_norm3_reduced(self.conv3_reduced(x)))
        t3 = rm.relu(self.batch_norm3_1(self.conv3_1(t3)))
        t3 = rm.relu(self.batch_norm3_2(self.conv3_2(t3)))
        t3 = rm.relu(self.batch_norm3_3(self.conv3_3(t3)))
        t3 = rm.relu(self.batch_norm3_4(self.conv3_4(t3)))

        t4 = rm.max_pool2d(x, filter=3, stride=1, padding=1)
        t4 = rm.relu(self.batch_norm4(self.conv4(t4)))

        return rm.concat([
            t1, t2, t3, t4
        ])


class InceptionV2BlockD(rm.Model):
    def __init__(self, channels=[192, 320, 192, 192]):
        self.conv1_reduced = rm.Conv2d(channels[0], filter=1)
        self.batch_norm1_reduced = rm.BatchNormalize(mode='feature')
        self.conv1 = rm.Conv2d(channels[1], filter=3, stride=2)
        self.batch_norm1 = rm.BatchNormalize(mode='feature')

        self.conv2_reduced = rm.Conv2d(channels[2], filter=1)
        self.batch_norm2_reduced = rm.BatchNormalize(mode='feature')
        self.conv2_1 = rm.Conv2d(channels[3], filter=3, padding=1)
        self.batch_norm2_1 = rm.BatchNormalize(mode='feature')
        self.conv2_2 = rm.Conv2d(channels[3], filter=3, stride=2)
        self.batch_norm2_2 = rm.BatchNormalize(mode='feature')

    def forward(self, x):
        t1 = rm.relu(self.batch_norm1_reduced(self.conv1_reduced(x)))
        t1 = rm.relu(self.batch_norm1(self.conv1(t1)))

        t2 = rm.relu(self.batch_norm2_reduced(self.conv2_reduced(x)))
        t2 = rm.relu(self.batch_norm2_1(self.conv2_1(t2)))
        t2 = rm.relu(self.batch_norm2_2(self.conv2_2(t2)))

        t3 = rm.max_pool2d(x, filter=3, stride=2)
        return rm.concat([
            t1, t2, t3
        ])


class InceptionV2BlockE(rm.Model):
    def __init__(self, channels=[320, 384, 384, 448, 384, 192]):
        self.conv1 = rm.Conv2d(channels[0], filter=1)
        self.batch_norm1 = rm.BatchNormalize(mode='feature')

        self.conv2_reduced = rm.Conv2d(channels[1], filter=1)
        self.batch_norm2_reduced = rm.BatchNormalize(mode='feature')
        self.conv2_1 = rm.Conv2d(channels[2], filter=(3, 1), padding=(1, 0))
        self.batch_norm2_1 = rm.BatchNormalize(mode='feature')
        self.conv2_2 = rm.Conv2d(channels[2], filter=(1, 3), padding=(0, 1))
        self.batch_norm2_2 = rm.BatchNormalize(mode='feature')

        self.conv3_reduced = rm.Conv2d(channels[3], filter=1)
        self.batch_norm3_reduced = rm.BatchNormalize(mode='feature')
        self.conv3_1 = rm.Conv2d(channels[4], filter=3, padding=1)
        self.batch_norm3_1 = rm.BatchNormalize(mode='feature')
        self.conv3_2 = rm.Conv2d(channels[4], filter=(3, 1), padding=(1, 0))
        self.batch_norm3_2 = rm.BatchNormalize(mode='feature')
        self.conv3_3 = rm.Conv2d(channels[4], filter=(1, 3), padding=(0, 1))
        self.batch_norm3_3 = rm.BatchNormalize(mode='feature')

        self.conv4 = rm.Conv2d(channels[5], filter=1)
        self.batch_norm4 = rm.BatchNormalize(mode='feature')

    def forward(self, x):
        t1 = rm.relu(self.batch_norm1(self.conv1(x)))

        t2 = rm.relu(self.batch_norm2_reduced(self.conv2_reduced(x)))
        t2_1 = rm.relu(self.batch_norm2_1(self.conv2_1(t2)))
        t2_2 = rm.relu(self.batch_norm2_2(self.conv2_2(t2)))
        t2 = rm.concat([t2_1, t2_2])

        t3 = rm.relu(self.batch_norm3_reduced(self.conv3_reduced(x)))
        t3 = rm.relu(self.batch_norm3_1(self.conv3_1(t3)))
        t3_1 = rm.relu(self.batch_norm3_2(self.conv3_2(t3)))
        t3_2 = rm.relu(self.batch_norm3_3(self.conv3_3(t3)))
        t3 = rm.concat([t3_1, t3_2])

        t4 = rm.max_pool2d(x, filter=3, stride=1, padding=1)
        t4 = rm.relu(self.batch_norm4(self.conv4(t4)))
        return rm.concat([
            t1, t2, t3, t4
        ])

class InceptionV2Stem(rm.Model):
    def __init__(self):
        self.conv1 = rm.Conv2d(32, filter=3, padding=0, stride=2)
        self.batch_norm1 = rm.BatchNormalize(mode='feature')

        self.conv2 = rm.Conv2d(32, filter=3, padding=0, stride=1)
        self.batch_norm2 = rm.BatchNormalize(mode='feature')

        self.conv3 = rm.Conv2d(64, filter=3, padding=1, stride=1)
        self.batch_norm3 = rm.BatchNormalize(mode='feature')

        self.conv4 = rm.Conv2d(80, filter=3, stride=1)
        self.batch_norm4 = rm.BatchNormalize(mode='feature')
        self.conv5 = rm.Conv2d(192, filter=3, stride=2)
        self.batch_norm5 = rm.BatchNormalize(mode='feature')
        self.conv6 = rm.Conv2d(192, filter=3, stride=1, padding=1)
        self.batch_norm6 = rm.BatchNormalize(mode='feature')

    def forward(self, x):
        t = rm.relu(self.batch_norm1(self.conv1(x)))
        t = rm.relu(self.batch_norm2(self.conv2(t)))
        t = rm.relu(self.batch_norm3(self.conv3(t)))

        t = rm.max_pool2d(t, filter=3, stride=2)
        t = rm.relu(self.batch_norm4(self.conv4(t)))
        t = rm.relu(self.batch_norm5(self.conv5(t)))
        t = rm.relu(self.batch_norm6(self.conv6(t)))
        return t

class InceptionV3(InceptionBase):
    """ Inception V3 model
    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        class_map: Array of class names
        load_weight(bool): True if the pre-trained weight is loaded.
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna
    Rethinking the Inception Architecture for Computer Vision
    https://arxiv.org/abs/1512.00567
    """
    WEIGHT_URL = "https://app.box.com/shared/static/eovmxxgzyh5vg2kpcukjj8ypnxng4j5v.h5"
    WEIGHT_PATH = os.path.join(DIR, 'inceptionv3.h5')

    def __init__(self, class_map, load_weight=False, imsize=(299, 299), opt=rm.Sgd(0.045, 0.9), train_whole_network=True):
        n_class = len(class_map)
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        self.imsize = imsize
        self._opt = opt
        self._train_whole_network = train_whole_network
        self.base1 = rm.Sequential([
                InceptionV2Stem(),
                InceptionV2BlockA([64, 48, 64, 64, 96, 32]),
                InceptionV2BlockA(),
                InceptionV2BlockA(),
                InceptionV2BlockB(),
                InceptionV2BlockC([192, 128, 192, 128, 192, 192]),
                InceptionV2BlockC(),
                InceptionV2BlockC(),
                InceptionV2BlockC()])
        self.aux1 = rm.Sequential([
                rm.AveragePool2d(filter=5, stride=3),
                rm.Conv2d(128, filter=1),
                rm.BatchNormalize(mode='feature'),
                rm.Relu(),
                rm.Conv2d(768, filter=1),
                rm.BatchNormalize(mode='feature'),
                rm.Relu(),
                rm.Flatten(),
                rm.Dense(n_class)])

        self.base2 = rm.Sequential([
                InceptionV2BlockD(),
                InceptionV2BlockE(),
                InceptionV2BlockE(),
                rm.AveragePool2d(filter=8),
                rm.Flatten()])

        self.aux2 = rm.Dense(n_class)

        super(InceptionV3, self).__init__(class_map)
        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if n_class != 1000:
            self.aux1.params = {}
            self.aux2.params = {}

    def forward(self, x):
        self.base1.set_auto_update(self._train_whole_network)
        self.base2.set_auto_update(self._train_whole_network)
        t = self.base1(x)
        out1 = self.aux1(t)
        t = self.base2(t)
        out2 = self.aux2(t)

        return out1, out2

    def loss(self, x, y):
        return rm.softmax_cross_entropy(x[0], y) + rm.softmax_cross_entropy(x[1], y)

    def predict(self, img_list):
        self.set_models(inference=True)
        if isinstance(img_list, (list, str)):
            if isinstance(img_list, (tuple, list)):
                img_array = np.vstack([load_img(path, self.imsize)[None] for path in img_list])
                img_array = self.preprocess(img_array)
            else:
                img_array = load_img(img_list, self.imsize)[None]
                img_array = self.preprocess(img_array)
                return np.argmax(rm.softmax(self(img_array)[1]).as_ndarray(), axis=1)[0]
        else:
            img_array = img_list
        return np.argmax(rm.softmax(self(img_array)[1]).as_ndarray(), axis=1)


class InceptionV2(InceptionBase):
    """ Inception V2 model
    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        class_map: Array of class names
        load_weight(bool): True if the pre-trained weight is loaded.
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna
    Rethinking the Inception Architecture for Computer Vision
    https://arxiv.org/abs/1512.00567
    """

    WEIGHT_URL = "https://app.box.com/shared/static/eovmxxgzyh5vg2kpcukjj8ypnxng4j5v.h5"
    WEIGHT_PATH = os.path.join(DIR, 'inceptionv2.h5')

    def __init__(self, class_map, load_weight=False, imsize=(299, 299), opt=rm.Sgd(0.045, 0.9), train_whole_network=True):
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        n_class = len(class_map)
        self.imsize = imsize
        self._opt = opt
        self._train_whole_network = train_whole_network
        self.base1 = rm.Sequential([
                InceptionV2Stem(),
                InceptionV2BlockA([64, 48, 64, 64, 96, 32]),
                InceptionV2BlockA(),
                InceptionV2BlockA(),
                InceptionV2BlockB(),
                InceptionV2BlockC([192, 128, 192, 128, 192, 192]),
                InceptionV2BlockC(),
                InceptionV2BlockC(),
                InceptionV2BlockC()])
        self.aux1 = rm.Sequential([
                rm.AveragePool2d(filter=5, stride=3),
                rm.Conv2d(128, filter=1),
                rm.Relu(),
                rm.Conv2d(768, filter=1),
                rm.Relu(),
                rm.Flatten(),
                rm.Dense(n_class)])

        self.base2 = rm.Sequential([
                InceptionV2BlockD(),
                InceptionV2BlockE(),
                InceptionV2BlockE(),
                rm.AveragePool2d(filter=8),
                rm.Flatten()])

        self.aux2 = rm.Dense(n_class)

        super(InceptionV2, self).__init__(class_map)
        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if n_class != 1000:
            self.aux1.params = {}
            self.aux2.params = {}

    def forward(self, x):
        self.base1.set_auto_update(self._train_whole_network)
        self.base2.set_auto_update(self._train_whole_network)
        t = self.base1(x)
        out1 = self.aux1(t)
        t = self.base2(t)
        out2 = self.aux2(t)

        return out1, out2

    def loss(self, x, y):
        return rm.softmax_cross_entropy(x[0], y) + rm.softmax_cross_entropy(x[1], y)

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
            if current_epoch % 2 == 0:
                lr = self._opt._lr * 0.94
            self._opt._lr = lr
            return self._opt

    def predict(self, img_list):
        self.set_models(inference=True)
        if isinstance(img_list, (list, str)):
            if isinstance(img_list, (tuple, list)):
                img_array = np.vstack([load_img(path, self.imsize)[None] for path in img_list])
                img_array = self.preprocess(img_array)
            else:
                img_array = load_img(img_list, self.imsize)[None]
                img_array = self.preprocess(img_array)
                return np.argmax(rm.softmax(self(img_array)[1]).as_ndarray(), axis=1)[0]
        else:
            img_array = img_list
        return np.argmax(rm.softmax(self(img_array)[1]).as_ndarray(), axis=1)

class InceptionV4Stem(rm.Model):
    def __init__(self):
        self.conv1 = rm.Conv2d(32, filter=3, padding=0, stride=2)
        self.batch_norm1 = rm.BatchNormalize(mode='feature')

        self.conv2 = rm.Conv2d(32, filter=3, padding=0, stride=1)
        self.batch_norm2 = rm.BatchNormalize(mode='feature')

        self.conv3 = rm.Conv2d(64, filter=3, padding=1, stride=1)
        self.batch_norm3 = rm.BatchNormalize(mode='feature')

        self.conv4 = rm.Conv2d(96, filter=3, stride=2)
        self.batch_norm4 = rm.BatchNormalize(mode='feature')

        self.conv5_1_1 = rm.Conv2d(64, filter=1)
        self.batch_norm5_1_1 = rm.BatchNormalize(mode='feature')
        self.conv5_1_2 = rm.Conv2d(96, filter=3)
        self.batch_norm5_1_2 = rm.BatchNormalize(mode='feature')

        self.conv5_2_1 = rm.Conv2d(64, filter=1)
        self.batch_norm5_2_1 = rm.BatchNormalize(mode='feature')
        self.conv5_2_2 = rm.Conv2d(64, filter=(7, 1), padding=(3, 0))
        self.batch_norm5_2_2 = rm.BatchNormalize(mode='feature')
        self.conv5_2_3 = rm.Conv2d(64, filter=(1, 7), padding=(0, 3))
        self.batch_norm5_2_3 = rm.BatchNormalize(mode='feature')
        self.conv5_2_4 = rm.Conv2d(96, filter=3)
        self.batch_norm5_2_4 = rm.BatchNormalize(mode='feature')

        self.conv6 = rm.Conv2d(192, filter=3, stride=2)
        self.batch_norm6 = rm.BatchNormalize(mode='feature')

    def forward(self, x):
        t = rm.relu(self.batch_norm1(self.conv1(x)))
        t = rm.relu(self.batch_norm2(self.conv2(t)))
        t = rm.relu(self.batch_norm3(self.conv3(t)))

        t1 = rm.max_pool2d(t, filter=3, stride=2)
        t2 = rm.relu(self.batch_norm4(self.conv4(t)))

        t = rm.concat([t1, t2])

        t1 = rm.relu(self.batch_norm5_1_1(self.conv5_1_1(t)))
        t1 = rm.relu(self.batch_norm5_1_2(self.conv5_1_2(t1)))

        t2 = rm.relu(self.batch_norm5_2_1(self.conv5_2_1(t)))
        t2 = rm.relu(self.batch_norm5_2_2(self.conv5_2_2(t2)))
        t2 = rm.relu(self.batch_norm5_2_3(self.conv5_2_3(t2)))
        t2 = rm.relu(self.batch_norm5_2_4(self.conv5_2_4(t2)))
        t = rm.concat([t1, t2])

        t1 = rm.relu(self.batch_norm6(self.conv6(t)))
        t2 = rm.max_pool2d(t, filter=3, stride=2)
        return rm.concat([t1, t2])


class InceptionV4BlockA(rm.Model):
    def __init__(self, channels=[64, 48, 64, 64, 96, 32]):
        self.conv1 = rm.Conv2d(96, filter=1)
        self.batch_norm1 = rm.BatchNormalize(mode='feature')

        self.conv2_reduced = rm.Conv2d(64, filter=1)
        self.batch_norm2_reduced = rm.BatchNormalize(mode='feature')
        self.conv2 = rm.Conv2d(96, filter=3, padding=1)
        self.batch_norm2 = rm.BatchNormalize(mode='feature')

        self.conv3_reduced = rm.Conv2d(64, filter=1)
        self.batch_norm3_reduced = rm.BatchNormalize(mode='feature')
        self.conv3_1 = rm.Conv2d(96, filter=3, padding=1)
        self.batch_norm3_1 = rm.BatchNormalize(mode='feature')
        self.conv3_2 = rm.Conv2d(96, filter=3, padding=1)
        self.batch_norm3_2 = rm.BatchNormalize(mode='feature')

        self.conv4 = rm.Conv2d(96, filter=1)
        self.batch_norm4 = rm.BatchNormalize(mode='feature')

    def forward(self, x):
        t1 = rm.relu(self.batch_norm1(self.conv1(x)))

        t2 = rm.relu(self.batch_norm2_reduced(self.conv2_reduced(x)))
        t2 = rm.relu(self.batch_norm2(self.conv2(t2)))

        t3 = rm.relu(self.batch_norm3_reduced(self.conv3_reduced(x)))
        t3 = rm.relu(self.batch_norm3_1(self.conv3_1(t3)))
        t3 = rm.relu(self.batch_norm3_2(self.conv3_2(t3)))

        t4 = rm.max_pool2d(x, filter=3, stride=1, padding=1)
        t4 = rm.relu(self.batch_norm4(self.conv4(t4)))

        return rm.concat([
            t1, t2, t3, t4
        ])


class InceptionV4ReductionA(rm.Model):
    def __init__(self):
        # k, l, m, n
        # 192, 224, 256, 384
        self.conv1 = rm.Conv2d(384, filter=3, stride=2)
        self.batch_norm1 = rm.BatchNormalize(mode='feature')

        self.conv2_red = rm.Conv2d(192, filter=1)
        self.batch_norm2_red = rm.BatchNormalize(mode='feature')
        self.conv2_1 = rm.Conv2d(224, filter=3, padding=1)
        self.batch_norm2_1 = rm.BatchNormalize(mode='feature')
        self.conv2_2 = rm.Conv2d(256, filter=3, stride=2)
        self.batch_norm2_2 = rm.BatchNormalize(mode='feature')

    def forward(self, x):
        t1 = rm.max_pool2d(x, filter=3, stride=2)

        t2 = rm.relu(self.batch_norm1(self.conv1(x)))

        t3 = rm.relu(self.batch_norm2_red(self.conv2_red(x)))
        t3 = rm.relu(self.batch_norm2_1(self.conv2_1(t3)))
        t3 = rm.relu(self.batch_norm2_2(self.conv2_2(t3)))

        return rm.concat([
            t1, t2, t3
        ])


class InceptionV4BlockB(rm.Model):
    def __init__(self):
        self.conv1 = rm.Conv2d(128, filter=1)
        self.batch_norm1 = rm.BatchNormalize(mode='feature')

        self.conv2 = rm.Conv2d(384, filter=3, padding=1)
        self.batch_norm2 = rm.BatchNormalize(mode='feature')

        self.conv3_1 = rm.Conv2d(192, filter=1)
        self.batch_norm3_1 = rm.BatchNormalize(mode='feature')
        self.conv3_2 = rm.Conv2d(224, filter=(1, 7), padding=(0, 3))
        self.batch_norm3_2 = rm.BatchNormalize(mode='feature')
        self.conv3_3 = rm.Conv2d(256, filter=(7, 1), padding=(3, 0))
        self.batch_norm3_3 = rm.BatchNormalize(mode='feature')

        self.conv4_1 = rm.Conv2d(192, filter=1)
        self.batch_norm4_1 = rm.BatchNormalize(mode='feature')
        self.conv4_2 = rm.Conv2d(192, filter=(1, 7), padding=(0, 3))
        self.batch_norm4_2 = rm.BatchNormalize(mode='feature')
        self.conv4_3 = rm.Conv2d(224, filter=(7, 1), padding=(3, 0))
        self.batch_norm4_3 = rm.BatchNormalize(mode='feature')
        self.conv4_4 = rm.Conv2d(224, filter=(1, 7), padding=(0, 3))
        self.batch_norm4_4 = rm.BatchNormalize(mode='feature')
        self.conv4_5 = rm.Conv2d(256, filter=(7, 1), padding=(3, 0))
        self.batch_norm4_5 = rm.BatchNormalize(mode='feature')

    def forward(self, x):
        t1 = rm.average_pool2d(x, filter=3, padding=1)
        t1 = rm.relu(self.batch_norm1(self.conv1(t1)))

        t2 = rm.relu(self.batch_norm2(self.conv2(x)))

        t3 = rm.relu(self.batch_norm3_1(self.conv3_1(x)))
        t3 = rm.relu(self.batch_norm3_2(self.conv3_2(t3)))
        t3 = rm.relu(self.batch_norm3_3(self.conv3_3(t3)))

        t4 = rm.relu(self.batch_norm4_1(self.conv4_1(x)))
        t4 = rm.relu(self.batch_norm4_2(self.conv4_2(t4)))
        t4 = rm.relu(self.batch_norm4_3(self.conv4_3(t4)))
        t4 = rm.relu(self.batch_norm4_4(self.conv4_4(t4)))
        t4 = rm.relu(self.batch_norm4_5(self.conv4_5(t4)))
        return rm.concat([t1, t2, t3, t4])


class InceptionV4ReductionB(rm.Model):
    def __init__(self):
        # k, l, m, n
        # 192, 224, 256, 384
        self.conv1_red = rm.Conv2d(192, filter=1)
        self.batch_norm1_red = rm.BatchNormalize(mode='feature')
        self.conv1 = rm.Conv2d(192, filter=3, stride=2)
        self.batch_norm1 = rm.BatchNormalize(mode='feature')

        self.conv2_red = rm.Conv2d(256, filter=1)
        self.batch_norm2_red = rm.BatchNormalize(mode='feature')
        self.conv2_1 = rm.Conv2d(256, filter=(1, 7), padding=(0, 3))
        self.batch_norm2_1 = rm.BatchNormalize(mode='feature')
        self.conv2_2 = rm.Conv2d(320, filter=(7, 1), padding=(3, 0))
        self.batch_norm2_2 = rm.BatchNormalize(mode='feature')
        self.conv2_3 = rm.Conv2d(320, filter=3, stride=2)
        self.batch_norm2_3 = rm.BatchNormalize(mode='feature')

    def forward(self, x):
        t1 = rm.max_pool2d(x, filter=3, stride=2)

        t2 = rm.relu(self.batch_norm1_red(self.conv1_red(x)))
        t2 = rm.relu(self.batch_norm1(self.conv1(t2)))

        t3 = rm.relu(self.batch_norm2_red(self.conv2_red(x)))
        t3 = rm.relu(self.batch_norm2_1(self.conv2_1(t3)))
        t3 = rm.relu(self.batch_norm2_2(self.conv2_2(t3)))
        t3 = rm.relu(self.batch_norm2_3(self.conv2_3(t3)))

        return rm.concat([
            t1, t2, t3
        ])


class InceptionV4BlockC(rm.Model):
    def __init__(self):
        self.conv1 = rm.Conv2d(256, filter=1)
        self.batch_norm1 = rm.BatchNormalize(mode='feature')

        self.conv2 = rm.Conv2d(256, filter=1)
        self.batch_norm2 = rm.BatchNormalize(mode='feature')

        self.conv3_red = rm.Conv2d(384, filter=1)
        self.batch_norm3_red = rm.BatchNormalize(mode='feature')
        self.conv3_1 = rm.Conv2d(256, filter=(1, 3), padding=(0, 1))
        self.batch_norm3_1 = rm.BatchNormalize(mode='feature')
        self.conv3_2 = rm.Conv2d(256, filter=(3, 1), padding=(1, 0))
        self.batch_norm3_2 = rm.BatchNormalize(mode='feature')

        self.conv4_red = rm.Conv2d(384, filter=1)
        self.batch_norm4_red = rm.BatchNormalize(mode='feature')
        self.conv4_1 = rm.Conv2d(448, filter=(1, 3), padding=(0, 1))
        self.batch_norm4_1 = rm.BatchNormalize(mode='feature')
        self.conv4_2 = rm.Conv2d(512, filter=(3, 1), padding=(1, 0))
        self.batch_norm4_2 = rm.BatchNormalize(mode='feature')
        self.conv4_3 = rm.Conv2d(256, filter=(1, 3), padding=(0, 1))
        self.batch_norm4_3 = rm.BatchNormalize(mode='feature')
        self.conv4_4 = rm.Conv2d(256, filter=(3, 1), padding=(1, 0))
        self.batch_norm4_4 = rm.BatchNormalize(mode='feature')

    def forward(self, x):
        t1 = rm.average_pool2d(x, filter=3, stride=1, padding=1)
        t1 = rm.relu(self.batch_norm1(self.conv1(t1)))

        t2 = rm.relu(self.batch_norm2(self.conv2(x)))

        t3 = rm.relu(self.batch_norm3_red(self.conv3_red(x)))
        t3_1 = rm.relu(self.batch_norm3_1(self.conv3_1(t3)))
        t3_2 = rm.relu(self.batch_norm3_2(self.conv3_2(t3)))

        t4 = rm.relu(self.batch_norm4_red(self.conv4_red(x)))
        t4 = rm.relu(self.batch_norm4_1(self.conv4_1(t4)))
        t4 = rm.relu(self.batch_norm4_2(self.conv4_2(t4)))
        t4_1 = rm.relu(self.batch_norm4_3(self.conv4_3(t4)))
        t4_2 = rm.relu(self.batch_norm4_4(self.conv4_4(t4)))

        return rm.concat([
            t1, t2, t3_1, t3_2, t4_1, t4_2
        ])


class InceptionV4(InceptionBase):
    """ Inception V4 model
    If the argument load_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        class_map: Array of class names
        load_weight(bool): True if the pre-trained weight is loaded.
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument n_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
    Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
    https://arxiv.org/abs/1602.07261
    """

    WEIGHT_URL = "https://app.box.com/shared/static/eovmxxgzyh5vg2kpcukjj8ypnxng4j5v.h5"
    WEIGHT_PATH = os.path.join(DIR, 'inceptionv4.h5')

    def __init__(self, class_map, load_weight=False, imsize=(299, 299), opt=rm.Sgd(0.045, 0.9), train_whole_network=False):
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        self.imsize = imsize
        n_class = len(class_map)
        self._opt = opt

        layers = [InceptionV4Stem(),
                  InceptionV4BlockA(),
                  InceptionV4BlockA(),
                  InceptionV4BlockA(),
                  InceptionV4BlockA(),
                  InceptionV4ReductionA(),
                  InceptionV4BlockB(),
                  InceptionV4BlockB(),
                  InceptionV4BlockB(),
                  InceptionV4BlockB(),
                  InceptionV4BlockB(),
                  InceptionV4BlockB(),
                  InceptionV4BlockB(),
                  InceptionV4ReductionB(),
                  InceptionV4BlockC(),
                  InceptionV4BlockC(),
                  InceptionV4BlockC(),
                  rm.AveragePool2d(filter=8),
                  rm.Flatten()
                  ]

        self._freezed_network = rm.Sequential(layers)
        self._network = rm.Dense(n_class)
        self._train_whole_network = train_whole_network
        self.imsize = imsize

        super(InceptionV4, self).__init__(class_map)

        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)
        if n_class != 1000:
            self.network.params = {}

    @property
    def freezed_network(self):
        return self._freezed_network

    @property
    def network(self):
        return self._network

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
            if current_epoch % 8 == 0:
                lr = self._opt._lr * 0.94
            self._opt._lr = lr
            return self._opt

    def forward(self, x):
        self.freezed_network.set_auto_update(self._train_whole_network)
        t = self.freezed_network(x)
        t = rm.dropout(t, 0.2)

        t = self.network(t)
        return t
