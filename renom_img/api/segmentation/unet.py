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

class UNet(rm.Model):
    """ U-Net: Convolutional Networks for Biomedical Image Segmentation

    Args:
        n_class (int): The number of classes

    Example:
        >>> import renom as rm
        >>> import numpy as np
        >>> from renom_img.api.segmentation.unet import UNet
        >>> n, c, h, w = (2, 12, 64, 64)
        >>> x = rm.Variable(np.random.rand(n, c, h, w))
        >>> model = UNet(12)
        >>> t = model(x)
        >>> t.shape
        (2, 12, 64, 64)

    Note:
        Olaf Ronneberger, Philipp Fischer, Thomas Brox
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        https://arxiv.org/pdf/1505.04597.pdf
    """

    WEIGHT_URL = "https://app.box.com/shared/static/eovmxxgzyh5vg2kpcukjj8ypnxng4j5v.h5"
    WEIGHT_PATH = os.path.join(DIR, 'u-net.h5')

    def __init__(self, class_map, load_weight=False, imsize=(512, 512), train_whole_network=False):
        n_class = len(class_map)
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
        self.conv9 = rm.Conv2d(n_class, filter=1)
        self._train_whole_network = train_whole_network
        self.imsize = imsize
        self.class_map = class_map
        self._opt = rm.Adam(lr=1e-3)

        if load_weight:
            try:
                self.load(self.WEIGHT_PATH)
            except:
                download(self.WEIGHT_URL, self.WEIGHT_PATH)
            self.load(self.WEIGHT_PATH)

    def preprocess(self, x):
        """Image preprocess for Yolov1.

        :math:`new_x = x*2/255. - 1`

        Args:
            x (ndarray):

        Returns:
            (ndarray): Preprocessed data.
        """
        return x / 255. * 2 - 1

    def loss(self, x, y):
        return rm.softmax_cross_entropy(x, y) / (self.imsize[0] * self.imsize[1])

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


    def predict(self, img_list):
        self.set_models(inference=True)
        if isinstance(img_list, (list, str)):
            if isinstance(img_list, (tuple, list)):
                img_array = np.vstack([load_img(path, self.imsize)[None] for path in img_list])
                img_array = self.preprocess(img_array)
            else:
                img_array = load_img(img_list, self.imsize)[None]
                img_array = self.preprocess(img_array)
                return np.argmax(rm.softmax(self(img_array)).as_ndarray(), axis=1)[0]
        else:
            img_array = img_list
        return np.argmax(self(img_array).as_ndarray(), axis=1)

    def regularize(self):
        """Regularize term. You can use this function to add regularize term to 
        loss function.

        Example:
            >>> import numpy as np
            >>> from renom_img.api.detection.yolo_v1 import Yolov1
            >>> x = np.random.rand(1, 3, 224, 224)
            >>> y = np.random.rand(1, (5*2+20)*7*7)
            >>> model = Yolov1()
            >>> loss = model.loss(x, y)
            >>> reg_loss = loss + model.regularize() # Add weight decay term.

        """
        reg = 0
        for layer in self.iter_models():
            if hasattr(layer, "params") and hasattr(layer.params, "w"):
                reg += rm.sum(layer.params.w * layer.params.w)
        return 0.0004 * reg

    def _freeze_network(self):
        self.conv1_1.set_auto_update(self._train_whole_network)
        self.conv1_2.set_auto_update(self._train_whole_network)
        self.conv2_1.set_auto_update(self._train_whole_network)
        self.conv2_2.set_auto_update(self._train_whole_network)
        self.conv3_1.set_auto_update(self._train_whole_network)
        self.conv3_2.set_auto_update(self._train_whole_network)
        self.conv4_1.set_auto_update(self._train_whole_network)
        self.conv4_2.set_auto_update(self._train_whole_network)
        self.conv5_1.set_auto_update(self._train_whole_network)
        self.conv5_2.set_auto_update(self._train_whole_network)

    def forward(self, x):
        self._freeze_network()
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

    def fit(self, train_img_path_list=None, train_annotation_path_list=None, augmentation=None, valid_img_path_list=None, valid_annotation_path_list=None,  epoch=200, batch_size=16, callback_end_epoch=None):
        if train_img_path_list is not None and train_annotation_path_list is not None:
            train_dist = ImageDistributor(train_img_path_list, train_annotation_path_list, augmentation=augmentation)
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


