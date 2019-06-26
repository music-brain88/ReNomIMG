import os
import sys
sys.setrecursionlimit(3000)
import renom as rm
import numpy as np
from tqdm import tqdm
from PIL import Image
from renom_img import __version__
from renom_img.api.utility.misc.download import download
from renom_img.api.classification import Classification
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.target import DataBuilderClassification
from renom_img.api.utility.optimizer import OptimizerInception
from renom_img.api.cnn.inception import CNN_InceptionV1, CNN_InceptionV2, CNN_InceptionV3, CNN_InceptionV4

RESIZE_METHOD = Image.BILINEAR

class TargetBuilderInception():
    def __init__(self, class_map, imsize):
        self.class_map = class_map
        self.imsize = imsize

    def __call__(self, *args, **kwargs):
        return self.build(*args, **kwargs)

    def preprocess(self, x):

        return x/255.

    def resize_img(self, img_list, label_list):
        im_list = []

        for img in img_list:
            channel_last = img.transpose(1, 2, 0)
            img = Image.fromarray(np.uint8(channel_last))
            img = img.resize(self.imsize, RESIZE_METHOD).convert('RGB')
            im_list.append(np.asarray(img))

        return np.asarray(im_list).transpose(0, 3, 1, 2).astype(np.float32), np.asarray(label_list)

    def _load(self, path):
        """ Loads an image

        Args:
            path(str): A path of an image

        Returns:
            (tuple): Returns image(numpy.array), the ratio of the given width to the actual image width,
                     and the ratio of the given height to the actual image height
        """
        img = Image.open(path)
        img.load()
        w, h = img.size
        img = img.convert('RGB')
        # img = img.resize(self.imsize, RESIZE_METHOD)
        img = np.asarray(img).transpose(2, 0, 1).astype(np.float32)
        return img, self.imsize[0] / float(w), self.imsize[1] / h


    def build(self, img_path_list, annotation_list=None, augmentation=None, **kwargs):
        """ Builds an array of images and corresponding labels

        Args:
            img_path_list(list): List of input image paths.
            annotation_list(list): List of class id
                                    [1, 4, 6 (int)]
            augmentation(Augmentation): Instance of the augmentation class.

        Returns:
            (tuple): Batch of images and corresponding one hot labels for each image in a batch
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
        for img_path, an_data in zip(img_path_list, annotation_list):
            one_hot = np.zeros(n_class)
            img, sw, sh = self._load(img_path)
            img_list.append(img)
            one_hot[an_data] = 1.
            label_list.append(one_hot)

        if augmentation is not None:
            img_list, label_list = augmentation(img_list, label_list, mode="classification")

        img_list, label_list = self.resize_img(img_list, label_list)

        return self.preprocess(np.array(img_list)), np.array(label_list)


class InceptionV1(Classification):
    """ Inception V1 model

    If the argument load_pretrained_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        class_map: Array of class names
        load_pretrained_weight(bool): True if the pre-trained weight is loaded.
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument num_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    References:
        Christian Szegedy, Wei Liu, Yangqing Jia , Pierre Sermanet, Scott Reed ,Dragomir Anguelov,
        Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich
        Going Deeper with Convolutions
        https://www.cs.unc.edu/~wliu/papers/GoogLeNet.pdf
    """

    def __init__(self, class_map=[], imsize=(224, 224), load_pretrained_weight=False, train_whole_network=False):

        self._model = CNN_InceptionV1(1)
        super(InceptionV1, self).__init__(class_map, imsize, load_pretrained_weight,
                                          train_whole_network, self._model)

        self._model.set_output_size(self.num_class)
        self._model.set_train_whole(train_whole_network)

        self.default_optimizer = OptimizerInception(1)
        self.decay_rate = 0.0005
 

        self._model.aux1.params = {}
        self._model.aux2.params = {}
        self._model.aux3.params = {}

    def loss(self, x, y):
        return 0.3 * rm.softmax_cross_entropy(x[0], y) + 0.3 * rm.softmax_cross_entropy(x[1], y) + rm.softmax_cross_entropy(x[2], y)

    def predict(self, img_list, batch_size=1):
        self.set_models(inference=True)
        if isinstance(img_list, (list, str)):
            img_builder = self.build_data()
            if isinstance(img_list, (tuple, list)):
                results = []
                bar = tqdm(range(int(np.ceil(len(img_list) / batch_size))))
                for batch_num in range(0,len(img_list),batch_size):
                    results.extend(np.argmax(rm.softmax(self(img_builder(img_path_list=img_list[batch_num:batch_num+batch_size]))[2]).as_ndarray(), axis=1))
                    bar.update(1)
                bar.close()
                return results
            else:
                return np.argmax(rm.softmax(self(img_builder(img_path_list=[img_list]))[2]).as_ndarray(),axis=1)[0]
        else:
            img_array = img_list
        return np.argmax(rm.softmax(self(img_array)[2]).as_ndarray(), axis=1)


    def build_data(self):
        return TargetBuilderInception(self.class_map, self.imsize)


class InceptionV3(Classification):
    """ Inception V3 model

    If the argument load_pretrained_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        class_map: Array of class names
        load_pretrained_weight(bool): True if the pre-trained weight is loaded.
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument num_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    References:
        Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna
        Rethinking the Inception Architecture for Computer Vision
        https://arxiv.org/abs/1512.00567
    """

    def __init__(self, class_map=[], imsize=(299, 299), load_pretrained_weight=False, train_whole_network=True):

        self._model = CNN_InceptionV3(1)

        super(InceptionV3, self).__init__(class_map, imsize, load_pretrained_weight,
                                          train_whole_network, self._model)

        self._model.set_output_size(self.num_class)
        self._model.set_train_whole(train_whole_network)

        self.default_optimizer = OptimizerInception(3)
        self.decay_rate = 0.0005

    def loss(self, x, y):
        return rm.softmax_cross_entropy(x[0], y) + rm.softmax_cross_entropy(x[1], y)

    def predict(self, img_list, batch_size=1):
        self.set_models(inference=True)
        if isinstance(img_list, (list, str)):
            img_builder = self.build_data()
            if isinstance(img_list, (tuple, list)):
                results = []
                bar = tqdm(range(int(np.ceil(len(img_list) / batch_size))))
                for batch_num in range(0,len(img_list),batch_size):
                    results.extend(np.argmax(rm.softmax(self(img_builder(img_path_list=img_list[batch_num:batch_num+batch_size]))[1]).as_ndarray(), axis=1))
                    bar.update(1)
                bar.close()
                return results
            else:
                return np.argmax(rm.softmax(self(img_builder(img_path_list=[img_list]))[1]).as_ndarray(),axis=1)[0]
        else:
            img_array = img_list
        return np.argmax(rm.softmax(self(img_array)[1]).as_ndarray(), axis=1)


    def build_data(self):
        return TargetBuilderInception(self.class_map, self.imsize)

class InceptionV2(Classification):
    """ Inception V2 model

    If the argument load_pretrained_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        class_map: Array of class names
        load_pretrained_weight(bool): True if the pre-trained weight is loaded.
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument num_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    References:
        Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna
        Rethinking the Inception Architecture for Computer Vision
        https://arxiv.org/abs/1512.00567
    """

    def __init__(self, class_map=[], imsize=(299, 299), load_pretrained_weight=False, train_whole_network=True):
        self._model = CNN_InceptionV2(1)

        super(InceptionV2, self).__init__(class_map, imsize, load_pretrained_weight,
                                          train_whole_network, self._model)

        self._model.set_output_size(self.num_class)
        self._model.set_train_whole(train_whole_network)

        self.default_optimizer = OptimizerInception(2)
        self.decay_rate = 0.0005

    def loss(self, x, y):
        return rm.softmax_cross_entropy(x[0], y) + rm.softmax_cross_entropy(x[1], y)


    def predict(self, img_list, batch_size=1):
        self.set_models(inference=True)
        if isinstance(img_list, (list, str)):
            img_builder = self.build_data()
            if isinstance(img_list, (tuple, list)):
                results = []
                bar = tqdm(range(int(np.ceil(len(img_list) / batch_size))))
                for batch_num in range(0,len(img_list),batch_size):
                    results.extend(np.argmax(rm.softmax(self(img_builder(img_path_list=img_list[batch_num:batch_num+batch_size]))[1]).as_ndarray(), axis=1))
                    bar.update(1)
                bar.close()
                return results
            else:
                return np.argmax(rm.softmax(self(img_builder(img_path_list=[img_list]))[1]).as_ndarray(),axis=1)[0]
        else:
            img_array = img_list
        return np.argmax(rm.softmax(self(img_array)[1]).as_ndarray(), axis=1)

    def build_data(self):
        return TargetBuilderInception(self.class_map, self.imsize)

class InceptionV4(Classification):
    """ Inception V4 model

    If the argument load_pretrained_weight is True, pretrained weight will be downloaded.
    The pretrained weight is trained using ILSVRC2012.

    Args:
        class_map: Array of class names
        load_pretrained_weight(bool): True if the pre-trained weight is loaded.
        imsize(int or tuple): Input image size.
        train_whole_network(bool): True if the overal model is trained.

    Note:
        if the argument num_class is not 1000, last dense layer will be reset because
        the pretrained weight is trained on 1000 classification dataset.

    References:
        Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi
        Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning
        https://arxiv.org/abs/1602.07261
    """

    def __init__(self, class_map=[], imsize=(299, 299), load_pretrained_weight=False, train_whole_network=True):

        self._model = CNN_InceptionV4(1)
        super(InceptionV4, self).__init__(class_map, imsize, load_pretrained_weight,
                                          train_whole_network, self._model)

        self._model.set_output_size(self.num_class)
        self._model.set_train_whole(train_whole_network)

        self.default_optimizer = OptimizerInception(4)
        self.decay_rate = 0.0005

    def build_data(self):
        return TargetBuilderInception(self.class_map, self.imsize)


