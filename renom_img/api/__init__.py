import abc
import os
import sys
import inspect
import types
import numpy as np
import renom as rm
from tqdm import tqdm

from renom_img.api.utility.optimizer import BaseOptimizer
from renom_img.api.utility.misc.download import download
from renom_img.api.utility.distributor.distributor import ImageDistributor


def adddoc(cls):
    """Insert parent doc strings to inherited class.
    """
    for m in cls.__dict__.values():
        if isinstance(m, types.FunctionType):
            parent_list = cls.mro()
            first = 1
            end = parent_list.index(Base) + 1
            add_string = ""
            last_add_string = None
            for parent in parent_list[first:end][::-1]:
                parent_meth = getattr(parent, m.__name__, False)
                if parent_meth and parent_meth.__doc__:
                    if last_add_string != parent_meth.__doc__:
                        add_string += parent_meth.__doc__
                        last_add_string = parent_meth.__doc__
            if m.__doc__:
                m.__doc__ = add_string + m.__doc__
            else:
                m.__doc__ = add_string
    return cls


class Base(rm.Model):
    """Base class of all ReNomIMG algorithm api.
    """

    SERIALIZED = ("imsize", "class_map", "num_class")
    WEIGHT_URL = None

    def __init__(self, class_map=None, imsize=(224, 224),
                 load_pretrained_weight=False, train_whole_network=False, load_target=None):

        # 0. General setting.
        self.default_optimizer = rm.Sgd(0.001, 0.9)

        # 1. Cast class_map to list and encodes the class names to ascii.
        if class_map is None:
            self.class_map = []
        else:
            if isinstance(class_map, list):
                self.class_map = [c.encode("ascii", "ignore") for i, c in enumerate(class_map)]
            elif isinstance(class_map, dict):
                self.class_map = [k.encode("ascii", "ignore") for k, v in class_map.items()]

        # Determines last layer's unit size according to the class number.
        self.num_class = len(self.class_map)

        # 2. Accepts imsize both tuple and int.
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)
        self.imsize = imsize

        # Train whole or not.
        self.train_whole_network = train_whole_network

        # 4. Load pretrained weight.
        self.load_pretrained_weight = load_pretrained_weight
        if load_pretrained_weight and load_target is not None:
            assert self.WEIGHT_URL, \
                "The class '{}' has no pretrained weight.".format(self.__class__.__name__)

            if isinstance(load_pretrained_weight, bool):
                weight_path = self.__class__.__name__ + '.h5'
            elif isinstance(load_pretrained_weight, str):
                weight_path = load_pretrained_weight

            if not os.path.exists(weight_path):
                download(self.WEIGHT_URL, weight_path)
            load_target.load(weight_path)

    def regularize(self):
        """
        Regularization term to a loss function.

        Example:
            >>> x = numpu.random.rand(1, 3, 224, 224)
            >>> y = numpy.random.rand(1, (5*2+20)*7*7)
            >>> model = ${class}()
            >>> loss = model.loss(x, y)
            >>> reg_loss = loss + model.regularize() # Add weight decay term.
        """
        reg = 0
        for layer in self.iter_models():
            if hasattr(layer, "params") and hasattr(layer.params, "w"):
                reg += rm.sum(layer.params.w * layer.params.w)
        return (self.decay_rate / 2) * reg

    def preprocess(self, x):
        """Performs preprocess for a given array.

        Args:
            x(ndarray, Node): Image array for preprocessing.
        """
        return x

    def forward(self, x):
        assert len(self.class_map) > 0, \
            "Class map is empty. Please set the attribute class_map when instantiate model class. " +\
            "Or, please load already trained model using the method 'load()'."
        return self.model(x)

    def fit(self, train_img_path_list=None, train_annotation_list=None,
            valid_img_path_list=None, valid_annotation_list=None,
            epoch=136, batch_size=64, optimizer=None, augmentation=None, callback_end_epoch=None):
        """
        This function performs training with given data and hyper parameters.

        Args:
            train_img_path_list(list): List of image path.
            train_annotation_list(list): List of annotations.
            valid_img_path_list(list): List of image path for validation.
            valid_annotation_list(list): List of annotations for validation.
            epoch(int): Number of training epoch.
            batch_size(int): Number of batch size.
            augmentation(Augmentation): Augmentation object.
            callback_end_epoch(function): Given function will be called at the end of each epoch.

        Returns:
            (tuple): Training loss list and validation loss list.

        Example:
            >>> train_img_path_list, train_annot_list = ... # Define own data.
            >>> valid_img_path_list, valid_annot_list = ...
            >>> model = ${class}() # Any algorithm which provided by ReNomIMG here.
            >>> model.fit(
            ...     # Feeds image and annotation data.
            ...     train_img_path_list,
            ...     train_annot_list,
            ...     valid_img_path_list,
            ...     valid_annot_list,
            ...     epoch=8,
            ...     batch_size=8)
            >>> 

        Following arguments will be given to the function ``callback_end_epoch``.

        - **epoch** (int) - Number of current epoch.
        - **model** (Model) - Model object.
        - **avg_train_loss_list** (list) - List of average train loss of each epoch.
        - **avg_valid_loss_list** (list) - List of average valid loss of each epoch.

        """

        # Train Logs.
        avg_train_loss_list = []
        avg_valid_loss_list = []

        # Distributor setting.
        train_dist = ImageDistributor(
            train_img_path_list, train_annotation_list, augmentation=augmentation)
        valid_dist = ImageDistributor(valid_img_path_list, valid_annotation_list)

        # Number of batch iteration.
        batch_loop = int(np.ceil(len(train_dist) / batch_size))

        # Optimizer settings.
        if optimizer is None:
            opt = self.default_optimizer
        else:
            opt = optimizer
        assert opt is not None
        if isinstance(opt, BaseOptimizer):
            opt.setup(batch_loop, epoch)

        # Training loop.
        for e in range(epoch):
            bar = tqdm(range(batch_loop))
            display_loss = 0

            # Batch loop.
            for i, (train_x, train_y) in enumerate(train_dist.batch(batch_size, target_builder=self.build_data())):
                self.set_models(inference=False)

                # Modify optimizer.
                if isinstance(opt, BaseOptimizer):
                    opt.set_information(i, e, avg_train_loss_list, avg_valid_loss_list)

                # Gradient descent.
                with self.train():
                    loss = self.loss(self(train_x), train_y)
                    reg_loss = loss + self.regularize()
                reg_loss.grad().update(opt)
                try:
                    loss = loss.as_ndarray()[0]
                except:
                    loss = loss.as_ndarray()

                display_loss += loss
                bar.set_description("Epoch:{:03d} Train Loss:{:5.3f}".format(e, loss))
                bar.update(1)
            avg_train_loss = display_loss / (i + 1)
            avg_train_loss_list.append(avg_train_loss)

            if valid_img_path_list is not None:
                bar.n = 0
                bar.total = int(np.ceil(len(valid_dist) / batch_size))
                display_loss = 0
                for i, (valid_x, valid_y) in enumerate(valid_dist.batch(batch_size, target_builder=self.build_data())):
                    self.set_models(inference=True)
                    loss = self.loss(self(valid_x), valid_y)
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

    def predict(self, img_list):
        """Perform prediction.
        Argument can be an image array, image path list or a image path.
        The form of return value depends on your task(classification, detection or segmentation).

        Args:
            img_list(ndarray, list, string): Image array, image path list or image path.

        """
        pass

    def loss(self, x, y):
        """
        Loss function of ${class} algorithm.

        Args:
            x(ndarray, Node): Output of model.
            y(ndarray, Node): Target array.

        Returns:
            (Node): Loss between x and y.

        """
        pass
