import abc
import os
import sys
import inspect
import types
import numpy as np
import renom as rm
from tqdm import tqdm

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
                print(cls, parent, m)
                parent_meth = getattr(parent, m.__name__, False)
                if parent_meth and parent_meth.__doc__:
                    if last_add_string != parent_meth.__doc__:
                        add_string += parent_meth.__doc__
                        last_add_string = parent_meth.__doc__
                    else:
                        print(last_add_string)
            if m.__doc__:
                m.__doc__ = add_string + m.__doc__
            else:
                print(add_string)
                m.__doc__ = add_string
    return cls


class Base(rm.Model):
    """Base class of all ReNomIMG algorithm api.
    """

    def get_optimizer(self, current_epoch=None, total_epoch=None, current_batch=None, total_batch=None, **kwargs):
        """
        Returns an instance of Optimizer for training ${class} algorithm.
        If all argument(current_epoch, total_epoch, current_batch, total_batch) are given,
        an optimizer object which whose learning rate is modified according to the
        number of training iteration. Otherwise, constant learning rate is set.

        Args:
            current_epoch (int): The number of current epoch.
            total_epoch (int): The number of total epoch.
            current_batch (int): The number of current batch.
            total_epoch (int): The number of total batch.

        Returns:
            (Optimizer): Optimizer object.
        """
        pass

    def regularize(self):
        """
        L2 Regularization term. You can use this function to add L2 regularization 
        term to a loss function.

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
        return self.decay_rate * reg

    def preprocess(self, x):
        """Performs preprocess for given array.

        Args:
            x(ndarray, Node): Image array for preprocessing.
        """
        pass

    def fit(self, train_img_path_list=None, train_annotation_list=None,
            valid_img_path_list=None, valid_annotation_list=None,
            epoch=136, batch_size=64, augmentation=None, callback_end_epoch=None):
        """
        This function performs training with given data and hyper parameters.

        Following arguments will be given to the function ``callback_end_epoch``.

        - **epoch** (int) - Number of current epoch.
        - **model** (Model) - Yolo1 object.
        - **avg_train_loss_list** (list) - List of average train loss of each epoch.
        - **avg_valid_loss_list** (list) - List of average valid loss of each epoch.

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

        """

        train_dist = ImageDistributor(
            train_img_path_list, train_annotation_list, augmentation=augmentation)
        valid_dist = ImageDistributor(valid_img_path_list, valid_annotation_list)

        batch_loop = int(np.ceil(len(train_dist) / batch_size))
        avg_train_loss_list = []
        avg_valid_loss_list = []
        for e in range(epoch):
            bar = tqdm(range(batch_loop))
            display_loss = 0
            for i, (train_x, train_y) in enumerate(train_dist.batch(batch_size, target_builder=self.build_data())):
                self.set_models(inference=False)
                train_x = self.preprocess(train_x)
                with self.train():
                    loss = self.loss(self(train_x), train_y)
                    reg_loss = loss + self.regularize()
                reg_loss.grad().update(self.get_optimizer(e, epoch, i, batch_loop, avg_valid_loss_list=avg_valid_loss_list))
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
                    valid_x = self.preprocess(valid_x)
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

    def _freeze(self):
        pass

    def forward(self, x):
        """
        Performs forward propagation.
        You can call this function using ``__call__`` method.

        Args:
            x(ndarray, Node): Input to ${class}.
        """
        self._freeze()
        return self._model(x)
