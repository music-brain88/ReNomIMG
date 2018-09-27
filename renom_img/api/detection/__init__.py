import os
import sys
import numpy as np
import renom as rm
from tqdm import tqdm
from renom_img.api.utility.load import load_img

from renom_img.api import Base
from renom_img.api.utility.target import DataBuilderClassification
from renom_img.api.utility.distributor.distributor import ImageDistributor


class Detection(Base):

    def preprocess(self, x):
        return x / 255.

    def predict(self, img_list, score_threshold=0.3, nms_threshold=0.4):
        """
        This method accepts either ndarray and list of image path.

        Args:
            img_list (string, list, ndarray): Path to an image, list of path or ndarray.
            score_threshold (float): The threshold for confidence score.
                                     Predicted boxes which have lower confidence score than the threshold are discarderd.
                                     Defaults to 0.3
            nms_threshold (float): The threshold for non maximum supression. Defaults to 0.4

        Return:
            (list): List of predicted bbox, score and class of each image.
            The format of return value is bellow. Box coordinates and size will be returned as
            ratio to the original image size. Therefore the range of 'box' is [0 ~ 1].

        .. code-block :: python

            # An example of return value.
            [
                [ # Prediction of first image.
                    {'box': [x, y, w, h], 'score':(float), 'class':(int), 'name':(str)},
                    {'box': [x, y, w, h], 'score':(float), 'class':(int), 'name':(str)},
                    ...
                ],
                [ # Prediction of second image.
                    {'box': [x, y, w, h], 'score':(float), 'class':(int), 'name':(str)},
                    {'box': [x, y, w, h], 'score':(float), 'class':(int), 'name':(str)},
                    ...
                ],
                ...
            ]

        Example:
            >>>
            >>> model.predict(['img01.jpg', 'img02.jpg']])
            [[{'box': [0.21, 0.44, 0.11, 0.32], 'score':0.823, 'class':1, 'name':'dog'}],
             [{'box': [0.87, 0.38, 0.84, 0.22], 'score':0.423, 'class':0, 'name':'cat'}]]

        Note:
            Box coordinate and size will be returned as ratio to the original image size.
            Therefore the range of 'box' is [0 ~ 1].

        """
        batch_size = 32
        self.set_models(inference=True)
        if isinstance(img_list, (list, str)):
            if isinstance(img_list, (tuple, list)):
                if len(img_list) >= 32:
                    test_dist = ImageDistributor(img_list)
                    results = []
                    bar = tqdm()
                    bar.total = int(np.ceil(len(test_dist) / batch_size))
                    for i, (x_img_list, _) in enumerate(test_dist.batch(batch_size, shuffle=False)):
                        img_array = np.vstack([load_img(path, self.imsize)[None]
                                               for path in x_img_list])
                        img_array = self.preprocess(img_array)
                        results.extend(self.get_bbox(self(img_array).as_ndarray(),
                                                     score_threshold,
                                                     nms_threshold))
                        bar.update(1)
                    return results
                img_array = np.vstack([load_img(path, self.imsize)[None] for path in img_list])
                img_array = self.preprocess(img_array)
            else:
                img_array = load_img(img_list, self.imsize)[None]
                img_array = self.preprocess(img_array)
                return self.get_bbox(self(img_array).as_ndarray(),
                                     score_threshold,
                                     nms_threshold)[0]
        else:
            img_array = img_list
        return self.get_bbox(self(img_array).as_ndarray(),
                             score_threshold,
                             nms_threshold)

    @property
    def freezed_network(self):
        return self._freezed_network

    @property
    def network(self):
        return self._network


    def loss(self, x, y):
        """
        Loss function of ${class} algorithm.

        Args:
            x(ndarray, Node): Output of model.
            y(ndarray, Node): Target array.

        Returns:
            (Node): Loss between x and y.
        Example:
            >>> builder = model.build_data()  # This will return function.
            >>> x, y = builder(image_path_list, annotation_list)
            >>> z = model(x)
            >>> loss = model.loss(z, y)
        """
        raise NotImplementedError

    def build_data(self):
        """
        This function returns a function which creates input data and target data
        specified for ${class}.

        Returns:
            (function): Returns function which creates input data and target data.

        Example:
            >>> builder = model.build_data()  # This will return function.
            >>> x, y = builder(image_path_list, annotation_list)
            >>> z = model(x)
            >>> loss = model.loss(z, y)
        """
        raise NotImplementedError

    def fit(self, train_img_path_list, train_annotation_list,
            valid_img_path_list=None, valid_annotation_list=None,
            epoch=136, batch_size=64, augmentation=None, callback_end_epoch=None):
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
            >>> from renom_img.api.detection.yolo_v1 import Yolov1
            >>> train_img_path_list, train_annot_list = ... # Define own data.
            >>> valid_img_path_list, valid_annot_list = ...
            >>> model = Yolov1()
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
        - **model** (Model) - Yolo1 object.
        - **avg_train_loss_list** (list) - List of average train loss of each epoch.
        - **avg_valid_loss_list** (list) - List of average valid loss of each epoch.

        """

        train_dist = ImageDistributor(
            train_img_path_list, train_annotation_list, augmentation=augmentation)
        if valid_img_path_list is not None and valid_annotation_list is not None:
            valid_dist = ImageDistributor(valid_img_path_list, valid_annotation_list)
        else:
            valid_dist = None

        batch_loop = int(np.ceil(len(train_dist) / batch_size))
        avg_train_loss_list = []
        avg_valid_loss_list = []
        for e in range(epoch):
            bar = tqdm(range(batch_loop))
            display_loss = 0
            for i, (train_x, train_y) in enumerate(train_dist.batch(batch_size, target_builder=self.build_data())):
                self.set_models(inference=False)
                with self.train():
                    loss = self.loss(self(train_x), train_y)
                    reg_loss = loss + self.regularize()
                reg_loss.grad().update(self.get_optimizer(e, epoch, i, batch_loop))
                try:
                    loss = loss.as_ndarray()[0]
                except:
                    loss = loss.as_ndarray()
                loss = float(loss)
                display_loss += loss
                bar.set_description("Epoch:{:03d} Train Loss:{:5.3f}".format(e, loss))
                bar.update(1)
            avg_train_loss = display_loss / (i + 1)
            avg_train_loss_list.append(avg_train_loss)

            if valid_dist is not None:
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
                    loss = float(loss)
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
