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

    def predict(self, img_list, batch_size=1, score_threshold=0.3, nms_threshold=0.4):
        """
        This method accepts either ndarray and list of image path.

        Args:
            img_list (string, list, ndarray): Path to an image, list of path or ndarray.
            score_threshold (float): The threshold for confidence score.
                                     Predicted boxes which have lower confidence score than the threshold are discarderd.
                                     Defaults is 0.3
            nms_threshold (float): The threshold for non maximum supression. Defaults is 0.4

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
        self.set_models(inference=True)
        if isinstance(img_list, (list, str)):
            img_builder = self.build_data()
            if isinstance(img_list, (tuple, list)):
                results = []
                bar = tqdm()
                bar.total = int(np.ceil(len(img_list) / batch_size))
                for batch_num in range(0,len(img_list),batch_size):
                    results.extend(self.get_bbox(self(img_builder(img_path_list=img_list[batch_num:batch_num+batch_size])).as_ndarray(),
                                                     score_threshold,
                                                     nms_threshold))
                    bar.update(1)
                bar.close()
                return results
            else:
                return self.get_bbox(self(img_builder(img_path_list=[img_list])).as_ndarray(),score_threshold,nms_threshold)
        else:
            img_array = img_list
        return self.get_bbox(self(img_array).as_ndarray(),
                             score_threshold,
                             nms_threshold)

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


