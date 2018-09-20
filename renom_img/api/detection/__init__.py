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

    def predict(self, img_list):
        """Perform prediction.
        Argument can be an image array, image path list or a image path.

        Args:
            img_list(ndarray, list, string): Image array, image path list or image path.

        Return:
            (list): List of class of each image.

        """
        self.set_models(inference=True)

        batch_size = 8
        if isinstance(img_list, (list, str)):
            if isinstance(img_list, (tuple, list)):
                if len(img_list) > batch_size:
                    results = []
                    L = int(np.ceil(len(test_dist) / batch_size)) 
                    bar = tqdm(range(L))
                    for i in range(L):
                        x = self.preprocess(img_array[i*batch_size:(i+1)*batch_size])
                        results.extend(self.get_bbox(self(x).as_ndarray()))
                        bar.update(1)
                    return results
                else:
                    x = self.preprocess(img_array)
                    return self.get_bbox(self(x))
            else:
                img_array = load_img(img_list, self.imsize)[None]
                img_array = self.preprocess(img_array)
                return self.get_bbox(self(img_array).as_ndarray())[0]
        else:
            img_array = img_list
            return self.get_bbox(self(img_array).as_ndarray())

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
