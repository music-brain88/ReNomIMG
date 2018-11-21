import os
import sys
import numpy as np
import renom as rm
from tqdm import tqdm
from renom_img.api.utility.load import load_img

from renom_img.api import Base
from renom_img.api.utility.target import DataBuilderClassification
from renom_img.api.utility.distributor.distributor import ImageDistributor


class Classification(Base):

    def preprocess(self, x):
        return x / 255.

    def predict(self, img_list, batch_size=1):
        """Perform prediction.
        Argument can be an image array, image path list or a image path.

        Args:
            img_list(ndarray, list, string): Image array, image path list or image path.

        Return:
            (list): List of class of each image.

        """
        self.set_models(inference=True)
        if isinstance(img_list, (list, str)):
            if isinstance(img_list, (tuple, list)):
                if len(img_list) > batch_size:
                    test_dist = ImageDistributor(img_list)
                    results = []
                    bar = tqdm(range(int(np.ceil(len(test_dist) / batch_size))))
                    for i, (x_img_list, _) in enumerate(test_dist.batch(batch_size, shuffle=False)):
                        img_array = np.vstack([load_img(path, self.imsize)[None]
                                               for path in x_img_list])
                        img_array = self.preprocess(img_array)
                        results.extend(np.argmax(rm.softmax(self(img_array)).as_ndarray(), axis=1))
                        bar.update(1)
                    return results
                img_array = np.vstack([load_img(path, self.imsize)[None] for path in img_list])
                img_array = self.preprocess(img_array)
            else:
                img_array = load_img(img_list, self.imsize)[None]
                img_array = self.preprocess(img_array)
                return np.argmax(rm.softmax(self(img_array)).as_ndarray(), axis=1)[0]
        else:
            img_array = img_list
        return np.argmax(rm.softmax(self(img_array)).as_ndarray(), axis=1)

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
        return rm.softmax_cross_entropy(x, y)

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
        return DataBuilderClassification(self.class_map, self.imsize)
