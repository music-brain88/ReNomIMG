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
        self.model.set_models(inference=True)
        if isinstance(img_list, (list, str)):
            if isinstance(img_list, (tuple, list)):
                test_dist = ImageDistributor(img_list)
                results = []
                bar = tqdm(range(int(np.ceil(len(test_dist) / batch_size))))
                for i, (x_img_list) in enumerate(test_dist.batch(batch_size, target_builder=self.build_data(), shuffle=False)):
                    if len(img_list) < batch_size:
                        return np.argmax(rm.softmax(self.model(x_img_list)).as_ndarray(), axis=1)[0]
                    results.extend(np.argmax(rm.softmax(self.model(x_img_list)).as_ndarray(), axis=1))
                    bar.update(1)
                bar.close()
                return results
     
        else:
            img_array = img_list
        return np.argmax(rm.softmax(self.model(img_array)).as_ndarray(), axis=1)

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
        b = DataBuilderClassification(self.class_map, self.imsize)

        def builder(img_path_list, annotation_list, augmentation=None, **kwargs):
            imgs, targets = b(img_path_list, annotation_list, augmentation=augmentation, **kwargs)
            return self.preprocess(imgs), targets
        return builder
