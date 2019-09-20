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

    def predict(self, img_list, batch_size=1, return_scores=False):
        """Perform prediction.
        Argument can be an image array, image path list or a image path.

        Args:
            img_list (ndarray, list, string): Image array, image path list or image path.
            batch_size (int): Batch size for processing input images.
            return_scores (bool): Optional flag to return prediction scores for all classes when set to True. Default is False.

        Return:
            (array): List of predicted class for each image. Also returns array of all probability scores if return_scores is set to True.

        """
        self.set_models(inference=True)
        if isinstance(img_list, (list, str)):
            img_builder = self.build_data()
            if isinstance(img_list, (tuple, list)):
                results = []
                scores = []
                bar = tqdm(range(int(np.ceil(len(img_list) / batch_size))))
                for batch_num in range(0, len(img_list), batch_size):
                    score = rm.softmax(self(img_builder(img_path_list=img_list[batch_num:batch_num + batch_size]))).as_ndarray()
                    if return_scores:
                        scores.extend(score)
                    results.extend(np.argmax(score, axis=1))
                    bar.update(1)
                bar.close()
                if return_scores:
                    return results, scores
                else:
                    return results
            else:
                score = rm.softmax(self(img_builder(img_path_list=[img_list]))).as_ndarray()
                result = np.argmax(score, axis=1)[0]
                if return_scores:
                    return result, score
                else:
                    return result
        else:
            img_array = img_list
            score = rm.softmax(self(img_array)).as_ndarray()
            result = np.argmax(score, axis=1)
            if return_scores:
                return result, score
        return result


    def loss(self, x, y):
        """
        Loss function of ${class} algorithm.

        Args:
            x(ndarray, Node): Output of model.
            y(ndarray, Node): Target array.

        Returns:
            (Node): Loss between x and y.
        Example:
            >>> builder = model.build_data()  # This will return a builder function.
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
