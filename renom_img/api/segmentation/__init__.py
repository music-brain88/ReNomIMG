import os
import sys
import numpy as np
import renom as rm
from tqdm import tqdm

from renom_img.api import adddoc
from renom_img.api.utility.load import load_img
from renom_img.api import Base
from renom_img.api.utility.target import DataBuilderSegmentation
from renom_img.api.utility.distributor.distributor import ImageDistributor

class SemanticSegmentation(Base):

    def predict(self, img_list):
        """
        Returns:
            (Numpy.array or list): If only an image or a path is given, an array whose shape is **(width, height)** is returned.
            If multiple images or paths are given, then a list in which there are arrays whose shape is **(width, height)** is returned.
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
        return rm.softmax_cross_entropy(x, y) / (self.imsize[0] * self.imsize[1])

    def build_data(self):
        return DataBuilderSegmentation(self.class_map, self.imsize)
