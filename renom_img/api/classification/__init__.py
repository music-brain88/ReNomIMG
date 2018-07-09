import os, sys
import numpy as np
import renom as rm
from tqdm import tqdm

from renom_img.api import Base
from renom_img.api.utility.target import DataBuilderClassification

class Classification(Base):
    def get_optimizer(self, current_epoch=None, total_epoch=None, current_batch=None, total_batch=None, **kwargs):
        pass

    def preprocess(self, x):
        return x/255.

    def predict(self, img_list):
        self.set_models(inference=True)
        if isinstance(img_list, (list, str)):
            if isinstance(img_list, (tuple, list)):
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
        return rm.softmax_cross_entropy(x, y)

    def build_data(self):
        return DataBuilderClassification(self.imsize, self.class_map)
