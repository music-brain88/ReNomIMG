import os
import numpy as np
import renom as rm
from tqdm import tqdm
from renom_img.api.utility.misc.download import download
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.target import DataBuilderClassification

class ClassificationBase(rm.Model):
    def __init__(self, class_map):
        self.class_map = class_map

    def get_optimizer(self, current_epoch=None, total_epoch=None, current_batch=None, total_batch=None):
        pass

    def regularize(self, decay_rate=0.0005):
        reg = 0
        for layer in self.iter_models():
            if hasattr(layer, "params") and hasattr(layer.params, "w"): reg += rm.sum(layer.params.w * layer.params.w)
        return decay_rate * reg

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

    def preprocess(self, x):
        pass

    def loss(self, x, y):
        return rm.softmax_cross_entropy(x, y)

    def fit(self, train_img_path_list=None, train_annotation_list=None, augmentation=None, valid_img_path_list=None, valid_annotation_list=None,  epoch=200, batch_size=16, callback_end_epoch=None):
        pass
