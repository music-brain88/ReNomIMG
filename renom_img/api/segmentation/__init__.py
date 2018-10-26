import os
import sys
import numpy as np
import renom as rm
from tqdm import tqdm
from collections import defaultdict
from PIL import Image

from renom_img.api import adddoc
from renom_img.api.utility.load import load_img
from renom_img.api import Base
from renom_img.api.utility.target import DataBuilderSegmentation
from renom_img.api.utility.distributor.distributor import ImageDistributor

# LSVRC2012 used by VGG16
MEAN_BGR = np.array([104.00698793, 116.66876762, 122.67891434])

class SemanticSegmentation(Base):

    def get_preprocessed_data(self, img_list, index):
        img_file = img_list[index]
        img = Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        img = img[:, :, ::-1].astype(np.float32) # RGB -> BGR
        img -= MEAN_BGR
        img = img.transpose(2, 0, 1)
        return img

    def get_label(self, img_list, index):
        lbl_file = img_list[index]
        lbl = Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl==255] = -1
        return lbl

    def get_unique_label(self, lbl_list):
        uniq_label = []
        for i in tqdm.trange(len(lbl_list)):
            lbl_file = lbl_list[i]
            lbl = Image.open(lbl_file)
            for l in set(lbl.flatten()):
                if not l in uniq_label:
                    uniq_label.append(l)
        return uniq_label

    def predict(self, img_list, batch_size=1):
        """
        Returns:
            (Numpy.array or list): If only an image or a path is given, an array whose shape is **(width, height)** is returned.
            If multiple images or paths are given, then a list in which there are arrays whose shape is **(width, height)** is returned.
        """

        self.set_models(inference=True)
        if isinstance(img_list, (list, str)):
            if isinstance(img_list, (tuple, list)):
                if len(img_list) > batch_size:
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

    def fit(self, train_img_path_list=None, train_annotation_list=None,
            valid_img_path_list=None, valid_annotation_list=None,
            epoch=136, batch_size=64, augmentation=None, callback_end_epoch=None, class_weight=None):

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
                    loss = self.loss(self(train_x), train_y, class_weight=class_weight)
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
                    loss = self.loss(self(valid_x), valid_y, class_weight=class_weight)
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

    def loss(self, x, y, class_weight=None):
        if class_weight is not None:
            mask = np.array(class_weight)[y.astype(np.int)]
            loss = rm.softmax_cross_entropy(x, y, reduce_sum=False)
            loss *= mask
            loss = rm.sum(loss) / (self.imsize[0] * self.imsize[1])
        else:
            loss = rm.softmax_cross_entropy(x, y)
        return loss / (self.imsize[0] * self.imsize[1])

    def build_data(self):
        return DataBuilderSegmentation(self.class_map, self.imsize)

    def build_class_weight(self, annotation_path_list):
        counter = defaultdict(int)
        N = len(annotation_path_list)
        bar = tqdm(range(N))
        for annot_path in annotation_path_list:
            annot = Image.open(annot_path)
            annot.load()
            annot = annot.resize(self.imsize, Image.BILINEAR)
            annot = np.array(annot)
            for i in range(annot.shape[0]):
                for j in range(annot.shape[1]):
                    if int(annot[i, j]) >= self.num_class:
                        counter[0] += 1
                    else:
                        counter[int(annot[i, j])] += 1
            bar.update(1)
        bar.close()

        total = np.sum(list(counter.values()))

        class_weight = {}
        for key in counter.keys():
            class_weight[key] = float(total) / (self.num_class * counter[key])
        class_weight = [v for k, v in sorted(class_weight.items(), key=lambda x: x[0])]
        return class_weight
