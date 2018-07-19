import os
import sys
import numpy as np
import renom as rm
from tqdm import tqdm

from renom_img.api.utility.distributor.distributor import ImageDistributor


class Base(rm.Model):

    def get_optimizer(self, current_epoch=None, total_epoch=None, current_batch=None, total_batch=None, **kwargs):
        pass

    def regularize(self):
        """L2 Regularization term. You can use this function to add L2 regularization term to a loss function.

        Example:
            >>> import numpy as np
            >>> from renom_img.api.model.vgg import VGG16
            >>> x = np.random.rand(1, 3, 224, 224)
            >>> y = np.random.rand(1, (5*2+20)*7*7)
            >>> model = VGG16()
            >>> loss = model.loss(x, y)
            >>> reg_loss = loss + model.regularize() # Add weight decay term.

        """
        reg = 0
        for layer in self.iter_models():
            if hasattr(layer, "params") and hasattr(layer.params, "w"):
                reg += rm.sum(layer.params.w * layer.params.w)
        return self.decay_rate * reg

    def preprocess(self, x):
        pass

    def fit(self, train_img_path_list=None, train_annotation_list=None,
            valid_img_path_list=None, valid_annotation_list=None,
            epoch=136, batch_size=64, augmentation=None, callback_end_epoch=None):

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
                with self.train():
                    loss = self.loss(self(train_x), train_y)
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
                    loss = self.loss(self(train_x), train_y)
                    try:
                        loss = loss.as_ndarray()[0]
                    except:
                        loss = loss.as_ndarray()
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

    def predict(self, img_list):
        pass

    def loss(self, x, y):
        pass

    def _freeze(self):
        pass

    def forward(self, x):
        self._freeze()
        return self._model(x)
