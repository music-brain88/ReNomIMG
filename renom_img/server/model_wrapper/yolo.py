
import os
import sys
import numpy as np
import renom as rm
from renom_img.api.detection.yolo import Yolo
from renom_img.api.model.darknet import Darknet

from renom_img.server.model_wrapper.wrapper_base import Wrapper

class WrapperYoloDarknet(Wrapper):

    def __init__(self, num_class, cell, bbox, img_size):
        super(WrapperYoloDarknet, self).__init__(l2reg=0.0005)
        last_dense_size = cell * cell * (5 * bbox + num_class)
        model = Darknet(last_dense_size)
        self._freezed_model = rm.Sequential(model[:-7])
        self._learnable_model = rm.Sequential(model[-7:])
        self._optimizer = rm.Sgd(momentum=0.9)

    def build_target(self, label):
        return

    def optimizer(self, nth_epoch, nth_batch, total_epoch, total_batch_loop):
        lr_list = [0.001] \
            + [0.01] * int(total_epoch * 0.5) \
            + [0.001] * int(total_epoch * 0.25) \
            + [0.0001] * int(total_epoch * 0.25)
        if len(lr_list) < total_epoch:
            lr_list += [0.0001] * (len(total_epoch) - len(lr_list))
        lr_list = lr_list[:total_epoch]

        if epoch == 0:
            lr = nth_batch * ((0.01 - 0.001) /
                               total_batch_loop) + lr_list[nth_epoch]
        else:
            lr = lr_list[nth_epoch]

        self._optimizer._lr = lr
        return self._optimizer

