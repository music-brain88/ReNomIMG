import os
import numpy as np
import renom as rm


class Wrapper(rm.Model):

    def __init__(self, l2reg=0):
        self._wd = l2reg
        self._freezed_model = None
        self._learnable_model = None
        self._optimizer = None

    def load(self, path):
        super(Wrapper, self).load(path)

    def build_target(self, label):
        pass

    def optimizer(self, nth_epoch, nth_batch, total_epoch, total_batch_loop):
        """
        Algorithm specific optimizer.
        """
        raise NotImplemented

    def freezed_forward(self, x):
        """
        Layers that are not learnt.
        """
        return self._freezed_model(x)

    def forward(self, x):
        return self._learnable_model(x)

    def loss_func(self, x, y):
        pass

    def _walk(self):
        for layer in self._learnable_model.iter_models():
            yield layer

    def weight_decay(self):
        if self._wd == 0:
            return 0
        reg = 0
        for layer in self._walk():
            if hasattr(layer, 'params') and layer.params:
                reg += rm.sum(layer.params.w * layer.params.w)
        return reg * self._wd
