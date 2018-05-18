import os
import numpy as np
import renom as rm


class Wrapper(rm.Model):

    def __init__(self, l2reg=0):
        self._wd = l2reg
        self._freezed_model = None
        self._learnable_model = None
        self._optimizer = None

    def build_target(self):
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

    def _walk(self):
        yield self
        for layer in self.iter_models():
            yield layer.iter_models()
  
    def weight_decay(self):
        if self._wd == 0: return 0
        reg = 0
        for layer in self._walk():
            if hasattr(layer, 'params'):
                reg += rm.sum(layer.params.w * layer.params.w)
        return reg
