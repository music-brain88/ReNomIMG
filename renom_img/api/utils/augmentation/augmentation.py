
import numpy as np
from renom_img.api.utils.augmentation.process import MODE


class Augmentation(object):

    def __init__(self, process_list):
        self._process_list = process_list

    def __call__(self, x, y=None, mode="classification"):
        return self.transform(x, y, mode)

    def transform(self, x, y=None, mode="classification"):
        assert_msg = "{} is not supported transformation mode. {} are available."
        assert mode in MODE, assert_msg.format(mode, MODE)
        for process in self._process_list:
            if np.random.rand() >= 0.9:
                continue
            x, y = process(x, y, mode)
        return x, y
