import numpy as np


class WhiteNoise(object):

    def __init__(self, std=0.01):
        self._std = std

    def __call__(self, x):
        return self.transform(x)

    def transform(self, x):
        assert len(x.shape) == 4
        return x + self._std * np.random.randn(*x.shape)


def white_noise(x, std=0.01):
    return WhiteNoise(std)(x)
