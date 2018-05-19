import numpy as np


class Shift(object):

    def __init__(self, horizontal, vertivcal):
        self._h = horizontal
        self._v = vertivcal

    def __call__(self, x):
        return self.transform(x)

    def transform(self, x):
        """
        Target format must be specified.
        """
        assert len(x.shape) == 4
        n, c, h, w = x.shape
        new_x = np.zeros_like(x)
        rand_h = ((np.random.rand(n) * 2 - 1) * self._h).astype(np.int)
        rand_v = ((np.random.rand(n) * 2 - 1) * self._v).astype(np.int)

        new_min_x = np.clip(rand_h, 0, w)
        new_min_y = np.clip(rand_v, 0, h)
        new_max_x = np.clip(rand_h + w, 0, w)
        new_max_y = np.clip(rand_v + h, 0, h)

        orig_min_x = np.maximum(-rand_h, 0)
        orig_min_y = np.maximum(-rand_v, 0)
        orig_max_x = np.minimum(-rand_h + w, w)
        orig_max_y = np.minimum(-rand_v + h, h)

        for i in range(n):
            new_x[i, :, new_min_y[i]:new_max_y[i], new_min_x[i]:new_max_x[i]] = \
                x[i, :, orig_min_y[i]:orig_max_y[i], orig_min_x[i]:orig_max_x[i]]
        return new_x


def shift(x, horizontal, vertivcal):
    return Shift(horizontal, vertivcal)(x)
