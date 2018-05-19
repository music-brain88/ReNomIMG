import numpy as np


class Flip(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return self.transform(x)

    def transform(self, x):
        assert len(x.shape) == 4
        n = x.shape[0]
        new_x = np.empty_like(x)
        flip_flag = np.random.randint(3, size=(n, ))
        for i, f in enumerate(flip_flag):
            if f == 0:
                new_x[i, :, :, :] = x[i, :, :, :]
            elif f == 1:
                new_x[i, :, :, :] = x[i, :, :, ::-1]
            elif f == 2:
                new_x[i, :, :, :] = x[i, :, ::-1, :]
        return new_x


def flip(x):
    return Flip()(x)
