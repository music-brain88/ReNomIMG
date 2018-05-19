import numpy as np

class Rotate(object):

    def __init__(self):
        pass

    def __call__(self, x):
        return self.transform(x)

    def transform(self, x):
        assert len(x.shape) == 4
        n = x.shape[0]
        new_x = np.empty_like(x)
        rotate_frag = np.random.randint(3, size=(n, ))

        for i, r in enumerate(rotate_frag):
            new_x[i, :, :, :] = np.rot90(x[i], r, axes=(1, 2))

        return new_x

def rotate(x):
    return Rotate()(x)
