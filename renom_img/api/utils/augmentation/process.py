import numpy as np

MODE = [
    "classification",
    "detection",
    "segmentation"
]

class ProcessBase(object):
    """
    X and Y must be resized as specified img size.
    """

    def __init__(self):
        pass

    def __call__(self, x, y=None, mode="classification"):
        return self.transform(x, y, mode)

    def transform(self, x, y=None, mode="classification"):
        assert_msg = "{} is not supported transformation mode. {} are available."
        assert mode in MODE, assert_msg.format(mode, MODE)

        if mode == MODE[0]:
            # Returns only x.
            return self._transform_classification(x, y)
        elif mode == MODE[1]:
            return self._transform_detection(x, y)
        elif mode == MODE[2]:
            return self._transform_segmentation(x, y)

    def _transform_classification(self, x, y):
        raise NotImplemented

    def _transform_detection(self, x, y):
        raise NotImplemented

    def _transform_segmentation(self, x, y):
        raise NotImplemented


class Flip(ProcessBase):

    def __init__(self):
        super(Flip, self).__init__()

    def _transform_classification(self, x, y):
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
        return new_x, y

    def _transform_detection(self, x, y):
        assert len(x.shape) == 4
        assert len(y.shape) == 2
        n = x.shape[0]
        new_x = np.empty_like(x)
        new_y = np.empty_like(y)
        flip_flag = np.random.randint(3, size=(n, ))
        for i, f in enumerate(flip_flag):
            if f == 0:
                new_x[i, :, :, :] = x[i, :, :, :]
                new_y[i, :] = y[i, :]
            elif f == 1:
                # Horizontal flip.
                c_x = x.shape[3] // 2
                new_x[i, :, :, :] = x[i, :, :, ::-1]
                new_y[i, 0::5] = (2 * c_x - y[i, 0::5]) * (y[i, 2::5] != 0)  # X
                new_y[i, 1::5] = y[i, 1::5]  # Y
                new_y[i, 2::5] = y[i, 2::5]  # W
                new_y[i, 3::5] = y[i, 3::5]  # H
                new_y[i, 4::5] = y[i, 4::5]  # C
            elif f == 2:
                c_y = x.shape[2] // 2
                new_x[i, :, :, :] = x[i, :, ::-1, :]
                new_y[i, 0::5] = y[i, 0::5]  # X
                new_y[i, 1::5] = (2 * c_y - y[i, 1::5]) * (y[i, 3::5] != 0)  # Y
                new_y[i, 2::5] = y[i, 2::5]  # W
                new_y[i, 3::5] = y[i, 3::5]  # H
                new_y[i, 4::5] = y[i, 4::5]  # C
        return new_x, new_y

    def _transform_segmentation(self, x, y):
        assert len(x.shape) == 4
        n = x.shape[0]
        new_x = np.empty_like(x)
        new_y = np.empty_like(y)
        flip_flag = np.random.randint(3, size=(n, ))
        for i, f in enumerate(flip_flag):
            if f == 0:
                new_x[i, :, :, :] = x[i, :, :, :]
                new_y[i, :, :, :] = y[i, :, :, :]
            elif f == 1:
                new_x[i, :, :, :] = x[i, :, :, ::-1]
                new_y[i, :, :, :] = y[i, :, :, ::-1]
            elif f == 2:
                new_x[i, :, :, :] = x[i, :, ::-1, :]
                new_y[i, :, :, :] = y[i, :, ::-1, :]
        return new_x, new_y


def flip(x, y=None, mode="classification"):
    return Flip()(x, y, mode)


class Shift(ProcessBase):

    def __init__(self, horizontal=10, vertivcal=10):
        super(Shift, self).__init__()
        self._h = horizontal
        self._v = vertivcal

    def _transform_classification(self, x, y):
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
        return new_x, y

    def _transform_detection(self, x, y):
        assert len(x.shape) == 4
        assert len(y.shape) == 2
        n, c, h, w = x.shape
        new_x = np.zeros_like(x)
        new_y = np.zeros_like(y)
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
        flag = y[:, 2::5] != 0
        new_y[:, 0::5] = np.clip(y[:, 0::5] + rand_h[:, None], 0, w) * flag
        new_y[:, 1::5] = np.clip(y[:, 1::5] + rand_v[:, None], 0, h) * flag
        new_y[:, 2::5] = y[:, 2::5]
        new_y[:, 3::5] = y[:, 3::5]
        new_y[:, 4::5] = y[:, 4::5]
        return new_x, new_y


def shift(x, y=None, horizontal=10, vertivcal=10, mode="classification"):
    return Shift(horizontal, vertivcal)(x)


class Rotate(ProcessBase):

    def __init__(self):
        super(Rotate, self).__init__()

    def _transform_classification(self, x, y):
        assert len(x.shape) == 4
        n, c, h, w = x.shape
        new_x = np.empty_like(x)

        if h==w:
            # 0, 90, 180 or 270 degree.
            rotate_frag = np.random.randint(4, size=(n, ))
        else:
            # 0 or 180 degree.
            rotate_frag = np.random.randint(2, size=(n, ))*2

        for i, r in enumerate(rotate_frag):
            new_x[i, :, :, :] = np.rot90(x[i], r, axes=(1, 2))
        return new_x, y

    def _transform_detection(self, x, y):
        assert len(x.shape) == 4
        n, c, h, w = x.shape
        c_w = w // 2
        c_h = h // 2
        new_x = np.empty_like(x)
        new_y = np.empty_like(y)

        if w == h:
            rotate_frag = np.random.randint(4, size=(n, ))
        else:
            rotate_frag = np.random.randint(2, size=(n, )) * 2

        for i, r in enumerate(rotate_frag):
            new_x[i, :, :, :] = np.rot90(x[i], r, axes=(1, 2))
            flag = y[i, 2::5] != 0
            if r == 0:
                new_y[i, 0::5] = y[i, 0::5]
                new_y[i, 1::5] = y[i, 1::5]
                new_y[i, 2::5] = y[i, 2::5]
                new_y[i, 3::5] = y[i, 3::5]
            elif r == 1:
                new_y[i, 0::5] = y[i, 1::5]
                new_y[i, 1::5] = (2 * c_h - y[i, 0::5]) * flag
                new_y[i, 2::5] = y[i, 3::5]
                new_y[i, 3::5] = y[i, 2::5]
            elif r == 2:
                new_y[i, 0::5] = (2 * c_w - y[i, 0::5]) * flag
                new_y[i, 1::5] = (2 * c_h - y[i, 1::5]) * flag
                new_y[i, 2::5] = y[i, 2::5]
                new_y[i, 3::5] = y[i, 3::5]
            elif r == 3:
                new_y[i, 0::5] = (2 * c_w - y[i, 1::5]) * flag
                new_y[i, 1::5] = y[i, 0::5]
                new_y[i, 2::5] = y[i, 3::5]
                new_y[i, 3::5] = y[i, 2::5]
            new_y[i, 4::5] = y[i, 4::5]
        return new_x, new_y


def rotate(x, y=None, mode="classification"):
    return Rotate()(x, y, mode)


class WhiteNoise(ProcessBase):

    def __init__(self, std=0.01):
        super(WhiteNoise, self).__init__()
        self._std = std

    def _transform_classification(self, x, y):
        assert len(x.shape) == 4
        return x + self._std * np.random.randn(*x.shape), y

    def _transform_detection(self, x, y):
        assert len(x.shape) == 4
        return x + self._std * np.random.randn(*x.shape), y


def white_noise(x, y=None, std=0.01, mode="classification"):
    return WhiteNoise(std)(x, y, mode)


class Jitter(ProcessBase):

    def __init__(self):
        pass

    def _transform_classification(self, x, y):
        """
        Color is "RGB" 0~255 image.
        """
        raise NotImplementedError
        N = len(x)
        dim0 = np.arange(N)
        scale_h = np.random.uniform(0.9, 1.1)
        scale_s = np.random.uniform(0.9, 1.1)
        scale_v = np.random.uniform(0.9, 1.1)
        new_x = np.zeros_like(x)
        max_color_index = np.argmax(x, axis=1)
        max_color = x[(dim0, max_color_index)]
        min_color = np.min(x, axis=1)
        h = (x[(dim0, max_color_index - 2)] - x[(dim0, max_color_index - 1)]) / \
            (max_color - min_color) * 60 + max_color_index * 120
        s = (max_color - min_color) / max_color
        v = max_color

        h = np.clip(h * scale_h, 0, 359)
        s = np.clip(s * scale_s, 0, 1)
        v = np.clip(v * scale_v, 0, 255)
        return x, y
