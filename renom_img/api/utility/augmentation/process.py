import numpy as np

MODE = [
    "classification",
    "detection",
    "segmentation"
]


class ProcessBase(object):
    """Base class for applying augmentation to images.

    Note:
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
        """Format must be 1d array or list of integer."""
        raise NotImplemented

    def _transform_detection(self, x, y):
        """Format must be list of dictionary"""
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
        n = x.shape[0]
        new_x = np.empty_like(x)
        new_y = []
        flip_flag = np.random.randint(3, size=(n, ))

        for i, f in enumerate(flip_flag):
            if f == 0:
                new_x[i, :, :, :] = x[i, :, :, :]
                new_y.append(y[i])
            elif f == 1:
                # Horizontal flip.
                c_x = x.shape[3] // 2
                new_x[i, :, :, :] = x[i, :, :, ::-1]
                new_y.append([
                    {
                        "box": [
                            (2 * c_x - obj["box"][0]),
                            obj["box"][1],
                            obj["box"][2],
                            obj["box"][3],
                        ],
                        "name":obj["name"],
                        "class":obj["class"],
                    }
                    for j, obj in enumerate(y[i])])

            elif f == 2:
                c_y = x.shape[2] // 2
                new_x[i, :, :, :] = x[i, :, ::-1, :]
                new_y.append([
                    {
                        "box": [
                            obj["box"][0],
                            (2 * c_y - obj["box"][1]),
                            obj["box"][2],
                            obj["box"][3],
                        ],
                        "name":obj["name"],
                        "class":obj["class"],
                    }
                    for j, obj in enumerate(y[i])])
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
    """Flip image randomly.
    Args:
        x(list of numpy.array): List of images.
        y(list of dict): List of annotation results.
        mode(str): Type of task. You could chooose Classification, Detection or Segmentation.

    Returns:
        (list of numpy.array): List of flipped images.
        (list of dict): List of annotation results.
    Examples:
        >>> from renom_img.api.utility.augmentation.process import Flip
        >>> from PIL import Image
        >>>
        >>> img1 = Image.open(img_path1)
        >>> img2 = Image.open(img_path2)
        >>> img_list = np.array([img1, img2])
        >>> flipped_img = flip(img_list)
    """
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
        n, c, h, w = x.shape
        new_x = np.zeros_like(x)
        new_y = []
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
            ny = []
            for j, obj in enumerate(y[i]):
                pw = obj["box"][2]
                ph = obj["box"][3]
                px1 = np.clip(obj["box"][0] - pw / 2. + rand_h[i], 0, w - 1)
                py1 = np.clip(obj["box"][1] - ph / 2. + rand_v[i], 0, h - 1)
                px2 = np.clip(obj["box"][0] + pw / 2. + rand_h[i], 0, w - 1)
                py2 = np.clip(obj["box"][1] + ph / 2. + rand_v[i], 0, h - 1)
                pw = px2 - px1
                ph = py2 - py1
                px = px1 + pw / 2.
                py = py1 + ph / 2.
                ny.append({
                    "box": [px, py, pw, ph],
                    "class": obj["class"],
                    "name": obj["name"]
                })
            new_y.append(ny)
        return new_x, new_y


def shift(x, y=None, horizontal=10, vertivcal=10, mode="classification"):
    """Shift images randomly according to given parameter.
    Args:
        x(list of numpy.array): List of images.
        y(list of dict): List of annotation results.
        mode(str): Type of task. You could chooose Classification, Detection or Segmentation.

    Returns:
        (list of numpy.array): List of shifted images.
        (list of dict): List of annotation results.

    Examples:
        >>> from renom_img.api.utility.augmentation.process import shift
        >>> from PIL import Image
        >>>
        >>> img1 = Image.open(img_path1)
        >>> img2 = Image.open(img_path2)
        >>> img_list = np.array([img1, img2])
        >>> shifted_img = shift(img_list)
    """
    return Shift(horizontal, vertivcal)(x)


class Rotate(ProcessBase):

    def __init__(self):
        super(Rotate, self).__init__()

    def _transform_classification(self, x, y):
        assert len(x.shape) == 4
        n, c, h, w = x.shape
        new_x = np.empty_like(x)

        if h == w:
            # 0, 90, 180 or 270 degree.
            rotate_frag = np.random.randint(4, size=(n, ))
        else:
            # 0 or 180 degree.
            rotate_frag = np.random.randint(2, size=(n, )) * 2

        for i, r in enumerate(rotate_frag):
            new_x[i, :, :, :] = np.rot90(x[i], r, axes=(1, 2))
        return new_x, y

    def _transform_detection(self, x, y):
        assert len(x.shape) == 4
        n, c, h, w = x.shape
        c_w = w // 2
        c_h = h // 2
        new_x = np.empty_like(x)
        new_y = []

        if w == h:
            rotate_frag = np.random.randint(4, size=(n, ))
        else:
            rotate_frag = np.random.randint(2, size=(n, )) * 2

        for i, r in enumerate(rotate_frag):
            new_x[i, :, :, :] = np.rot90(x[i], r, axes=(1, 2))
            if r == 0:
                new_y.append(y[i])
            elif r == 1:
                new_y.append([
                    {
                        "box": [
                            obj["box"][1],
                            (2 * c_h - obj["box"][0]),
                            obj["box"][3],
                            obj["box"][2],
                        ],
                        "name":obj["name"],
                        "class":obj["class"],
                    }
                    for j, obj in enumerate(y[i])])
            elif r == 2:
                new_y.append([
                    {
                        "box": [
                            (2 * c_w - obj["box"][0]),
                            (2 * c_h - obj["box"][1]),
                            obj["box"][2],
                            obj["box"][3],
                        ],
                        "name":obj["name"],
                        "class":obj["class"],
                    }
                    for j, obj in enumerate(y[i])])
            elif r == 3:
                new_y.append([
                    {
                        "box": [
                            (2 * c_w - obj["box"][1]),
                            obj["box"][0],
                            obj["box"][3],
                            obj["box"][2],
                        ],
                        "name":obj["name"],
                        "class":obj["class"],
                    }
                    for j, obj in enumerate(y[i])])
        return new_x, new_y


def rotate(x, y=None, mode="classification"):
    """Rotate images randomly from 0, 90, 180, 270 degree.

    Args:
        x(list of numpy.array): List of images.
        y(list of dict): List of annotation results.
        mode(str): Type of task. You could chooose Classification, Detection or Segmentation.

    Returns:
        (list of numpy.array): List of rotated images.
        (list of dict): List of annotation results.

    Examples:
        >>> from renom_img.api.utility.augmentation.process import rotate
        >>> from PIL import Image
        >>>
        >>> img1 = Image.open(img_path1)
        >>> img2 = Image.open(img_path2)
        >>> img_list = np.array([img1, img2])
        >>> rotated_img = flip(img_list)
    """
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
    """Add white noise to images.

    Args:
        x(list of numpy.array): List of images.
        y(list of dict): List of annotation results.
        mode(str): Type of task. You could chooose Classification, Detection or Segmentation.

    Returns:
        (list of numpy.array): List of images added white noise.
        (list of dict): List of annotation results.

    Examples:
        >>> from renom_img.api.utility.augmentation.process import white_noise
        >>> from PIL import Image
        >>>
        >>> img1 = Image.open(img_path1)
        >>> img2 = Image.open(img_path2)
        >>> img_list = np.array([img1, img2])
        >>> noise_img = white_noise(img_list)
    """
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

class ContrastNorm(ProcessBase):
    def __init__(self, alpha=0.5):
        super(ContrastNorm, self).__init__()
        if isinstance(alpha, list):
            assert alpha == 2, "Expected list with 2 entries, got {} entries".format(len(alpha))

        self._alpha = alpha

    def draw_sample(size=1):
        if isinstance(alpha, list):
            return np.random.uniform(alpha[0], alpha[1], size)
        else:
            return self.alpha

    def _transform_classification(self, x, y):
        assert len(x.shape) == 4
        n = x.shape[0]
        new_x = np.empty_like(x)
        for i in range(n):
            alpha = draw_sample()
            new_x[i] = alpha*(x[i] - 128) + 128
        return new_x, y

    def _transform_detection(self, x, y):
        assert len(x.shape) == 4
        n = x.shape[0]
        new_x = np.empty_like(x)
        for i in range(n):
            alpha = draw_sample()
            new_x[i] = alpha*(x[i] - 128) + 128
        return new_x, y

