import numpy as np
from PIL import Image

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
        # assert len(x.shape) == 4
        n = len(x)
        img_list = []
        for i in range(n):
            f = np.random.randint(3)
            if f == 0:
                img_list.append(x[i][:, :, :])
            elif f == 1:
                img_list.append(x[i][:, :, ::-1])
            elif f == 2:
                img_list.append(x[i][:, ::-1, :])
        return img_list, y

    def _transform_detection(self, x, y):
        # assert len(x.shape) == 4
        n = len(x)
        img_list = []
        new_y = []


        for i in range(n):
            f = np.random.randint(3)
            if f == 0:
                img_list.append(x[i])
                new_y.append(y[i])
            elif f == 1:
                # Horizontal flip.
                c_x = x[i].shape[2] // 2
                img_list.append(x[i][:, :, ::-1])
                new_y.append([
                    {
                        "box": [
                            (2 * c_x - obj["box"][0]),
                            obj["box"][1],
                            obj["box"][2],
                            obj["box"][3],
                        ],
                        **{k: v for k, v in obj.items() if k != 'box'}
                    }
                    for j, obj in enumerate(y[i])])

            elif f == 2:
                c_y = x[i].shape[1] // 2
                img_list.append(x[i][:, ::-1, :])
                new_y.append([
                    {
                        "box": [
                            obj["box"][0],
                            (2 * c_y - obj["box"][1]),
                            obj["box"][2],
                            obj["box"][3],
                        ],
                        **{k: v for k, v in obj.items() if k != 'box'}
                    }
                    for j, obj in enumerate(y[i])])
        return img_list, new_y

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
        x (list of str): List of path of images.
        y (list of annotation): list of annotation for x. It is only used when prediction.

    Returns:
        tupple: list of transformed images and list of annotation for x.

    .. code-block :: python

        [
            x (list of numpy.ndarray), # List of transformed images.
            y (list of annotation) # list of annotation for x.
        ]

    Examples:
        >>> from renom_img.api.utility.augmentation.process import Flip
        >>> from PIL import Image
        >>>
        >>> img1 = Image.open(img_path1)
        >>> img2 = Image.open(img_path2)
        >>> img_list = np.array([img1, img2])
        >>> flipped_img = flip(img_list)
    """
    return Flip()(x, y, mode=mode)


class HorizontalFlip(ProcessBase):

    def __init__(self):
        super(HorizontalFlip, self).__init__()

    def _transform_classification(self, x, y):
        # assert len(x.shape) == 4
        n = len(x)
        img_list = []
        for i in range(n):
            f= np.random.randint(2)
            if f == 0:
                img_list.append(x[i][:, :, :])
            elif f == 1:
                img_list.append(x[i][:, :, ::-1])
        return img_list, y

    def _transform_detection(self, x, y):
        """Yet to be implemented"""
        raise NotImplemented

    def _transform_segmentation(self, x, y):
        """Yet to be implemented"""
        raise NotImplemented


def horizontalflip(x, y=None, mode="classification"):
    """Flip image randomly, only about vertical axis.

    Args:
        x (list of str): List of path of images.
        y (list of annotation): list of annotation for x. It is only used when prediction.

    Returns:
        tupple: list of transformed images and list of annotation for x.

    .. code-block :: python

        [
            x (list of numpy.ndarray), # List of transformed images.
            y (list of annotation) # list of annotation for x.
        ]

    Examples:
        >>> from renom_img.api.utility.augmentation.process import HorizontalFlip
        >>> from PIL import Image
        >>>
        >>> img1 = Image.open(img_path1)
        >>> img2 = Image.open(img_path2)
        >>> img_list = np.array([img1, img2])
        >>> flipped_img = horizontalflip(img_list)
    """
    return HorizontalFlip()(x, y, mode=mode)


# class RandomCrop(ProcessBase):
#
#     def __init__(self, padding=4):
#         super(RandomCrop, self).__init__()
#         self.padding = padding
#
#         self.sample_options = (
#         None, # using entire original input image
#         (0.1, None),
#         (0.3, None),
#         (0.7, None),
#         (0.9, None),
#         (None, None), # randomly sample a patch
#         )
#
#     def _transform_classification(self, x, y): # according to the ResNet paper
#         assert len(x.shape) == 4
#         new_x = np.empty_like(x)
#         h,w = x.shape[2],x.shape[3]
#
#         p = int(self.padding / 2)  # pad length of each side
#         x = np.pad(x, pad_width=((0, 0), (0, 0), (p, p), (p, p)),
#                    mode='constant', constant_values=0)
#
#         _h = x.shape[2]  # changed height
#         _w = x.shape[3]  # changed width
#         n = x.shape[0]  # number of batch images
#
#         rand_top = np.random.randint(0, _h - h, size=(n, ))
#         rand_left = np.random.randint(0, _w - w, size=(n, ))
#
#         for i, (top, left) in enumerate(zip(rand_top, rand_left)):
#             new_x[i, :, :, :] = x[i, :, top:top + h, left:left + w]
#
#         return new_x, y
#
#     def jaccard_overlap(self,b1_x1,b1_y1,b1_x2,b1_y2,b2_x1,b2_y1,b2_x2,b2_y2):
#         area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
#         area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
#         xA = np.fmax(b1_x1, b2_x1)
#         yA = np.fmax(b1_y1, b2_y1)
#         xB = np.fmin(b1_x2, b2_x2)
#         yB = np.fmin(b1_y2, b2_y2)
#
#         if (xB-xA) < 0 or (yB-yA) < 0:
#             return 0
#         intersect = (xB - xA) * (yB - yA)
#         union = (area1 + area2 - intersect)
#
#         return intersect / union
#
#     def check_point(self, left, top, right, down, x, y):
#         if x > left and x < right and y > top and y < down:
#             return True
#         else:
#             return False
#
#     def upscaled(self,x,y,width,height):
#         _y = []
#         h,w = x.shape[1],x.shape[2]
#         sw, sh = width / float(w), height / float(h)
#         reduced = np.squeeze(x)
#         channel_last = np.rollaxis(reduced,0,3)
#         im = Image.fromarray(np.uint8(channel_last))
#         resized = im.resize((width,height),Image.BILINEAR).convert('RGB')
#
#         arr = np.asarray(resized)
#         arr = np.expand_dims(arr,0)
#         arr = np.rollaxis(arr,3,1)
#         for j, obj in enumerate(y):
#             _y.append({
#                 "box": [obj["box"][0] * sw, obj["box"][1] * sh, obj["box"][2] * sw, obj["box"][3] * sh],
#                 **{k: v for k, v in obj.items() if k != 'box'}
#             })
#         return arr, _y
#
#     def _transform_detection(self, x, y): # according to ssd paper
#         assert len(x.shape) == 4
#         n, c, h, w = x.shape
#         new_x = np.empty_like(x)
#         new_y = []
#         for i in range(n):
#             # considering the following choice for whole batch can speed the calculation
#             choice = np.random.choice(self.sample_options)
#             if choice == None:
#                 # return the original input image
#                 new_x [i, :, :, :] = x[i, :, :, :]
#                 new_y.append(y[i])
#             elif choice[0] is not None:
#                 # do cal iou case
#                 # act_w = y[i]["size"][0]
#                 # act_h = y[i]["size"][1]
#                 temp_y = []
#                 min = choice[0]
#                 success = False
#                 while (not success):
#                     sw = int(np.random.uniform(0.3*act_w, act_w))
#                     sh = int(np.random.uniform(0.3*act_h, act_h))
#                     if sh / sw < 0.5 or sh / sw > 2:
#                         continue
#                     left = int(np.random.uniform(w - sw))
#                     top = int(np.random.uniform(h - sh))
#                     # print('cropped for :',i)
#                     # print('cropped:',left+(sw/2),top+(sh/2),sw,sh)
#                     for j, obj in enumerate(y[i]):
#                         ox,oy,ow,oh = obj["box"]
#                         # print('truth:',ox,oy,ow,oh)
#                         if self.check_point(left,top,left+sw,top+sh,ox,oy):
#                             overlap = self.jaccard_overlap(left,top,left+sw,top+sh,\
#                             ox-(ow/2),oy-(oh/2),ox+(ow/2),oy+(oh/2))
#                             # print('iou case overlap = ',overlap)
#                             # print('minimum = ',min)
#                             if overlap > min:
#                                 pw = np.clip(ox+(ow/2),0,left+sw)
#                                 ph = np.clip(oy+(oh/2),0,top+sh)
#                                 temp_y.append({
#                                     "box": [ox, oy, pw, ph],
#                                     **{k: v for k, v in obj.items() if k != 'box'}
#                                 })
#                     if len(temp_y) > 0:
#                         # ar, y = self.upscaled(x[i,:,left:left+sw,top:top+sh],temp_y,w,h)
#                         new_x [i,:,:,:], _y = self.upscaled(x[i,:,left:left+sw,top:top+sh],temp_y,w,h)
#                         new_y.append(_y)
#                         success = True
#
#
#             else:# randomly sample case
#                 temp_y = []
#                 success = False
#                 while (not success):
#                     sw = int(np.random.uniform(0.3*w, w))
#                     sh = int(np.random.uniform(0.3*h, h))
#
#                     if sh / sw < 0.5 or sh / sw > 2:
#                         continue
#                     left = int(np.random.uniform(w - sw))
#                     top = int(np.random.uniform(h - sh))
#
#                     for j, obj in enumerate(y[i]):
#                         ox,oy,ow,oh = obj["box"]
#                         if self.check_point(left,top,left+sw,top+sh,ox,oy):
#                             pw = np.clip(ox+(ow/2),0,left+sw)
#                             ph = np.clip(oy+(oh/2),0,top+sh)
#                             temp_y.append({
#                                 "box": [ox, oy, pw, ph],
#                                 **{k: v for k, v in obj.items() if k != 'box'}
#                             })
#                     if len(temp_y) > 0:
#                         # ar, y = self.upscaled(x[i,:,left:left+sw,top:top+sh],temp_y,w,h)
#                         new_x [i,:,:,:], _y = self.upscaled(x[i,:,left:left+sw,top:top+sh],temp_y,w,h)
#                         new_y.append(_y)
#                         success = True
#
#         return new_x, new_y
#
#
#     def _transform_segmentation(self, x, y):
#
#         assert len(x.shape) == 4
#         new_x = np.empty_like(x)
#         new_y = np.empty_like(y)
#         w,h = x.shape[2],x.shape[3]
#
#         p = int(self.padding / 2)  # pad length of each side
#         x = np.pad(x, pad_width=((0, 0), (0, 0), (p, p), (p, p)),
#                    mode='constant', constant_values=0)
#
#         _w = x.shape[2]  # changed height
#         _h = x.shape[3]  # changed width
#         n = x.shape[0]  # number of batch images
#
#         rand_top = np.random.randint(0, _w - w, size=(n, ))
#         rand_left = np.random.randint(0, _h - h, size=(n, ))
#
#         for i, (top, left) in enumerate(zip(rand_top, rand_left)):
#             new_x[i, :, :, :] = x[i, :, top:top + w, left:left + h]
#             new_y[i, :, :, :] = y[i, :, top:top + w, left:left + h]
#
#         return new_x, new_y
#
#
# def random_crop(x, y=None, padding=4, mode="classification"):
#     """crop image randomly.
#
#     Args:
#         x (list of str): List of path of images.
#         y (list of annotation): list of annotation for x. It is only used when prediction.
#
#     Returns:
#         tupple: list of transformed images and list of annotation for x.
#
#     .. code-block :: python
#
#         [
#             x (list of numpy.ndarray), # List of transformed images.
#             y (list of annotation) # list of annotation for x.
#         ]
#
#     Examples:
#         >>> from renom_img.api.utility.augmentation.process import RandomCrop
#         >>> from PIL import Image
#         >>>
#         >>> img1 = Image.open(img_path1)
#         >>> img2 = Image.open(img_path2)
#         >>> img_list = np.array([img1, img2])
#         >>> cropped_img = random_crop(img_list)
#     """
#     return RandomCrop(padding)(x, y, mode=mode)


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
        # assert len(x.shape) == 1
        img_list = []
        new_y = []
        n = len(x)
        for i in range(n):
            c, h, w = x[i].shape
            new_x = np.zeros_like(np.asarray(x[i]))

            rand_h = ((np.random.rand(1) * 2 - 1) * self._h).astype(np.int)
            rand_v = ((np.random.rand(1) * 2 - 1) * self._v).astype(np.int)

            new_min_x = np.clip(rand_h, 0, w)
            new_min_y = np.clip(rand_v, 0, h)
            new_max_x = np.clip(rand_h + w, 0, w)
            new_max_y = np.clip(rand_v + h, 0, h)

            orig_min_x = np.maximum(-rand_h, 0)
            orig_min_y = np.maximum(-rand_v, 0)
            orig_max_x = np.minimum(-rand_h + w, w)
            orig_max_y = np.minimum(-rand_v + h, h)

            new_x[ :, new_min_y[0] : new_max_y[0], new_min_x[0]:new_max_x[0]] = \
                 x[i][:, orig_min_y[0]:orig_max_y[0], orig_min_x[0]:orig_max_x[0]]
            ny = []
            for j, obj in enumerate(y[i]):
                pw = obj["box"][2]
                ph = obj["box"][3]
                px1 = np.clip(obj["box"][0] - pw / 2. + rand_h[0], 0, w - 1)
                py1 = np.clip(obj["box"][1] - ph / 2. + rand_v[0], 0, h - 1)
                px2 = np.clip(obj["box"][0] + pw / 2. + rand_h[0], 0, w - 1)
                py2 = np.clip(obj["box"][1] + ph / 2. + rand_v[0], 0, h - 1)
                pw = px2 - px1
                ph = py2 - py1
                px = px1 + pw / 2.
                py = py1 + ph / 2.
                ny.append({
                    "box": [px, py, pw, ph],
                    **{k: v for k, v in obj.items() if k != 'box'}
                })
            new_y.append(ny)
            img_list.append(np.asarray(new_x))

        return img_list, new_y

    def _transform_segmentation(self, x, y):
        assert len(x.shape) == 4
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
            new_y[i, :, new_min_y[i]:new_max_y[i], new_min_x[i]:new_max_x[i]] = \
                y[i, :, orig_min_y[i]:orig_max_y[i], orig_min_x[i]:orig_max_x[i]]
        return new_x, new_y


def shift(x, y=None, horizontal=10, vertivcal=10, mode="classification"):
    """Shift images randomly according to given parameter.

    Args:
        x (list of str): List of path of images.
        y (list of annotation): list of annotation for x. It is only used when prediction.

    Returns:
        tupple: list of transformed images and list of annotation for x.

    .. code-block :: python

        [
            x (list of numpy.ndarray), # List of transformed images.
            y (list of annotation) # list of annotation for x.
        ]


    Examples:
        >>> from renom_img.api.utility.augmentation.process import shift
        >>> from PIL import Image
        >>>
        >>> img1 = Image.open(img_path1)
        >>> img2 = Image.open(img_path2)
        >>> img_list = np.array([img1, img2])
        >>> shifted_img = shift(img_list)
    """
    return Shift(horizontal, vertivcal)(x, y, mode=mode)


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
        # assert len(x.shape) == 4
        n = len(x)
        img_list = []
        new_y = []


        for i in range(n):
            c, h, w = x[i].shape
            c_w = w // 2
            c_h = h // 2


            if w == h:
                r = np.random.randint(4)
            else:
                r = np.random.randint(2) * 2

            img_list.append(np.rot90(x[i], r, axes=(1, 2)))
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
                        **{k: v for k, v in obj.items() if k != 'box'}
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
                        **{k: v for k, v in obj.items() if k != 'box'}
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
                        **{k: v for k, v in obj.items() if k != 'box'}
                    }
                    for j, obj in enumerate(y[i])])
        return img_list, new_y

    def _transform_segmentation(self, x, y):
        assert len(x.shape) == 4
        n, c, h, w = x.shape
        new_x = np.empty_like(x)
        new_y = np.empty_like(y)

        if h == w:
            # 0, 90, 180 or 270 degree.
            rotate_frag = np.random.randint(4, size=(n, ))
        else:
            # 0 or 180 degree.
            rotate_frag = np.random.randint(2, size=(n, )) * 2

        for i, r in enumerate(rotate_frag):
            new_x[i, :, :, :] = np.rot90(x[i], r, axes=(1, 2))
            new_y[i, :, :, :] = np.rot90(y[i], r, axes=(1, 2))
        return new_x, new_y


def rotate(x, y=None, mode="classification"):
    """Rotate images randomly from 0, 90, 180, 270 degree.

    Args:
        x (list of str): List of path of images.
        y (list of annotation): list of annotation for x. It is only used when prediction.

    Returns:
        tupple: list of transformed images and list of annotation for x.

    .. code-block :: python

        [
            x (list of numpy.ndarray), # List of transformed images.
            y (list of annotation) # list of annotation for x.
        ]

    Examples:
        >>> from renom_img.api.utility.augmentation.process import rotate
        >>> from PIL import Image
        >>>
        >>> img1 = Image.open(img_path1)
        >>> img2 = Image.open(img_path2)
        >>> img_list = np.array([img1, img2])
        >>> rotated_img = rotate(img_list)
    """
    return Rotate()(x, y, mode=mode)


class WhiteNoise(ProcessBase):

    def __init__(self, std=0.01):
        super(WhiteNoise, self).__init__()
        self._std = std

    def _transform_classification(self, x, y):
        assert len(x.shape) == 4
        return x + self._std * np.random.randn(*x.shape), y

    def _transform_detection(self, x, y):
        # assert len(x.shape) == 4
        img_list = []
        n = len(x)
        for i in range(n):
            img_list.append(x[i] + self._std * np.random.randn(*x[i].shape))
        return img_list, y

    def _transform_segmentation(self, x, y):
        assert len(x.shape) == 4
        return x + self._std * np.random.randn(*x.shape), y


def white_noise(x, y=None, std=0.01, mode="classification"):
    """Add white noise to images.

    Args:
        x (list of str): List of path of images.
        y (list of annotation): list of annotation for x. It is only used when prediction.

    Returns:
        tupple: list of transformed images and list of annotation for x.

        .. code-block :: python

            [
                x (list of numpy.ndarray), # List of transformed images.
                y (list of annotation) # list of annotation for x.
            ]

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

    def _transform_detection(self, x, y):
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

    def _transform_segmentation(self, x, y):
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
    def __init__(self, alpha=0.5, per_channel=False):
        super(ContrastNorm, self).__init__()
        if isinstance(alpha, list):
            assert len(alpha) == 2, "Expected list with 2 entries, got {} entries".format(len(alpha))
        else:
            assert alpha >= 0.0, "Expected alpha to be larger or equal to 0.0, got {}".format(alpha)
        self._alpha = alpha
        self._per_channel = per_channel

    def _draw_sample(self, size=1):
        if isinstance(self._alpha, list):
            return np.random.uniform(self._alpha[0], self._alpha[1], size)
        else:
            return self._alpha

    def _transform_classification(self, x, y):
        assert len(x.shape) == 4
        n = x.shape[0]
        new_x = np.empty_like(x)
        for i in range(n):
            if self._per_channel and isinstance(self._alpha, list):
                channel = x.shape[1]
                alpha = self._draw_sample(size=channel)
                for c in range(channel):
                    new_x[i, c, :, :] = np.clip(alpha[c] * (x[i, c, :, :] - 128) + 128, 0, 255)
            else:
                alpha = self._draw_sample()
                new_x[i] = np.clip(alpha * (x[i] - 128) + 128, 0, 255)

        return new_x, y

    def _transform_detection(self, x, y):
        # assert len(x.shape) == 4
        n = len(x)
        img_list = []
        for i in range(n):
            if self._per_channel and isinstance(self._alpha, list):
                new_x = np.empty_like(x[i])
                channel = x[i].shape[0]
                alpha = self._draw_sample(size=channel)
                for c in range(channel):
                    new_x[c, :, :] = np.clip(alpha[c] * (x[i][c, :, :] - 128) + 128, 0, 255)
                img_list.append(new_x)
            else:
                alpha = self._draw_sample()
                img_list.append(np.clip(alpha * (x[i] - 128) + 128, 0, 255))

        return img_list, y

    def _transform_segmentation(self, x, y):
        assert len(x.shape) == 4
        n = x.shape[0]
        new_x = np.empty_like(x)
        for i in range(n):
            if self._per_channel and isinstance(self._alpha, list):
                channel = x.shape[1]
                alpha = self._draw_sample(size=channel)
                for c in range(channel):
                    new_x[i, c, :, :] = np.clip(alpha[c] * (x[i, c, :, :] - 128) + 128, 0, 255)
            else:
                alpha = self._draw_sample()
                new_x[i] = np.clip(alpha * (x[i] - 128) + 128, 0, 255)

        return new_x, y


def contrast_norm(x, y=None, alpha=0.5, per_channel=False, mode='classification'):
    """ Contrast Normalization

    Args:
        x (list of str): List of path of images.
        y (list of annotation): list of annotation for x. It is only used when prediction.
        alpha(float or list of two floats): Higher value increases contrast, and lower value decreases contrast.
                                            if a list [a, b], alpha value is sampled from uniform distribution ranging from [a, b).
                                            if a float, constant value of alpha is used.
        per_channel(Bool): Whether to apply contrast normalization for each channel.
                           If alpha is given a list, then different values for each channel are used.

    Returns:
        tupple: list of transformed images and list of annotation for x.

    .. code-block :: python

        [
            x (list of numpy.ndarray), # List of transformed images.
            y (list of annotation) # list of annotation for x.
        ]


    Example:
        >>> img = Image.open(img_path)
        >>> img.convert('RGB')
        >>> img = np.array(img).transpose(2, 0, 1).astype(np.float)
        >>> x = np.array([img])
        >>> new_x, new_y = contrast_norm(x, alpha=0.4)
    """
    return ContrastNorm(alpha, per_channel)(x, y, mode=mode)
