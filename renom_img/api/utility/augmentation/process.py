import numpy as np
from PIL import Image
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

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
        # assert len(x.shape) == 4
        n = len(x)
        img_list = []
        new_y = []
        for i in range(n):
            f = np.random.randint(3)
            if f == 0:
                img_list.append(x[i][:, :, :])
                new_y.append(y[i][:, :, :])
            elif f == 1:
                img_list.append(x[i][:, :, ::-1])
                new_y.append(y[i][:, :, ::-1])
            elif f == 2:
                img_list.append(x[i][:, ::-1, :])
                new_y.append(y[i][:, ::-1, :])
        return img_list, new_y


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

    def __init__(self,prob=True):
        super(HorizontalFlip, self).__init__()
        self.prob = prob

    def _transform_classification(self, x, y):
        # assert len(x.shape) == 4
        n = len(x)
        img_list = []
        for i in range(n):
            f= np.random.randint(2) if self.prob else 1
            if f == 0:
                img_list.append(x[i][:, :, :])
            elif f == 1:
                img_list.append(x[i][:, :, ::-1])
        return img_list, y

    def _transform_detection(self, x, y):
        n = len(x)
        new_x = []
        new_y = []
        for i in range(n):
            f = np.random.randint(2) if self.prob else 1
            if f ==0:
                new_x.append(x[i])
                new_y.append(y[i])
            else:
                new_x.append(x[i][:,:,::-1])
                c_x = x[i].shape[2] // 2
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
            return new_x,new_y

    def _transform_segmentation(self, x, y):
        n = len(x)
        img_list = []
        new_y = []
        for i in range(n):
            f= np.random.randint(2) if self.prob else 1
            if f == 0:
                img_list.append(x[i][:, :, :])
                new_y.append(y[i][:,:,:])
            elif f == 1:
                img_list.append(x[i][:, :, ::-1])
                new_y.append(y[i][:,:,::-1])
        return img_list, new_y


def horizontalflip(x, y=None, prob=True, mode="classification"):
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
    return HorizontalFlip(prob=prob)(x, y, mode=mode)

class VerticalFlip(ProcessBase):

    def __init__(self,prob=True):
        super(VerticalFlip, self).__init__()
        self.prob = prob

    def _transform_classification(self, x, y):
        # assert len(x.shape) == 4
        n = len(x)
        img_list = []
        for i in range(n):
            f= np.random.randint(2) if self.prob else 1
            if f == 0:
                img_list.append(x[i][:, :, :])
            elif f == 1:
                img_list.append(x[i][:, ::-1, :])
        return img_list, y

    def _transform_detection(self, x, y):
        n = len(x)
        new_x = []
        new_y = []
        for i in range(n):
            f = np.random.randint(2) if self.prob else 1
            if f ==0:
                new_x.append(x[i])
                new_y.append(y[i])
            else:
                c_y = x[i].shape[1] // 2
                new_x.append(x[i][:, ::-1, :])
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
            return new_x,new_y

    def _transform_segmentation(self, x, y):
        n = len(x)
        img_list = []
        new_y = []
        for i in range(n):
            f= np.random.randint(2) if self.prob else 1
            if f == 0:
                img_list.append(x[i][:, :, :])
                new_y.append(y[i][:,:,:])
            elif f == 1:
                img_list.append(x[i][:, ::-1, :])
                new_y.append(y[i][:,::-1,:])
        return img_list, new_y


def verticalflip(x, y=None, prob=True, mode="classification"):
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
    return VerticalFlip(prob=prob)(x, y, mode=mode)


class RandomCrop(ProcessBase):

    def __init__(self, padding=4):
        super(RandomCrop, self).__init__()
        self.padding = padding

        self.sample_options = (
        None, # using entire original input image
        (0.1, None),
        (0.3, None),
        (0.7, None),
        (0.9, None),
        (None, None), # randomly sample a patch
        )

    def _transform_classification(self, x, y): # according to the ResNet paper
        # assert len(x.shape) == 4
        img_list = []
        n = len(x)
        for i in range(n):
            h,w = x[i].shape[1],x[i].shape[2]

            p = int(self.padding / 2)  # pad length of each side
            x = np.pad(x[i], pad_width=((0, 0), (p, p), (p, p)),
                       mode='constant', constant_values=0)
            _h = x.shape[1]  # changed height
            _w = x.shape[2]  # changed width
            top = np.random.randint(0, _h - h)
            left = np.random.randint(0, _w - w)

            img_list.append(x[:, top:top + h, left:left + w])

        return img_list, y

    def jaccard_overlap(self,b1_x1,b1_y1,b1_x2,b1_y2,b2_x1,b2_y1,b2_x2,b2_y2):
        area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
        area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        xA = np.fmax(b1_x1, b2_x1)
        yA = np.fmax(b1_y1, b2_y1)
        xB = np.fmin(b1_x2, b2_x2)
        yB = np.fmin(b1_y2, b2_y2)

        intersect = (xB - xA) * (yB - yA)
        union = (area1 + area2 - intersect)
        iou = intersect/union

        return iou, (xB-((xB - xA)/2))-b1_x1,(yB-((yB-yA)/2))-b1_y1, xB - xA,yB - yA

    def check_point(self, left, top, right, down, x, y):
        if x > left and x < right and y > top and y < down:
            return True
        else:
            return False

    def _transform_detection(self, x, y): # according to ssd paper
        # assert len(x.shape) == 4
        n = len(x)
        img_list = []
        new_y = []
        for i in range(n):
            # considering the following choice for whole batch can speed the calculation
            choice = np.random.choice(self.sample_options)
            c, h, w = x[i].shape
            if choice == None:
                # print('original image')
                img_list.append(x[i])
                new_y.append(y[i])
            elif choice[0] is not None:
                # print('iou case')
                temp_y = []
                min = choice[0]
                success = False
                counter = 0
                iou_counter = 0
                while (not success):
                    sw = int(np.random.uniform(0.3*w, w))
                    sh = int(np.random.uniform(0.3*h, h))
                    if sw / sh < 0.5 or sw / sh > 2:
                        iou_counter += 1
                        if iou_counter > 20:
                            img_list.append(x[i])
                            new_y.append(y[i])
                            success = True
                        continue
                    left = int(np.random.uniform(w - sw))
                    top = int(np.random.uniform(h - sh))
                    for j, obj in enumerate(y[i]):
                        ox,oy,ow,oh = obj["box"]
                        if self.check_point(left,top,left+sw,top+sh,ox,oy):
                            overlap,px,py,pw,ph = self.jaccard_overlap(left,top,left+sw,top+sh,\
                            ox-(ow/2),oy-(oh/2),ox+(ow/2),oy+(oh/2))
                            if overlap >= min:
                                pw = np.clip(pw,0,left+sw)
                                ph = np.clip(ph,0,top+sh)
                                temp_y.append({
                                    "box": [px, py, pw, ph],
                                    **{k: v for k, v in obj.items() if k != 'box'}
                                })
                    if len(temp_y) > 0:
                        success = True
                        img_list.append(x[i][:,top:top+sh, left:left+sw])
                        new_y.append(temp_y)
                    else:
                        counter += 1
                        if counter > 50 or np.random.rand() >= 0.85:
                            success = True
                            img_list.append(x[i])
                            new_y.append(y[i])

            else:
                # print('random case')
                temp_y = []
                success = False
                counter = 0
                random_counter = 0
                while (not success):
                    sw = int(np.random.uniform(0.3*w, w))
                    sh = int(np.random.uniform(0.3*h, h))

                    if sw / sh < 0.5 or sw / sh > 2:
                        random_counter += 1
                        if random_counter > 20:
                            img_list.append(x[i])
                            new_y.append(y[i])
                            success = True
                        continue
                    left = int(np.random.uniform(w - sw))
                    top = int(np.random.uniform(h - sh))

                    for j, obj in enumerate(y[i]):
                        ox,oy,ow,oh = obj["box"]
                        if self.check_point(left,top,left+sw,top+sh,ox,oy):
                            overlap,px,py,pw,ph = self.jaccard_overlap(left,top,left+sw,top+sh,\
                            ox-(ow/2),oy-(oh/2),ox+(ow/2),oy+(oh/2))
                            pw = np.clip(pw,0,left+sw)
                            ph = np.clip(ph,0,top+sh)
                            temp_y.append({
                                "box": [px, py, pw, ph],
                                **{k: v for k, v in obj.items() if k != 'box'}
                            })
                    if len(temp_y) > 0:
                        success = True
                        img_list.append(x[i][:,top:top+sh,left:left+sw])
                        new_y.append(temp_y)
                    else:
                        counter += 1
                        if counter > 50:
                            success = True
                            img_list.append(x[i])
                            new_y.append(y[i])

        return img_list, new_y


    def _transform_segmentation(self, x, y):
        # assert len(x.shape) == 4
        img_list = []
        new_y = []
        n = len(x)
        for i in range(n):
            h,w = x[i].shape[1],x[i].shape[2]

            p = int(self.padding / 2)  # pad length of each side
            x = np.pad(x[i], pad_width=((0, 0), (p, p), (p, p)),
                       mode='constant', constant_values=0)
            y = np.pad(y[i], pad_width=((0, 0), (p, p), (p, p)),
                       mode='constant', constant_values=0)
            _h = x.shape[1]  # changed height
            _w = x.shape[2]  # changed width


            top = np.random.randint(0, _h - h)
            left = np.random.randint(0, _w - w)

            img_list.append(x[:, top:top + h, left:left + w])
            new_y.append(y[:, top:top + h, left:left + w])
        return img_list, new_y


def random_crop(x, y=None, padding=4, mode="classification"):
    """crop image randomly.

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
        >>> from renom_img.api.utility.augmentation.process import RandomCrop
        >>> from PIL import Image
        >>>
        >>> img1 = Image.open(img_path1)
        >>> img2 = Image.open(img_path2)
        >>> img_list = np.array([img1, img2])
        >>> cropped_img = random_crop(img_list)
    """
    return RandomCrop(padding)(x, y, mode=mode)

class CenterCrop(ProcessBase):

    def __init__(self, size=(224,224)):
        super(CenterCrop, self).__init__()
        assert len(size) == 2, "crop size should be a tuple, for example (224,224)"
        self.size = size

    def cal_overlap(self,b1_x1,b1_y1,b1_x2,b1_y2,b2_x1,b2_y1,b2_x2,b2_y2):
        xA = np.fmax(b1_x1, b2_x1)
        yA = np.fmax(b1_y1, b2_y1)
        xB = np.fmin(b1_x2, b2_x2)
        yB = np.fmin(b1_y2, b2_y2)

        return (xB-((xB - xA)/2))-b1_x1,(yB-((yB-yA)/2))-b1_y1, xB - xA,yB - yA

    def check_point(self, left, top, right, down, x, y):
        if x > left and x < right and y > top and y < down:
            return True
        else:
            return False

    def _transform_classification(self, x, y):
        n = len(x)
        new_x = []
        for i in range(n):
            c,h,w = x[i].shape
            assert self.size[0] < w and self.size[1] < h, 'crop size should be smaller than original image size'
            left = np.ceil((w-self.size[0])/2.).astype(int)
            top = np.ceil((h-self.size[1])/2.).astype(int)
            right = np.floor((w+self.size[0])/2.).astype(int)
            bottom = np.floor((h+self.size[1])/2.).astype(int)
            img = x[i][:,top:bottom,left:right]
            new_x.append(img)
        return new_x,y

    def _transform_detection(self,x,y):
        n = len(x)
        new_x = []
        new_y = []
        for i in range(n):
             c,h,w = x[i].shape
             temp_y = []
             assert self.size[0] < w and self.size[1] < h, 'crop size should be smaller than original image size'
             left = np.ceil((w-self.size[0])/2.).astype(int)
             top = np.ceil((h-self.size[1])/2.).astype(int)
             right = np.floor((w+self.size[0])/2.).astype(int)
             bottom = np.floor((h+self.size[1])/2.).astype(int)
             img = x[i][:,top:bottom,left:right]

             for j, obj in enumerate(y[i]):
                 ox,oy,ow,oh = obj["box"]
                 # print(ox,oy,ow,oh)
                 if self.check_point(left,top,right,bottom,ox,oy):
                     px,py,pw,ph = self.cal_overlap(left,top,right,bottom,ox-ow/2,oy-oh/2,ox+ow/2,oy+oh/2)
                     # print(px,py,pw,ph)
                     temp_y.append({
                         "box": [px, py, pw, ph],
                         **{k: v for k, v in obj.items() if k != 'box'}
                     })
             if len(temp_y)>0:
                new_x.append(img)
                new_y.append(temp_y)
             else:
                new_x.append(x[i])
                new_y.append(y[i])
        return new_x,new_y

    def _transform_segmentation(self,x,y):
        n = len(x)
        new_x = []
        new_y = []
        for i in range(n):
            c,h,w = x[i].shape
            assert self.size[0] < w and self.size[1] < h, 'crop size should be smaller than original image size'
            left = np.ceil((w-self.size[0])/2.).astype(int)
            top = np.ceil((h-self.size[1])/2.).astype(int)
            right = np.floor((w+self.size[0])/2.).astype(int)
            bottom = np.floor((h+self.size[1])/2.).astype(int)
            img = x[i][:,top:bottom,left:right]
            label = y[i][:,top:bottom,left:right]
            new_x.append(img)
            new_y.append(label)
        return new_x,new_y

def center_crop(x,y,mode="classification"):
    return CenterCrop(size=(224,224))(x, y, mode=mode)



class Shift(ProcessBase):

    def __init__(self, horizontal=10, vertivcal=10):
        super(Shift, self).__init__()
        self._h = horizontal
        self._v = vertivcal

    def _transform_classification(self, x, y):
        # assert len(x.shape) == 4
        n = len(x)
        img_list = []
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

            new_x[ :, new_min_y[0]:new_max_y[0], new_min_x[0]:new_max_x[0]] = \
                x[i][:, orig_min_y[0]:orig_max_y[0], orig_min_x[0]:orig_max_x[0]]
            img_list.append(np.asarray(new_x))
        return img_list, y

    def _transform_detection(self, x, y):
        # assert len(x.shape) == 1
        img_list = []
        new_y = []
        n = len(x)
        for i in range(n):
            success = False
            while (not success):
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

                new_x[:, new_min_y[0] : new_max_y[0], new_min_x[0]:new_max_x[0]] = \
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
                    if pw == 0 or ph == 0:
                        continue
                    px = px1 + pw / 2.
                    py = py1 + ph / 2.
                    ny.append({
                        "box": [px, py, pw, ph],
                        **{k: v for k, v in obj.items() if k != 'box'}
                    })
                if len(ny) > 0:
                    success = True
                    new_y.append(ny)
                    img_list.append(np.asarray(new_x))

        return img_list, new_y

    def _transform_segmentation(self, x, y):
        # assert len(x.shape) == 4
        n = len(x)
        img_list = []
        label_list = []
        for i in range(n):
            c , h, w = x[i].shape
            new_x = np.zeros_like(np.asarray(x[i]))
            new_y = np.zeros_like(np.asarray(y[i]))
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

            new_x[:, new_min_y[0]:new_max_y[0], new_min_x[0]:new_max_x[0]] = \
                x[i][:, orig_min_y[0]:orig_max_y[0], orig_min_x[0]:orig_max_x[0]]
            new_y[:, new_min_y[0]:new_max_y[0], new_min_x[0]:new_max_x[0]] = \
                y[i][:, orig_min_y[0]:orig_max_y[0], orig_min_x[0]:orig_max_x[0]]
            img_list.append(np.asarray(new_x))
            label_list.append(np.asarray(new_y))
        return img_list, label_list


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
        # assert len(x.shape) == 4
        n = len(x)
        img_list = []
        for i in range(n):
            c, h, w = x[i].shape
            if h == w:
                # 0, 90, 180 or 270 degree.
                r = np.random.randint(4)
            else:
                # 0 or 180 degree.
                r = np.random.randint(2) * 2

            img_list.append(np.rot90(x[i], r, axes=(1, 2)))

        return img_list, y

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
        # assert len(x.shape) == 4
        n= len(x)
        new_x = []
        new_y = []
        for i in range(n):
            c,h,w = x[i].shape
            if h == w:
                # 0, 90, 180 or 270 degree.
                r = np.random.randint(4)
            else:
                # 0 or 180 degree.
                r = np.random.randint(2) * 2
            new_x.append(np.rot90(x[i], r, axes=(1, 2)))
            new_y.append(np.rot90(y[i], r, axes=(1, 2)))
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
        # assert len(x.shape) == 4
        img_list = []
        n = len(x)
        for i in range(n):
            img_list.append(x[i] + self._std * np.random.randn(*x[i].shape))
        return img_list, y

    def _transform_detection(self, x, y):
        # assert len(x.shape) == 4
        img_list = []
        n = len(x)
        for i in range(n):
            img_list.append(x[i] + self._std * np.random.randn(*x[i].shape))
        return img_list, y

    def _transform_segmentation(self, x, y):
        # assert len(x.shape) == 4
        img_list = []
        n = len(x)
        for i in range(n):
            img_list.append(x[i] + self._std * np.random.randn(*x[i].shape))
        return img_list, y


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

class Distortion(ProcessBase):
    def __init__(self, random_state=None):
        super(Distortion, self).__init__()
        self.random_state= random_state
        self.choice_list = (None,
                            (1201,10),
                            (1501,12),
                            (991,8))
        if self.random_state is None:
            self.random_state = np.random.RandomState(None)

    def _transform_classification(self,x,y):
        n = len(x)
        new_x = []
        for i in range(n):
            choice = np.random.choice(self.choice_list)
            if choice is None:
                new_x.append(x[i])
            else:
                self.alpha,self.sigma = choice[0],choice[1]
                img = x[i]
                shape = img.shape
                dx = gaussian_filter((self.random_state.rand(*shape) * 2 -1),self.sigma,mode='constant',cval=0) * self.alpha
                dy = gaussian_filter((self.random_state.rand(*shape) * 2 -1),self.sigma,mode='constant',cval=0) * self.alpha

                ax, ay, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))

                indices = np.reshape(ay+dy, (-1, 1)), np.reshape(ax+dx, (-1, 1)), np.reshape(z, (-1, 1))
                # print(indices[0].shape,indices[1].shape,indices[2].shape)
                distorted_image = map_coordinates(img, indices, order=1, mode='reflect')
                new_x.append(distorted_image.reshape(img.shape))

        return new_x,y

    def _transform_detection(self,x,y):
        n = len(x)
        new_x = []
        for i in range(n):
            choice = np.random.choice(self.choice_list)
            if choice is None:
                # print('return original')
                new_x.append(x[i])
            else:
                # print('distorted')
                self.alpha,self.sigma = choice[0],choice[1]
                img = x[i]
                shape = img.shape
                dx = gaussian_filter((self.random_state.rand(*shape) * 2 -1),self.sigma,mode='constant',cval=0) * self.alpha
                dy = gaussian_filter((self.random_state.rand(*shape) * 2 -1),self.sigma,mode='constant',cval=0) * self.alpha

                ax, ay, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))

                indices = np.reshape(ay+dy, (-1, 1)), np.reshape(ax+dx, (-1, 1)), np.reshape(z, (-1, 1))
                # print(indices[0].shape,indices[1].shape,indices[2].shape)
                distorted_image = map_coordinates(img, indices, order=1, mode='reflect')

                new_x.append(distorted_image.reshape(img.shape))
        return new_x,y

    def _transform_segmentation(self,x,y):
        n = len(x)
        new_x = []
        new_y = []
        for i in range(n):
            choice = np.random.choice(self.choice_list)
            if choice is None:
                # print('return original')
                new_x.append(x[i])
                new_y.append(y[i])
            else:
                # print('distorted')
                self.alpha,self.sigma = choice[0],choice[1]
                img = x[i]
                shape = img.shape
                dx = gaussian_filter((self.random_state.rand(*shape) * 2 -1),self.sigma,mode='constant',cval=0) * self.alpha
                dy = gaussian_filter((self.random_state.rand(*shape) * 2 -1),self.sigma,mode='constant',cval=0) * self.alpha

                ax, ay, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))

                indices = np.reshape(ay+dy, (-1, 1)), np.reshape(ax+dx, (-1, 1)), np.reshape(z, (-1, 1))
                # print(indices[0].shape,indices[1].shape,indices[2].shape)
                distorted_image = map_coordinates(img, indices, order=1, mode='reflect')

                new_x.append(distorted_image.reshape(img.shape))

                label = y[i]
                shape = label.shape
                dx = gaussian_filter((self.random_state.rand(*shape) * 2 -1),self.sigma,mode='constant',cval=0) * self.alpha
                dy = gaussian_filter((self.random_state.rand(*shape) * 2 -1),self.sigma,mode='constant',cval=0) * self.alpha

                ax, ay, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))

                indices = np.reshape(ay+dy, (-1, 1)), np.reshape(ax+dx, (-1, 1)), np.reshape(z, (-1, 1))
                # print(indices[0].shape,indices[1].shape,indices[2].shape)
                distorted_label = map_coordinates(label, indices, order=1, mode='reflect')

                new_y.append(distorted_label.reshape(label.shape))
        return new_x,new_y


def distortion(x, y, mode='classification'):
    return Distortion()(x,y,mode=mode)

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


class RandomBrightness(ProcessBase):
    def __init__(self, delta=32):
        super(RandomBrightness, self).__init__()
        self._delta = delta

    def _transform_classification(self, x, y):
        # assert len(x.shape) == 4
        n = len(x)
        img_list = []
        for i in range(n):
            if np.random.randint(2):
                delta = np.random.uniform(-self._delta,self._delta)
                x[i] = x[i] + delta
                img_list.append(np.clip(x[i],0,255))
            else:
                img_list.append(x[i])

        return img_list, y

    def _transform_detection(self, x, y):
        # assert len(x.shape) == 4
        n = len(x)
        img_list = []
        for i in range(n):
            if np.random.randint(2):
                delta = np.random.uniform(-self._delta,self._delta)
                x[i] = x[i] + delta
                img_list.append(np.clip(x[i],0,255))
            else:
                img_list.append(x[i])

        return img_list, y

    def _transform_segmentation(self, x, y):
        # assert len(x.shape) == 4
        n = len(x)
        img_list = []
        for i in range(n):
            if np.random.randint(2):
                delta = np.random.uniform(-self._delta,self._delta)
                x[i] = x[i] + delta
                img_list.append(np.clip(x[i],0,255))
            else:
                img_list.append(x[i])

        return img_list, y


def random_brightness(x, y=None, delta=32, mode='classification'):

    return RandomBrightness(delta)(x, y, mode=mode)


class RandomHue(ProcessBase):
    def __init__(self, max_delta=0.3):
        super(RandomHue, self).__init__()
        self.max_delta = max_delta
        assert self.max_delta > 0 and self.max_delta <= 0.5, "max_delta must be in the interval [0, 0.5]."
        self.tyiq = np.array([[0.299, 0.587, 0.114],
                              [0.596, -0.274, -0.321],
                              [0.211, -0.523, 0.311]])
        self.ityiq = np.array([[1.0, 0.956, 0.621],
                               [1.0, -0.272, -0.647],
                               [1.0, -1.107, 1.705]])

    def _transform_classification(self, x, y):
        # assert len(x.shape) == 4
        n = len(x)
        img_list = []
        for i in range(n):
            alpha = np.random.uniform(-self.max_delta,self.max_delta)
            u = np.cos(alpha * np.pi)
            w = np.sin(alpha * np.pi)
            bt = np.array([[1.0, 0.0, 0.0],
                       [0.0, u, -w],
                       [0.0, w, u]])
            t = np.dot(np.dot(self.ityiq, bt), self.tyiq).T
            src = np.dot(x[i].transpose(1,2,0), np.array(t))
            img_list.append(src.transpose(2,0,1))
        return img_list,y

    def _transform_detection(self, x, y):
        # assert len(x.shape) == 4
        n = len(x)
        img_list = []
        for i in range(n):
            alpha = np.random.uniform(-self.max_delta,self.max_delta)
            u = np.cos(alpha * np.pi)
            w = np.sin(alpha * np.pi)
            bt = np.array([[1.0, 0.0, 0.0],
                       [0.0, u, -w],
                       [0.0, w, u]])
            t = np.dot(np.dot(self.ityiq, bt), self.tyiq).T
            src = np.dot(x[i].transpose(1,2,0), np.array(t))
            img_list.append(src.transpose(2,0,1))
        return img_list,y

    def _transform_segmentation(self, x, y):
        # assert len(x.shape) == 4
        n = len(x)
        img_list = []
        for i in range(n):
            alpha = np.random.uniform(-self.max_delta,self.max_delta)
            u = np.cos(alpha * np.pi)
            w = np.sin(alpha * np.pi)
            bt = np.array([[1.0, 0.0, 0.0],
                       [0.0, u, -w],
                       [0.0, w, u]])
            t = np.dot(np.dot(self.ityiq, bt), self.tyiq).T
            src = np.dot(x[i].transpose(1,2,0), np.array(t))
            img_list.append(src.transpose(2,0,1))
        return img_list,y


def random_hue(x, y=None, mode='classification'):
    return RandomHue(max_delta=0.3)(x, y, mode=mode)

class RandomSaturation(ProcessBase):
    def __init__(self, ratio=0.4):
        super(RandomSaturation, self).__init__()
        self.ratio = ratio
        assert self.ratio > 0 and self.ratio < 1, "ratio must be in the interval [0,1]."
        self.coef = np.array([[[0.299, 0.587, 0.114]]])

    def _transform_classification(self, x, y):
        # assert len(x.shape) == 4
        n = len(x)
        img_list = []
        for i in range(n):
            alpha = 1.0 + np.random.uniform(-self.ratio, self.ratio)
            gray = x[i].transpose(1,2,0) * self.coef
            gray = np.sum(gray, axis=2, keepdims=True)
            gray *= (1.0 - alpha)
            img = x[i].transpose(1,2,0) * alpha
            img += gray
            img_list.append(img.transpose(2,0,1))
        return img_list,y

    def _transform_detection(self, x, y):
        # assert len(x.shape) == 4
        n = len(x)
        img_list = []
        for i in range(n):
            alpha = 1.0 + np.random.uniform(-self.ratio, self.ratio)
            gray = x[i].transpose(1,2,0) * self.coef
            gray = np.sum(gray, axis=2, keepdims=True)
            gray *= (1.0 - alpha)
            img = x[i].transpose(1,2,0) * alpha
            img += gray
            img_list.append(img.transpose(2,0,1))
        return img_list,y

    def _transform_segmentation(self, x, y):
        # assert len(x.shape) == 4
        n = len(x)
        img_list = []
        for i in range(n):
            alpha = 1.0 + np.random.uniform(-self.ratio, self.ratio)
            gray = x[i].transpose(1,2,0) * self.coef
            gray = np.sum(gray, axis=2, keepdims=True)
            gray *= (1.0 - alpha)
            img = x[i].transpose(1,2,0) * alpha
            img += gray
            img_list.append(img.transpose(2,0,1))
        return img_list,y

def random_saturation(x, y=None, mode='classification'):
    return RandomSaturation(ratio=0.4)(x, y, mode=mode)

class RandomLighting(ProcessBase):
    def __init__(self):
        super(RandomLighting, self).__init__()
        self.choice = (None, (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def _transform_classification(self, x, y):
        # assert len(x.shape) == 4
        n = len(x)
        img_list = []
        for i in range(n):
            choice = np.random.choice(self.choice)
            if choice == None:
                choice = (0,1,2)
            new_x = np.empty_like(x[i])
            new_x[choice[0],:,:] = x[i][0,:,:]
            new_x[choice[1],:,:] = x[i][1,:,:]
            new_x[choice[2],:,:] = x[i][2,:,:]
            img_list.append(new_x)
        return img_list,y

    def _transform_detection(self, x, y):
        # assert len(x.shape) == 4
        n = len(x)
        img_list = []
        for i in range(n):
            choice = np.random.choice(self.choice)
            if choice == None:
                choice = (0,1,2)
            new_x = np.empty_like(x[i])
            new_x[choice[0],:,:] = x[i][0,:,:]
            new_x[choice[1],:,:] = x[i][1,:,:]
            new_x[choice[2],:,:] = x[i][2,:,:]
            img_list.append(new_x)
        return img_list,y

    def _transform_segmentation(self, x, y):
        # assert len(x.shape) == 4
        n = len(x)
        img_list = []
        for i in range(n):
            choice = np.random.choice(self.choice)
            if choice == None:
                choice = (0,1,2)
            new_x = np.empty_like(x[i])
            new_x[choice[0],:,:] = x[i][0,:,:]
            new_x[choice[1],:,:] = x[i][1,:,:]
            new_x[choice[2],:,:] = x[i][2,:,:]
            img_list.append(new_x)
        return img_list,y

def random_lighting(x, y=None, mode='classification'):
    return RandomLighting()(x, y, mode=mode)

class RandomExpand(ProcessBase):
    def __init__(self):
        super(RandomExpand, self).__init__()

    def _transform_classification(self, x, y):
        # assert len(x.shape) == 4
        n = len(x)
        img_list = []
        for i in range(n):
            c,h,w = x[i].shape
            ratio = np.random.uniform(1,4)
            left = np.random.uniform(0, w*ratio - w )
            top = np.random.uniform(0, h*ratio - h )
            expand_image = np.zeros((c,int(h*ratio),int(w*ratio)),dtype=x[i].dtype)
            expand_image[:,:,:] = np.mean(x[i])
            expand_image[:,int(top):int(top+h),
                        int(left):int(left+w)] = x[i]
            img_list.append(expand_image)
        return img_list,y

    def _transform_detection(self, x, y):
        # assert len(x.shape) == 4
        n = len(x)
        img_list = []
        new_y = []
        for i in range(n):
            ny = []
            c,h,w = x[i].shape
            ratio = np.random.uniform(1,4)
            left = np.random.uniform(0, w*ratio - w )
            top = np.random.uniform(0, h*ratio - h )
            expand_image = np.zeros((c,int(h*ratio),int(w*ratio)),dtype=x[i].dtype)
            expand_image[:,:,:] = np.mean(x[i])
            expand_image[:,int(top):int(top+h),
                        int(left):int(left+w)] = x[i]
            img_list.append(expand_image)
            for j, obj in enumerate(y[i]):
                ox,oy,ow,oh = obj["box"]
                ny.append({
                    "box": [ox+int(left), oy+int(top),ow,oh],
                    **{k: v for k, v in obj.items() if k != 'box'}
                })
            new_y.append(ny)
        return img_list,new_y

    def _transform_segmentation(self, x, y):
        # assert len(x.shape) == 4
        n = len(x)
        img_list = []
        new_y = []
        for i in range(n):
            c,h,w = x[i].shape
            c2,h2,w2 = y[i].shape
            ratio = np.random.uniform(1,4)
            left = np.random.uniform(0, w*ratio - w )
            top = np.random.uniform(0, h*ratio - h )
            expand_image = np.zeros((c,int(h*ratio),int(w*ratio)),dtype=x[i].dtype)
            expand_label = np.zeros((c2,int(h2*ratio),int(w2*ratio)),dtype=y[i].dtype)
            expand_image[:,:,:] = np.mean(x[i])
            # expand_label[:,:,:] = np.mean(y[i])
            expand_image[:,int(top):int(top+h),
                        int(left):int(left+w)] = x[i]

            expand_label[:,int(top):int(top+h),
                        int(left):int(left+w)] = y[i]
            img_list.append(expand_image)
            new_y.append(expand_label)
        return img_list,new_y

def random_expand(x, y=None, mode='classification'):
    return RandomExpand()(x, y, mode=mode)

class Shear(ProcessBase):
    def __init__(self, max_shear_factor=5):
        super(Shear, self).__init__()
        self.max_shear_factor=max_shear_factor
        self.choice_list = [0,1]

    def _transform_classification(self, x, y):
        n = len(x)
        new_x = []
        for i in range(n):
            c,h,w = x[i].shape
            angle_to_shear = int(np.random.uniform(-self.max_shear_factor,self.max_shear_factor))
            angle = np.tan(np.radians(angle_to_shear))
            choice =np.random.choice(self.choice_list)

            if choice == 0:
                # print('return original')
                new_x.append(x[i])
            elif choice == 1:
                # print('x axis')
                channel_last = x[i].transpose(1,2,0)
                img = Image.fromarray(np.uint8(channel_last))
                shift_in_pixels = angle * h
                if shift_in_pixels > 0:
                    shift_in_pixels = np.ceil(shift_in_pixels)
                else:
                    shift_in_pixels = np.floor(shift_in_pixels)
                offset = shift_in_pixels
                if angle_to_shear <= 0:
                    shift_in_pixels = abs(shift_in_pixels)
                    offset = 0
                    angle = abs(angle) * -1
                transform_matrix = (1, angle, -offset,0, 1, 0)
                img = img.transform((int(round(w + shift_in_pixels)), h),
                                        Image.AFFINE,
                                        transform_matrix,
                                        Image.BICUBIC)
                new_x.append(np.asarray(img).transpose(2,0,1))
            else:
                # print('y axis')
                channel_last = x[i].transpose(1,2,0)
                img = Image.fromarray(np.uint8(channel_last))
                shift_in_pixels = angle * w
                offset = shift_in_pixels
                if angle_to_shear <= 0:
                    shift_in_pixels = abs(shift_in_pixels)
                    offset = 0
                    angle = abs(angle) * -1

                transform_matrix = (1, 0, 0,angle, 1, -offset)

                img = img.transform((w, int(round(h + shift_in_pixels))),
                                        Image.AFFINE,
                                        transform_matrix,
                                        Image.BICUBIC)

                new_x.append(np.asarray(img).transpose(2,0,1))
        return new_x,y

    def _transform_detection(self, x, y):
        n = len(x)
        new_x = []
        new_y = []
        for i in range(n):
            c,h,w = x[i].shape
            img,label = x[i],y[i]
            angle_to_shear = int(np.random.uniform(-self.max_shear_factor,self.max_shear_factor))
            angle = np.tan(np.radians(angle_to_shear))
            choice =np.random.choice(self.choice_list)
            if choice == 0:
                # print('return original')
                new_x.append(img)
                new_y.append(label)
            elif choice == 1:
                # print('x axis')
                ny=[]
                shift_in_pixels = angle * h
                if shift_in_pixels > 0:
                    shift_in_pixels = np.ceil(shift_in_pixels)
                else:
                    shift_in_pixels = np.floor(shift_in_pixels)
                offset = shift_in_pixels
                if angle_to_shear <= 0:
                    shift_in_pixels = abs(shift_in_pixels)
                    offset = 0
                    angle = abs(angle) * -1.
                else:
                    tmp1, tmp2 = [],[]
                    tmp1.append(img)
                    tmp2.append(label)
                    img,label = horizontalflip(tmp1,tmp2,prob=False,mode='detection')
                    img = np.asarray(img[0])
                    label = label[0]
                channel_last = img.transpose(1,2,0)
                img = Image.fromarray(np.uint8(channel_last))
                transform_matrix = (1, angle, -offset,0, 1, 0)
                img = img.transform((int(round(w + shift_in_pixels)), h),
                                        Image.AFFINE,
                                        transform_matrix,
                                        Image.BICUBIC)

                for j, obj in enumerate(label):
                    ox,oy,ow,oh = obj["box"]
                    if angle == 0:
                        px,pw =  ox,ow
                    else:
                        px1 = ox-(ow/2) + (((oy-(oh/2))*abs(angle))).astype(int)
                        px2 = ox+(ow/2) + (((oy+(oh/2))*abs(angle))).astype(int)
                        pw = px2-px1
                        px = px1 + pw/2
                    ny.append({
                        "box": [px-(offset*angle), oy, pw, oh],
                        **{k: v for k, v in obj.items() if k != 'box'}
                    })
                if angle_to_shear > 0:
                    img = np.asarray(img).transpose(2,0,1)
                    tmp1, tmp2 = [],[]
                    tmp1.append(img)
                    tmp2.append(ny)
                    img,label = horizontalflip(tmp1,tmp2,prob=False,mode='detection')
                    new_x.append(np.asarray(img[0]))
                    new_y.append(label[0])
                else:
                    new_x.append(np.asarray(img).transpose(2,0,1))
                    new_y.append(ny)

        return new_x,new_y

    def _transform_segmentation(self, x, y):
        n = len(x)
        new_x = []
        new_y = []
        for i in range(n):
            c,h,w = x[i].shape
            angle_to_shear = int(np.random.uniform(-self.max_shear_factor,self.max_shear_factor))
            angle = np.tan(np.radians(angle_to_shear))
            choice =np.random.choice(self.choice_list)

            if choice == 0:
                # print('return original')
                new_x.append(x[i])
                new_y.append(y[i])
            elif choice == 1:
                # print('x axis')
                channel_last = x[i].transpose(1,2,0)
                img = Image.fromarray(np.uint8(channel_last))
                shift_in_pixels = angle * h
                if shift_in_pixels > 0:
                    shift_in_pixels = np.ceil(shift_in_pixels)
                else:
                    shift_in_pixels = np.floor(shift_in_pixels)
                offset = shift_in_pixels
                if angle_to_shear <= 0:
                    shift_in_pixels = abs(shift_in_pixels)
                    offset = 0
                    angle = abs(angle) * -1
                transform_matrix = (1, angle, -offset,0, 1, 0)
                img = img.transform((int(round(w + shift_in_pixels)), h),
                                        Image.AFFINE,
                                        transform_matrix,
                                        Image.BICUBIC)
                new_x.append(np.asarray(img).transpose(2,0,1))

                num_class,_,_ = y[i].shape
                new_label = []
                for z in range(num_class):
                        label = y[i][z,:,:]
                        img = Image.fromarray(np.uint8(label))
                        img = img.transform((int(round(w + shift_in_pixels)), h),
                                                Image.AFFINE,
                                                transform_matrix,
                                                Image.BICUBIC)
                        new_label.append(np.array(img))
                new_y.append(np.array(new_label))
            else:
                # print('y axis')
                channel_last = x[i].transpose(1,2,0)
                img = Image.fromarray(np.uint8(channel_last))
                shift_in_pixels = angle * w
                offset = shift_in_pixels
                if angle_to_shear <= 0:
                    shift_in_pixels = abs(shift_in_pixels)
                    offset = 0
                    angle = abs(angle) * -1

                transform_matrix = (1, 0, 0,angle, 1, -offset)

                img = img.transform((w, int(round(h + shift_in_pixels))),
                                        Image.AFFINE,
                                        transform_matrix,
                                        Image.BICUBIC)

                new_x.append(np.asarray(img).transpose(2,0,1))

                num_class,_,_ = y[i].shape
                new_label = []
                for z in range(num_class):
                        label = y[i][z,:,:]
                        img = Image.fromarray(np.uint8(label))
                        img = img.transform((int(round(w + shift_in_pixels)), h),
                                                Image.AFFINE,
                                                transform_matrix,
                                                Image.BICUBIC)
                        new_label.append(np.array(img))
                new_y.append(np.array(new_label))

        return new_x,new_y


def shear(x, y=None, mode='classification'):
    return Shear(max_shear_factor=20)(x, y, mode=mode)
