import os
import numpy as np
import renom as rm
from tqdm import tqdm
from PIL import Image

from renom_img import __version__
from renom_img.api import Base, adddoc
from renom_img.api.cnn.yolo_v1 import CnnYolov1
from renom_img.api.detection import Detection
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.misc.download import download
from renom_img.api.utility.optimizer import OptimizerYolov1
from renom_img.api.utility.box import transform2xy12
from renom_img.api.utility.load import prepare_detection_data, load_img, resize_detection_data
from renom_img.api.utility.exceptions.check_exceptions import *
from renom_img.api.utility.exceptions.exceptions import WeightLoadError


def make_box(box):
    x1 = box[:, :, :, 0] - box[:, :, :, 2] / 2.
    y1 = box[:, :, :, 1] - box[:, :, :, 3] / 2.
    x2 = box[:, :, :, 0] + box[:, :, :, 2] / 2.
    y2 = box[:, :, :, 1] + box[:, :, :, 3] / 2.
    return [x1, y1, x2, y2]


def calc_iou(box1, box2):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2
    area1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    xA = np.fmax(b1_x1, b2_x1)
    yA = np.fmax(b1_y1, b2_y1)
    xB = np.fmin(b1_x2, b2_x2)
    yB = np.fmin(b1_y2, b2_y2)
    intersect = (xB - xA) * (yB - yA)
    # case we are given two scalar boxes:
    if intersect.shape == ():
        if (xB < xA) or (yB < yA):
            return 0
    # case we are given an array of boxes:
    else:
        intersect[xB < xA] = 0.0
        intersect[yB < yA] = 0.0
    # 0.0001 to avoid dividing by zero
    union = (area1 + area2 - intersect + 0.0001)
    return intersect / union


class TargetBuilderYolov1():
    '''
    Target Builder for Yolov1.

    Args:
        class_map:
        cell:
        bbox:
        imsize:

    '''

    def __init__(self, class_map, cell, bbox, imsize):
        self.class_map = class_map
        self.cells = cell
        self.bbox = bbox
        self.imsize = imsize

    def __call__(self, *args, **kwargs):
        return self.build(*args, **kwargs)

    def preprocess(self, x):
        """Image preprocess for Yolov1.

        :math:`x_{new} = x*2/255 - 1`

        Args:
            x (ndarray):

        Returns:
            (ndarray): Preprocessed data.
        """
        return x / 255.

    def build(self, img_path_list, annotation_list=None, augmentation=None, **kwargs):
        check_missing_param(self.class_map)
        if annotation_list is None:
            img_array = np.vstack([load_img(path, self.imsize)[None]
                                   for path in img_path_list])
            img_array = self.preprocess(img_array)
            return img_array

        N = len(img_path_list)
        num_class = len(self.class_map)
        num_bbox = self.bbox
        cell_w, cell_h = self.cells
        target = np.zeros((N, cell_w, cell_h, 5 * num_bbox + num_class))

        img_data, label_data = prepare_detection_data(img_path_list,
                                                      annotation_list)

        if augmentation is not None:
            img_data, label_data = augmentation(img_data, label_data, mode="detection")

        img_data, label_data = resize_detection_data(img_data, label_data, self.imsize)

        # Create target.
        img_w, img_h = self.imsize
        for n in range(N):
            for obj in label_data[n]:
                tx = np.clip(obj["box"][0], 0, img_w) * .99 * cell_w / img_w
                ty = np.clip(obj["box"][1], 0, img_h) * .99 * cell_h / img_h
                tw = np.sqrt(np.clip(obj["box"][2], 0, img_w) / img_w)
                th = np.sqrt(np.clip(obj["box"][3], 0, img_h) / img_h)
                one_hot = [0] * obj["class"] + [1] + [0] * (num_class - obj["class"] - 1)
                target[n, int(ty), int(tx)] = \
                    np.concatenate(([1, tx % 1, ty % 1, tw, th] * num_bbox, one_hot))

        return self.preprocess(img_data), target.reshape(N, -1)


class Yolov1(Detection):
    """ Yolov1 object detection algorithm.

    Args:
        class_map (list, dict): List of class names.
        cells (int or tuple): Cell size.
        bbox (int): Number of boxes.
        imsize (int, tuple): Image size.
        load_pretrained_weight (bool, str): If True, pretrained weights will be
          downloaded to the current directory and loaded as the initial weight values.
          If a string is given, weight values will be loaded and initialized
          from the weights in the given file name.
        train_whole_network (bool): Flag specifying whether to freeze or train
          the base layers of the model during training. If True, trains all layers
          of the model. If False, the convolutional base is frozen during training.

    Example:
        >>> from renom_img.api.detection.yolo_v1 import Yolov1
        >>> from renom_img.api.utility.load import parse_xml_detection
        >>>
        >>> train_label_path_list = ...  # Provide list of training label paths
        >>> annotation_list, class_map = parse_xml_detection(train_label_path_list)
        >>>
        >>> model = Yolov1(class_map, cells=7, bbox=2, imsize=(224,224), load_pretrained_weight=True, train_whole_network=True)

    References:
        | Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
        | **You Only Look Once: Unified, Real-Time Object Detection**
        | https://arxiv.org/abs/1506.02640
        |

    """

    # Attributes here will be saved to hdf5 file.
    # So if an algorithm has its own hyper parameter, you need to put it here.
    SERIALIZED = ("_cells", "_bbox", *Base.SERIALIZED)
    WEIGHT_URL = CnnYolov1.WEIGHT_URL

    def __init__(self, class_map=None, cells=7, bbox=2, imsize=(224, 224), load_pretrained_weight=False, train_whole_network=False):
        # exceptions checking
        check_yolov1_init(cells, bbox)
        if not hasattr(cells, "__getitem__"):
            cells = (cells, cells)

        self._model = CnnYolov1()
        super(Yolov1, self).__init__(class_map, imsize,
                                     load_pretrained_weight, train_whole_network, self._model)
        self._cells = cells
        self._bbox = bbox
        self._model.set_output_size((self.num_class + 5 * bbox) * cells[0] * cells[1])
        self._model.set_train_whole(train_whole_network)
        self.default_optimizer = OptimizerYolov1()
        self.decay_rate = 0.0005

    def forward(self, x):
        check_missing_param(self.class_map)
        self._model.set_output_size((self.num_class + 5 * self._bbox)
                                    * self._cells[0] * self._cells[1])
        return self._model(x)

    def load(self, filename):
        """Load saved weights to model.

        Args:
            filename (str): File name of saved model.

        Example:
            >>> model = rm.Dense(2)
            >>> model.load("model.hd5")
        """
        import h5py
        f = h5py.File(filename, 'r+')
        values = f['values']
        types = f['types']

        names = sorted(values.keys())

        try:
            self._try_load(names, values, types)
        except AttributeError as e:
            try:
                names, values, types = self._mapping(names, values, types)
                self._try_load(names, values, types)
            except Exception as e:
                raise WeightLoadError('The {} weight file can not be loaded into the {} model.'.format(
                    filename, self.__class__.__name__))

    def _mapping(self, names, values, types):
        for name in names:
            if "._network" in name:
                values[name.replace("._network", "._model.classifier")] = values.pop(name)
                types[name.replace("._network", "._model.classifier")] = types.pop(name)
            elif "._freezed_network" in name:
                values[name.replace("._freezed_network",
                                    "._model.feature_extractor")] = values.pop(name)
                types[name.replace("._freezed_network", "._model.feature_extractor")
                      ] = types.pop(name)

        names = [n.replace("._network", "._model.classifier") for n in names]
        names = [n.replace("._freezed_network", "._model.feature_extractor") for n in names]

        return sorted(names), values, types

    def _try_load(self, names, values, types):

        def get_attr(root, names):
            names = names.split('.')[1:]
            ret = root
            for name in names:
                ret = getattr(ret, name)
            return ret

        target = self
        for name in names:
            target = get_attr(self, name)

            values_grp = values[name]
            types_grp = types[name]

            for k, v in values_grp.items():
                v = v.value
                if isinstance(v, np.ndarray):
                    type = types_grp.get(k, None)
                    if type:
                        if type.value == 'renom.Variable':
                            auto_update = types_grp[k + '._auto_update'].value
                            v = rm.Variable(v, auto_update=auto_update)
                        else:
                            v = rm.Node(v)

                if k.startswith('__dict__.'):
                    obj = target
                    name = k.split(".", 1)[1]
                else:
                    obj = target.params
                    name = k

                setattr(obj, name, v)

    def get_bbox(self, z, score_threshold=0.3, nms_threshold=0.4):
        """
        Calculates the bounding box location, size and class information for model predictions.

        Args:
            z (ndarray): Output array of neural network.
            score_threshold (float): The threshold for confidence score.
                                     Predicted boxes which have a lower confidence score than the threshold are discarded.
                                     The default is 0.3.
            nms_threshold (float): The threshold for non-maximum supression. The default is 0.4.

        Return:
            (list) : List of predicted bbox, score and class for each image.
            The format of the return value is shown below. Box coordinates and size will be returned as
            ratios to the original image size. Therefore, the values of 'box' are in the range [0 ~ 1].

        .. code-block :: python

            # An example of a return value.
            [
                [ # Prediction for first image.
                    {'box': [x, y, w, h], 'score':(float), 'class':(int), 'name':(str)},
                    {'box': [x, y, w, h], 'score':(float), 'class':(int), 'name':(str)},
                    ...
                ],
                [ # Prediction for second image.
                    {'box': [x, y, w, h], 'score':(float), 'class':(int), 'name':(str)},
                    {'box': [x, y, w, h], 'score':(float), 'class':(int), 'name':(str)},
                    ...
                ],
                ...
            ]

        Example:
            >>> z = model(x)
            >>> model.get_bbox(z)
            [[{'box': [0.21, 0.44, 0.11, 0.32], 'score':0.823, 'class':1, 'name':'dog'}],
             [{'box': [0.87, 0.38, 0.84, 0.22], 'score':0.423, 'class':0, 'name':'cat'}]]

        Note:
            Box coordinates and size will be returned as ratios to the original image size.
            Therefore, the values of 'box' are in the range [0 ~ 1].


        """
        if hasattr(z, 'as_ndarray'):
            z = z.as_ndarray()

        N = len(z)
        cell = self._cells[0]
        bbox = self._bbox
        probs = np.zeros((N, cell, cell, bbox, self.num_class))
        boxes = np.zeros((N, cell, cell, bbox, 4))
        yolo_format_out = z.reshape(
            N, cell, cell, bbox * 5 + self.num_class)
        offset = np.vstack([np.arange(cell) for c in range(cell)])

        for b in range(bbox):
            prob = yolo_format_out[:, :, :, b * 5][..., None] * yolo_format_out[:, :, :, bbox * 5:]
            probs[:, :, :, b, :] = prob
            boxes[:, :, :, b, :] = yolo_format_out[:, :, :, b * 5 + 1:b * 5 + 5]
            boxes[:, :, :, b, 0] += offset
            boxes[:, :, :, b, 1] += offset.T
            # because the output for width and height is square rooted
            boxes[:, :, :, b, 2] = boxes[:, :, :, b, 2]**2
            boxes[:, :, :, b, 3] = boxes[:, :, :, b, 3]**2
        boxes[:, :, :, :, 0:2] /= float(cell)

        # Clip bounding box.
        w = boxes[:, :, :, :, 2] / 2.
        h = boxes[:, :, :, :, 3] / 2.
        x1 = np.clip(boxes[:, :, :, :, 0] - w, 0, 1)
        y1 = np.clip(boxes[:, :, :, :, 1] - h, 0, 1)
        x2 = np.clip(boxes[:, :, :, :, 0] + w, 0, 1)
        y2 = np.clip(boxes[:, :, :, :, 1] + h, 0, 1)
        boxes[:, :, :, :, 2] = x2 - x1
        boxes[:, :, :, :, 3] = y2 - y1
        boxes[:, :, :, :, 0] = x1 + boxes[:, :, :, :, 2] / 2.
        boxes[:, :, :, :, 1] = y1 + boxes[:, :, :, :, 3] / 2.

        probs = probs.reshape(N, -1, self.num_class)
        boxes = boxes.reshape(N, -1, 4)

        probs[probs < score_threshold] = 0
        # Perform NMS

        argsort = np.argsort(probs, axis=1)[:, ::-1]
        for n in range(N):
            for cl in range(self.num_class):
                for b in range(len(boxes[n])):
                    if probs[n, argsort[n, b, cl], cl] == 0:
                        continue
                    b1 = transform2xy12(boxes[n, argsort[n, b, cl], :])
                    for comp in range(b + 1, len(boxes[n])):
                        b2 = transform2xy12(boxes[n, argsort[n, comp, cl], :])
                        if calc_iou(b1, b2) > nms_threshold:
                            probs[n, argsort[n, comp, cl], cl] = 0

        result = [[] for _ in range(N)]
        max_class = np.argmax(probs, axis=2)
        max_probs = np.clip(np.max(probs, axis=2), 0, 1)
        indexes = np.nonzero(max_probs)
        for i in range(len(indexes[0])):
            # Note: Take care types.
            result[indexes[0][i]].append({
                "class": int(max_class[indexes[0][i], indexes[1][i]]),
                "name": self.class_map[int(max_class[indexes[0][i], indexes[1][i]])].decode('ascii'),
                "box": boxes[indexes[0][i], indexes[1][i]].astype(np.float64).tolist(),
                "score": float(max_probs[indexes[0][i], indexes[1][i]])
            })
        return result

    def build_data(self):
        """
        This function returns a function which creates input data and target data
        specified for Yolov1.

        Returns:
            (function): Returns function which creates input data and target data.

        Example:
            >>> builder = model.build_data()  # This will return function.
            >>> x, y = builder(image_path_list, annotation_list)
            >>> z = model(x)
            >>> loss = model.loss(z, y)
        """

        # This interface is required to use builder with multiprocessing module.
        # We need to define function as top level.
        # Please pass arguments to builder through object attribute.

        # There are some cases that we want to change builder parameter in each epoch (ex Yolov2.
        # For this reason, there is a chance to modify builder in this function.
        return TargetBuilderYolov1(self.class_map, self._cells, self._bbox, self.imsize)

    def loss(self, x, y):
        """Loss function specified for yolov1.

        Args:
            x(Node, ndarray): Output data of neural network.
            y(Node, ndarray): Target data.

        Returns:
            (Node): Loss between x and y.

        Example:
            >>> z = model(x)
            >>> model.loss(z, y)
        """
        N = len(x)
        nd_x = x.as_ndarray()
        num_bbox = self._bbox
        target = y.reshape(N, self._cells[0], self._cells[1], 5 * num_bbox + self.num_class)
        mask = np.ones_like(target)
        nd_x = nd_x.reshape(target.shape)

        target_box = make_box(target[:, :, :, 1:5])
        iou = np.zeros((*target.shape[:3], num_bbox))
        no_obj_flag = np.where(target[:, :, :, 0] == 0)
        obj_flag = np.where(target[:, :, :, 0] == 1)

        mask[no_obj_flag[0], no_obj_flag[1], no_obj_flag[2], :] = 0
        for b in range(num_bbox):
            # No obj
            mask[no_obj_flag[0], no_obj_flag[1], no_obj_flag[2], 5 * b] = 0.5

            # Search best iou target. 1:5, 6:10
            predicted_box = make_box(nd_x[:, :, :, 1 + b * 5:(b + 1) * 5])
            iou[:, :, :, b] = calc_iou(predicted_box, target_box)
        iou_ind = np.argmax(iou, axis=3)
        # Obj
        for fn, fy, fx in zip(*obj_flag):
            mask[fn, fy, fx, :5 * num_bbox] = 0
            mask[fn, fy, fx, iou_ind[fn, fy, fx] * 5] = 1
            mask[fn, fy, fx, 1 + iou_ind[fn, fy, fx] * 5:(iou_ind[fn, fy, fx] + 1) * 5] = 5

        diff = (x - y)
        return rm.sum(diff * diff * mask.reshape(N, -1)) / N / 2.
