import os
import numpy as np
import renom as rm
from tqdm import tqdm
from PIL import Image

from renom_img import __version__
from renom_img.api import Base, adddoc
from renom_img.api.detection import Detection
from renom_img.api.classification.darknet import Darknet
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.misc.download import download
from renom_img.api.utility.box import transform2xy12
from renom_img.api.utility.load import prepare_detection_data, load_img


def make_box(box):
    x1 = box[0] - box[2] / 2.
    y1 = box[1] - box[3] / 2.
    x2 = box[0] + box[2] / 2.
    y2 = box[1] + box[3] / 2.
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

    if (xB - xA) < 0 or (yB - yA) < 0:
        return 0
    intersect = (xB - xA) * (yB - yA)
    union = (area1 + area2 - intersect)

    return intersect / union


def calc_rmse(box1, box2):

    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2

    rmse = np.sqrt(((b1_x1 - b2_x1)**2 + (b1_y1 - b2_y1)**2 +
                    (b1_x2 - b2_x2)**2 + (b1_y2 - b2_y2)**2))

    return rmse


class Yolov1(Detection):
    """ Yolo object detection algorithm.

    Args:
        num_class (int): Number of class.
        cells (int or tuple): Cell size.
        boxes (int): Number of boxes.
        imsize (int, tuple): Image size.
        load_pretrained_weight (bool, str): If true, pretrained weight will be
          downloaded to current directory. If string is given, pretrained weight
          will be saved as given name.

    References:
        | Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
        | **You Only Look Once: Unified, Real-Time Object Detection**
        | https://arxiv.org/abs/1506.02640
        |

    """

    SERIALIZED = ("_cells", "_bbox", *Base.SERIALIZED)
    WEIGHT_URL = Darknet.WEIGHT_URL

    def __init__(self, class_map=None, cells=7, bbox=2, imsize=(224, 224), load_pretrained_weight=False, train_whole_network=False):

        if not hasattr(cells, "__getitem__"):
            cells = (cells, cells)

        self._cells = cells
        self._bbox = bbox
        model = Darknet()
        super(Yolov1, self).__init__(class_map, imsize,
                                     load_pretrained_weight, train_whole_network, model)

        self._last_dense_size = (self.num_class + 5 * bbox) * cells[0] * cells[1]
        self._freezed_network = rm.Sequential(model[:-4])
        self._network = rm.Sequential([
            rm.Conv2d(channel=1024, filter=3, padding=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, padding=1, stride=2, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, padding=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, padding=1, ignore_bias=True),
            rm.BatchNormalize(mode='feature'),
            rm.LeakyRelu(slope=0.1),
            rm.Flatten(),
            rm.Dense(4096),  # instead of locally connected layer, we are using Dense layer
            rm.LeakyRelu(slope=0.1),
            rm.Dropout(0.5),
            rm.Dense(self._last_dense_size)
        ])
        self._opt = rm.Sgd(0.0005, 0.9)

    def set_last_layer_unit(self, unit_size):
        pass

    def get_optimizer(self, current_loss=None, current_epoch=None, total_epoch=None, current_batch=None, total_batch=None, avg_valid_loss_list=None):
        """Returns an instance of Optimizer for training Yolov1 algorithm.

        If all argument(current_epoch, total_epoch, current_batch, total_batch) are given,
        an optimizer object which whose learning rate is modified according to the
        number of training iteration. Otherwise, constant learning rate is set.

        Args:
            current_epoch (int): The number of current epoch.
            total_epoch (int): The number of total epoch.
            current_batch (int): The number of current batch.
            total_batch (int): The number of total batch.

        Returns:
            (Optimizer): Optimizer object.
        """
        if any([num is None for num in
                [current_loss, current_epoch, total_epoch, current_batch, total_batch]]):
            return self._opt
        else:
            if current_loss is not None and current_loss > 50:
                self._opt._lr *= 0.1
                return self._opt

            ind1 = int(total_epoch * total_batch * 0.005)
            ind2 = int(total_epoch * total_batch * 0.012)
            ind3 = int(total_epoch * total_batch * 0.017)
            ind4 = int(total_epoch * total_batch * 0.55)
            ind5 = (total_epoch * total_batch) - (ind1 + ind2 + ind3 + ind4)

            lr_list = [0.0005] * ind1 + [0.00125] * ind2 + \
                [0.0025] * ind3 + [0.005] * ind4 + [0.0005] * ind5

            lr = lr_list[(current_epoch * total_batch) + current_batch]
            self._opt._lr = lr

            return self._opt

    def preprocess(self, x):
        """Image preprocess for Yolov1.

        :math:`x_{new} = x*2/255 - 1`

        Args:
            x (ndarray):

        Returns:
            (ndarray): Preprocessed data.
        """
        return x / 255.

    def forward(self, x):
        """Performs forward propagation.
        This function can be called using ``__call__`` method.
        See following example of method usage.

        Args:
            x (ndarray, Node): Input image as an tensor.

        Returns:
            (Node): Returns raw output of yolo v1.
            You can reform it to bounding box form using the method ``get_bbox``.

        Example:
            >>> import numpy as np
            >>> from renom_img.api.detection.yolo_v1 import Yolov1
            >>>
            >>> x = np.random.rand(1, 3, 224, 224)
            >>> class_map = ["dog", "cat"]
            >>> model = Yolov1(class_map)
            >>> y = model.forward(x) # Forward propagation.
            >>> y = model(x)  # Same as above result.
            >>>
            >>> bbox = model.get_bbox(y) # The output can be reformed using get_bbox method.

        """
        assert len(self.class_map) > 0, \
            "Class map is empty. Please set the attribute class_map when instantiate model class. " +\
            "Or, please load already trained model using the method 'load()'."
        self._freezed_network.set_auto_update(self.train_whole_network)
        out = self._freezed_network(x)
        out = self._network(out)
        return out

    def regularize(self):
        """Regularize term. You can use this function to add regularize term to
        loss function.

        In Yolo v1, weight decay of 0.0005 will be added.

        Example:
            >>> import numpy as np
            >>> from renom_img.api.detection.yolo_v1 import Yolov1
            >>> x = np.random.rand(1, 3, 224, 224)
            >>> y = np.random.rand(1, (5*2+20)*7*7)
            >>> model = Yolov1()
            >>> loss = model.loss(x, y)
            >>> reg_loss = loss + model.regularize() # Adding weight decay term.
        """

        reg = 0
        for layer in self.iter_models():
            if hasattr(layer, "params") and hasattr(layer.params, "w") and isinstance(layer, rm.Conv2d):
                reg += rm.sum(layer.params.w * layer.params.w)
        return (0.0005 / 2) * reg

    def get_bbox(self, z, score_threshold=0.3, nms_threshold=0.4):
        """

        Args:
            z (ndarray): Output array of neural network. The shape of array
            score_threshold (float): The threshold for confidence score.
                                     Predicted boxes which have lower confidence score than the threshold are discarderd.
                                     Defaults to 0.3
            nms_threshold (float): The threshold for non maximum supression. Defaults to 0.4

        Return:
            (list) : List of predicted bbox, score and class of each image.
            The format of return value is bellow. Box coordinates and size will be returned as
            ratio to the original image size. Therefore the range of 'box' is [0 ~ 1].

        .. code-block :: python

            # An example of return value.
            [
                [ # Prediction of first image.
                    {'box': [x, y, w, h], 'score':(float), 'class':(int), 'name':(str)},
                    {'box': [x, y, w, h], 'score':(float), 'class':(int), 'name':(str)},
                    ...
                ],
                [ # Prediction of second image.
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
            Box coordinate and size will be returned as ratio to the original image size.
            Therefore the range of 'box' is [0 ~ 1].


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
        max_probs = np.max(probs, axis=2)
        indexes = np.nonzero(np.clip(max_probs, 0, 1))
        for i in range(len(indexes[0])):
            # Note: Take care types.
            result[indexes[0][i]].append({
                "class": int(max_class[indexes[0][i], indexes[1][i]]),
                "name": self.class_map[int(max_class[indexes[0][i], indexes[1][i]])].decode("utf-8"),
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

        def builder(img_path_list, annotation_list, augmentation=None, **kwargs):
            """
            Args:
                x: Image path list.
                y: Detection formatted label.
            """
            N = len(img_path_list)
            num_bbox = self._bbox
            cell_w, cell_h = self._cells
            target = np.zeros((N, self._cells[0], self._cells[1], 5 * num_bbox + self.num_class))

            img_data, label_data = prepare_detection_data(img_path_list,
                                                          annotation_list, self.imsize)

            if augmentation is not None:
                img_data, label_data = augmentation(img_data, label_data, mode="detection")

            # Create target.
            cell_w, cell_h = self._cells
            img_w, img_h = self.imsize
            for n in range(N):
                for obj in label_data[n]:
                    tx = np.clip(obj["box"][0], 0, img_w) * .99 * cell_w / img_w
                    ty = np.clip(obj["box"][1], 0, img_h) * .99 * cell_h / img_h
                    tw = np.sqrt(np.clip(obj["box"][2], 0, img_w) / img_w)
                    th = np.sqrt(np.clip(obj["box"][3], 0, img_h) / img_h)
                    one_hot = [0] * obj["class"] + [1] + [0] * (self.num_class - obj["class"] - 1)
                    target[n, int(ty), int(tx)] = \
                        np.concatenate(([1, tx % 1, ty % 1, tw, th] * num_bbox, one_hot))

            return self.preprocess(img_data), target.reshape(N, -1)
        return builder

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
        mask = np.zeros_like(target)
        nd_x = nd_x.reshape(target.shape)
        x = x.reshape(target.shape)
        for i in range(N):
            for j in range(self._cells[0]):
                for k in range(self._cells[1]):
                    is_obj = target[i, j, k, 0]
                    for b in range(num_bbox):
                        mask[i, j, k, b * 5] = 0.5  # mask for noobject cell
                    best_rmse = 20
                    best_iou = 0
                    best_index = num_bbox
                    if is_obj == 0:
                        continue
                    target_box = make_box(target[i, j, k, 1:5])
                    for b in range(num_bbox):
                        predicted_box = make_box(nd_x[i, j, k, 1 + b * 5:(b + 1) * 5])
                        iou = calc_iou(predicted_box, target_box)
                        rmse = calc_rmse(predicted_box, target_box)
                        if best_iou > 0 or iou > 0:
                            if iou > best_iou:
                                best_iou = iou
                                best_index = b
                        else:
                            if rmse < best_rmse:
                                best_rmse = rmse
                                best_index = b

                    predicted_box = make_box(nd_x[i, j, k, 1 + best_index * 5:(best_index + 1) * 5])
                    # IOU needed to be calculated again, cause best index can be selected based on rmse also
                    iou = calc_iou(predicted_box, target_box)
                    # mask for the confidence of selected box
                    mask[i, j, k, 5 * best_index] = 1
                    # changing the confidence of target to iou
                    target[i, j, k, 5 * best_index] = iou
                    # mask for the coordinates
                    mask[i, j, k, 1 + best_index * 5:(best_index + 1) * 5] = 5
                    # mask for the class probabilities
                    mask[i, j, k, 5 * num_bbox:] = 1

        diff = target - x

        return rm.sum(diff * diff * mask) / N / 2.
