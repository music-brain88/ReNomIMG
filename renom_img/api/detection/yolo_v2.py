import os
from itertools import chain
import numpy as np
import math
import renom as rm
import matplotlib.pyplot as plt
from renom.cuda import release_mem_pool, is_cuda_active
from tqdm import tqdm
from PIL import Image, ImageDraw
from renom_img import __version__
from renom_img.api import Base, adddoc
from renom_img.api.detection import Detection
from renom_img.api.cnn.yolo_v2 import CnnYolov2
from renom_img.api.utility.load import prepare_detection_data, load_img, resize_detection_data
from renom_img.api.utility.box import calc_iou_xywh, transform2xy12
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.misc.download import download
from renom_img.api.utility.nms import nms
from renom_img.api.utility.optimizer import BaseOptimizer, OptimizerYolov2
from renom_img.api.utility.exceptions.check_exceptions import *
from renom_img.api.utility.exceptions.exceptions import WeightLoadError, InvalidOptimizerError


class BestAnchorBoxFinder(object):
    def __init__(self, ANCHORS):
        self.anchors = [BoundBox(0, 0, ANCHORS[i][0], ANCHORS[i][1])
                        for i in range(len(ANCHORS))]

    def _interval_overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2, x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2, x4) - x3

    def bbox_iou(self, box1, box2):
        intersect_w = self._interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
        intersect_h = self._interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

        intersect = intersect_w * intersect_h

        w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
        w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

        union = w1 * h1 + w2 * h2 - intersect

        return float(intersect) / union

    def find(self, center_w, center_h):
        # find the anchor that best predicts this box
        best_anchor = -1
        max_iou = -1
        # each Anchor box is specialized to have a certain shape.
        # e.g., flat large rectangle, or small square
        shifted_box = BoundBox(0, 0, center_w, center_h)
        # For given object, find the best anchor box!
        for i in range(len(self.anchors)):  # run through each anchor box
            anchor = self.anchors[i]
            iou = self.bbox_iou(shifted_box, anchor)
            if max_iou < iou:
                best_anchor = i
                max_iou = iou
        return(best_anchor, max_iou)


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, confidence=None, classes=None):
        self.xmin, self.ymin = xmin, ymin
        self.xmax, self.ymax = xmax, ymax
        # the code below are used during inference
        # probability
        self.confidence = confidence
        # class probaiblities [c1, c2, .. cNclass]
        self.set_class(classes)

    def set_class(self, classes):
        self.classes = classes
        self.label = np.argmax(self.classes)

    def get_label(self):
        return(self.label)

    def get_score(self):
        return(self.classes[self.label])


class AnchorYolov2(object):
    """
    This class contains anchors that are used in Yolov2.

    Args:
      anchor(list): List of anchors.
      imsize(tuple): Image size.

    """

    def __init__(self, anchor, imsize):
        self.anchor = anchor
        self.imsize = imsize

    def __len__(self):
        return len(self.anchor)


def create_anchor(annotation_list, n_anchor=5, base_size=(320, 320)):
    """
    This function creates 'anchors' for the Yolov2 algorithm using k-means clustering.

    The following annotation list is required.

    This function performs K-means clustering using a custom metric.
    Only the anchor sizes are relevant, so coordinates do not need to be considered.

    Args:
        annotation_list (list): Annotation list.
        n_anchor(int): Number of anchors.

    Returns:
        (AnchorYolov2): Anchor list.
    """
# -------------------------------------------------------------------------
    annotations = []
    for annot in annotation_list:
        for obj in annot:
            aw = base_size[0]
            ah = base_size[1]
            w = obj["box"][2] / aw * int(aw / 32)  # make the width range between [0,GRID_W)
            h = obj["box"][3] / ah * int(ah / 32)  # make the width range between [0,GRID_H)
            temp = [w, h]
            annotations.append(temp)
    annotations = np.array(annotations)

    def iou(box, clusters):

        x = np.minimum(clusters[:, 0], box[0])
        y = np.minimum(clusters[:, 1], box[1])

        intersection = x * y
        box_area = box[0] * box[1]
        cluster_area = clusters[:, 0] * clusters[:, 1]

        iou_ = intersection / (box_area + cluster_area - intersection)

        return iou_

    def kmeans(boxes, k, dist=np.median, seed=1):

        rows = boxes.shape[0]

        distances = np.empty((rows, k))
        last_clusters = np.zeros((rows,))

        np.random.seed(seed)

        # initialize k cluster centers
        clusters = boxes[np.random.choice(rows, k, replace=False)]

        while True:
            # Step 1: allocate each item to the closest cluster centers
            for icluster in range(k):
                distances[:, icluster] = 1 - iou(clusters[icluster], boxes)

            nearest_clusters = np.argmin(distances, axis=1)

            if (last_clusters == nearest_clusters).all():
                break

            # Step 2: calculate the cluster centers as mean
            for cluster in range(k):
                clusters[cluster] = dist(boxes[nearest_clusters == cluster], axis=0)

            last_clusters = nearest_clusters

        return clusters, nearest_clusters, distances

    clusters, nearest_clusters, distances = kmeans(annotations, n_anchor, seed=2, dist=np.mean)

    WithinClusterMeanDist = np.mean(distances[np.arange(distances.shape[0]), nearest_clusters])
    return AnchorYolov2(clusters, base_size)


class TargetBuilderYolov2():
    """
    Target Builder for Yolov2.

    Args:
        class_map (list, dict): List of class names.
        size_n:
        perm:
        imsize_list: List of image sizes.
        anchor:
        num_anchor: Number of anchors.
    """

    def __init__(self, class_map, size_n, perm, imsize_list, anchor, num_anchor):
        self.class_map = class_map
        self.size_N = size_n
        self.perm = perm
        self.imsize_list = imsize_list
        self.num_class = len(class_map)
        self.anchor = anchor
        self.num_anchor = num_anchor
        self.bestAnchorBoxFinder = BestAnchorBoxFinder(self.anchor)
        self.buffer = 50

    def __call__(self, *args, **kwargs):
        return self.build(*args, **kwargs)

    def preprocess(self, x):
        return x / 255.

    def build(self, img_path_list, annotation_list=None, augmentation=None, nth=0, **kwargs):
        """
        These parameters will be given by distributor.
        Args:
            img_path_list (list): List of input image.
            annotation_list (list): List of detection annotation.
            augmentation (Augmentation): Augmentation object.
            nth (int): Current batch index.
        """
        check_missing_param(self.class_map)
        if annotation_list is None:
            img_array = np.vstack([load_img(path, self.imsize_list[0])[None]
                                   for path in img_path_list])
            img_array = self.preprocess(img_array)

            return img_array

        N = len(img_path_list)
        # This ratio is specific to Darknet19.
        ratio_w = 32.
        ratio_h = 32.
        img_list = []
        num_class = self.num_class
        channel = num_class + 5
        offset = channel
        if nth % 10 == 0:
            self.perm[:] = np.random.permutation(self.size_N)
        size_index = self.perm[0]
        buff_batch = np.zeros((N, 1, 4, 1, 1, self.buffer))
        label = np.zeros(
            (N, self.num_anchor, channel, self.imsize_list[size_index][1] // 32, self.imsize_list[size_index][0] // 32))
        img_list, label_list = prepare_detection_data(img_path_list, annotation_list)

        if augmentation is not None:
            img_list, label_list = augmentation(img_list, label_list, mode="detection")

        img_list, label_list = resize_detection_data(
            img_list, label_list, self.imsize_list[size_index])
        im_size = self.imsize_list[size_index]

        for n, annotation in enumerate(label_list):
            # This returns resized image.
            # Target processing
            true_box_index = 0
            for obj in annotation:
                center_x, center_y = obj['box'][0] / float(im_size[0]) * (
                    im_size[0] // ratio_w), obj['box'][1] / float(im_size[1]) * (im_size[1] // ratio_h)

                grid_x, grid_y = int(np.floor(center_x)), int(np.floor(center_y))

                center_w, center_h = obj['box'][2] / float(im_size[0]) * (
                    im_size[0] // ratio_w), obj['box'][3] / float(im_size[1]) * (im_size[1] // ratio_h)

                box = [center_x, center_y, center_w, center_h]
                best_anchor, max_iou = self.bestAnchorBoxFinder.find(center_w, center_h)
                classes = np.array([0] * obj["class"] + [1] + [0] * (num_class - obj["class"] - 1))
                label[n, best_anchor, 0, grid_y, grid_x] = 1
                label[n, best_anchor, 1:5, grid_y, grid_x] = box
                label[n, best_anchor, 5:, grid_y, grid_x] = classes
                buff_batch[n, 0, :, 0, 0, true_box_index] = box
                true_box_index += 1
                true_box_index = true_box_index % self.buffer

        return self.preprocess(img_list), buff_batch, label


class Yolov2(Detection):
    """
    Yolov2 object detection algorithm.

    Args:
        class_map (list, dict): List of class names.
        anchor (AnchorYolov2): Anchors.
        imsize (list): Image size(s).
            This can be either an image size ex):(320, 320) or list of image sizes
            ex):[(288, 288), (320, 320)]. If a list of image sizes is provided,
            the prediction method uses the last image size of the list for prediction.
        load_pretrained_weight (bool, str): Argument specifying whether or not to load pretrained weight values.
          If True, pretrained weights will be downloaded to the current directory and loaded as the initial weight values.
          If a string is given, weight values will be loaded and initialized from the weights in the given file name.
        train_whole_network (bool): Flag specifying whether to freeze or train the base layers of the model during training.
          If True, trains all layers of the model. If False, the convolutional base is frozen during training.

    Example:
        >>> from renom_img.api.detection.yolo_v2 import Yolov2, create_anchor
        >>> from renom_img.api.utility.load import parse_xml_detection
        >>>
        >>> train_label_path_list = ...  # provide list of paths to training data
        >>> annotation_list, class_map = parse_xml_detection(train_label_path_list)
        >>> my_anchor = create_anchor(annotation_list)
        >>>
        >>> model = Yolov2(class_map, anchor=my_anchor, imsize=(320,320), load_pretrained_weight=True, train_whole_network=True)

    References:
        | Joseph Redmon, Ali Farhadi
        | **YOLO9000: Better, Faster, Stronger**
        | https://arxiv.org/abs/1612.08242
        |

    Note:
        If you save this model using the 'save' method, anchor information (list of anchors and their base sizes) will be
        saved. Therefore, when you load your own saved model, you do not need to provide the 'anchor' and 'anchor_size' arguments.
    """

    # Anchor information will be serialized by 'save' method.
    SERIALIZED = ("anchor", "num_anchor", "anchor_size",  *Base.SERIALIZED)
    WEIGHT_URL = CnnYolov2.WEIGHT_URL

    def __init__(self, class_map=None, anchor=None,
                 imsize=(320, 320), load_pretrained_weight=False, train_whole_network=False):
        # Exceptions checking
        check_yolov2_init(imsize)

        self._model = CnnYolov2()
        super(Yolov2, self).__init__(class_map, imsize,
                                     load_pretrained_weight, train_whole_network, self._model)
        self.anchor = [] if not isinstance(anchor, AnchorYolov2) else anchor.anchor
        self.anchor_size = imsize if not isinstance(anchor, AnchorYolov2) else anchor.imsize
        self.num_anchor = 0 if anchor is None else len(anchor)
        self.default_optimizer = OptimizerYolov2()

        self._model.set_output_size((self.num_class + 5) * self.num_anchor,
                                    self.class_map, self.num_anchor)
        self._model.set_train_whole(train_whole_network)
        self.decay_rate = 0.0005

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
            if "._freezed_network" in name:
                values[name.replace("._freezed_network", "._model._base")] = values.pop(name)
                types[name.replace("._freezed_network", "._model._base")] = types.pop(name)
            elif "root." in name:
                values[name.replace("root.", "root._model.")] = values.pop(name)
                types[name.replace("root.", "root._model.")] = types.pop(name)

        names = [n.replace("root.", "root._model.") for n in names]
        names = [n.replace("._freezed_network", "._base") for n in names]

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

    def forward(self, x):
        """
        Performs forward propagation.
        You can call this function using the ``__call__`` method.

        Args:
            x(ndarray, Node): Input to ${class}.
        """
        check_yolov2_forward(self.anchor, x)
        self._model.set_anchor(self.num_anchor)
        return self._model(x)

    def get_bbox(self, z, score_threshold=0.3, nms_threshold=0.4):
        """
        Calculates the bounding box location, size and class information for model predictions.

        Example:
            >>> z = model(x)
            >>> model.get_bbox(z)
            [[{'box': [0.21, 0.44, 0.11, 0.32], 'score':0.823, 'class':1, 'name':'dog'}],
             [{'box': [0.87, 0.38, 0.84, 0.22], 'score':0.423, 'class':0, 'name':'cat'}]]

        Args:
            z (ndarray): Output array of neural network.

        Return:
            (list) : List of predicted bbox, score and class of each image.
            The format of the return value is shown below. Box coordinates and size will be returned as
            ratios to the original image size. Therefore, the values in 'box' are in the range [0 ~ 1].

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

        Note:
            Box coordinates and size will be returned as ratios to the original image size.
            Therefore, the values in 'box' are in the range [0 ~ 1].

        """

        if hasattr(z, 'as_ndarray'):
            z = z.as_ndarray()

        imsize = self.imsize
#        asw = imsize[0] / self.anchor_size[0]
#        ash = imsize[1] / self.anchor_size[1]
        anchor = [[an[0], an[1]] for an in self.anchor]

        num_anchor = len(anchor)
        N, C, H, W = z.shape
        offset = self.num_class + 5
        FW, FH = imsize[0] // 32, imsize[1] // 32
        result_bbox = [[] for n in range(N)]

        for ind_a, anc in enumerate(anchor):
            a_pred = z[:, ind_a * offset:(ind_a + 1) * offset]
            score = a_pred[:, 0].reshape(N, 1, H, W)
            cls_score = a_pred[:, 5:]
            score = score * cls_score
            a_box = a_pred[:, 1:5]
            a_box[:, 0] += np.arange(FW).reshape(1, 1, FW)
            a_box[:, 1] += np.arange(FH).reshape(1, FH, 1)
            a_box[:, 0] /= W
            a_box[:, 1] /= H
            a_box[:, 2] *= anc[0]
            a_box[:, 3] *= anc[1]
            a_box[:, 2] /= W
            a_box[:, 3] /= H

            # Clip bounding box
            w = a_box[:, 2] / 2.
            h = a_box[:, 3] / 2.
            x1 = np.clip(a_box[:, 0] - w, 0, 1)
            y1 = np.clip(a_box[:, 1] - h, 0, 1)
            x2 = np.clip(a_box[:, 0] + w, 0, 1)
            y2 = np.clip(a_box[:, 1] + h, 0, 1)

            a_box[:, 2] = x2 - x1
            a_box[:, 3] = y2 - y1
            a_box[:, 0] = x1 + a_box[:, 2] / 2.
            a_box[:, 1] = y1 + a_box[:, 3] / 2.

            max_score = np.max(score, axis=1)
            keep = np.where(max_score >= score_threshold)
            for i, (b, c) in enumerate(zip(a_box[keep[0], :, keep[1], keep[2]],
                                           score[keep[0], :, keep[1], keep[2]])):
                for ind_c, class_score in enumerate(c):
                    if class_score < score_threshold:
                        continue
                    b = b if isinstance(b, list) else b.tolist()
                    result_bbox[keep[0][i]].append({
                        "box": b,
                        "name": self.class_map[int(ind_c)].decode('ascii'),
                        "class": int(ind_c),
                        "score": float(float(class_score))
                    })
        return nms(result_bbox, nms_threshold)

    def build_data(self, imsize_list=None):
        """
        This function returns a function which creates input data and target data
        specified for Yolov2.

        Returns:
            (function): Returns function which creates input data and target data.

        Example:
            >>> builder = model.build_data()  # This will return function.
            >>> x, y = builder(image_path_list, annotation_list)
            >>> z = model(x)
            >>> loss = model.loss(z, y)
        """
        if imsize_list is None:
            imsize_list = [self.imsize]

        size_N = len(imsize_list)
        perm = np.random.permutation(size_N)

        return TargetBuilderYolov2(self.class_map, size_N, perm, imsize_list, self.anchor, self.num_anchor)

    def get_cell_grid(self, grid_w, grid_h, batch, box):
        cell_x = np.reshape(np.tile(range(grid_w), grid_h), (1, 1, 1, grid_h, grid_w))
        cell_y = np.transpose(cell_x, (0, 1, 2, 4, 3))
        cell_grid = np.tile(np.concatenate([cell_x, cell_y], 2), [batch, box, 1, 1, 1])
        return cell_grid

    def adjust_scale_prediction(self, y_pred, cell_grid, anchors):
        pred_box_xy = y_pred[:, :, 1:3, :, :] + cell_grid
        pred_box_wh = y_pred[:, :, 3:5, :, :] * np.reshape(anchors, [1, self.num_anchor, 2, 1, 1])
        pred_box_conf = y_pred[:, :, 0, :, :]
        pred_box_class = y_pred[:, :, 5:, :, :]

        return pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class

    def extract_ground_truth(self, y_true):
        true_box_xy = y_true[:, :, 1:3, :, :]
        true_box_wh = y_true[:, :, 3:5, :, :]
        true_box_conf = y_true[:, :, 0, :, :]
        true_box_class = y_true[:, :, 5:, :, :]

        return true_box_xy, true_box_wh, true_box_conf, true_box_class

    def calc_loss_xywh(self, true_box_conf, coord_scale, true_box_xy, pred_box_xy, true_box_wh, pred_box_wh):
        coord_mask = np.expand_dims(true_box_conf, axis=2) * coord_scale
#        nb_coord_box = np.sum(coord_mask[np.where(coord_mask>0.0)])
        # /(nb_coord_box + 1e-6) / 2.
        loss_xy = rm.sum(rm.square(true_box_xy - pred_box_xy) * coord_mask)
        # /(nb_coord_box + 1e-6) / 2.
        loss_wh = rm.sum(rm.square(rm.sqrt(true_box_wh) - rm.sqrt(pred_box_wh)) * coord_mask)

        return loss_xy + loss_wh, coord_mask

    def cal_loss_class(self, true_box_conf, class_scale, true_box_class, pred_box_class):
        class_mask = true_box_conf * class_scale
        n, c, h, w = class_mask.shape
#        nb_class_box = np.sum(class_mask[np.where(class_mask>0.0)])
        loss_class = rm.cross_entropy(pred_box_class.transpose(
            0, 2, 1, 3, 4), true_box_class.transpose(0, 2, 1, 3, 4), False)  # keep the dimension
        loss_class = rm.sum(loss_class)  # / (nb_class_box + 1e-6)

        return loss_class

    def get_intersect_area(self, true_xy, true_wh, pred_xy, pred_wh):
        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = np.maximum(pred_mins, true_mins)
        intersect_maxes = np.minimum(pred_maxes, true_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[:, :, 0, ...] * intersect_wh[:, :, 1, ...]

        true_areas = true_wh[:, :, 0, ...] * true_wh[:, :, 1, ...]
        pred_areas = pred_wh[:, :, 0, ...] * pred_wh[:, :, 1, ...]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = intersect_areas / union_areas

        return iou_scores

    def calc_IOU_pred_true_assigned(self, true_box_conf, true_box_xy, true_box_wh, pred_box_xy, pred_box_wh):
        pred_box_xy = pred_box_xy.as_ndarray()
        pred_box_wh = pred_box_wh.as_ndarray()

        iou_scores = self.get_intersect_area(true_box_xy, true_box_wh, pred_box_xy, pred_box_wh)
        true_box_conf_IOU = iou_scores * true_box_conf

        return true_box_conf_IOU

    def calc_IOU_pred_true_best(self, pred_box_xy, pred_box_wh, true_boxes):
        pred_box_xy = pred_box_xy.as_ndarray()
        pred_box_wh = pred_box_wh.as_ndarray()
        true_xy = true_boxes[:, :, 0:2, :, :, :]
        true_wh = true_boxes[:, :, 2:4, :, :, :]
        pred_xy = np.expand_dims(pred_box_xy, -1)
        pred_wh = np.expand_dims(pred_box_wh, -1)  # expand dimension for the buffer size axis

        iou_scores = self.get_intersect_area(true_xy, true_wh, pred_xy, pred_wh)
        best_ious = np.amax(iou_scores, axis=4)
        return best_ious

    def get_conf_mask(self, best_ious, true_box_conf, true_box_conf_IOU, LAMBDA_NO_OBJECT, LAMBDA_OBJECT):
        selected = best_ious < 0.6  # boolean array
        selected = selected.astype(float)  # convert to float array
        conf_mask = selected * (1 - true_box_conf) * LAMBDA_NO_OBJECT
        conf_mask = conf_mask + true_box_conf_IOU * LAMBDA_OBJECT

        return conf_mask

    def calc_loss_conf(self, conf_mask, true_box_conf_IOU, pred_box_conf):
        #        nb_conf_box = np.sum(conf_mask[np.where(conf_mask>0.0)])
        # /(nb_conf_box+1e-6) /2.
        loss_conf = rm.sum(rm.square(true_box_conf_IOU - pred_box_conf) * conf_mask)
        return loss_conf

    def loss(self, x, buffer, y):
        """
        Loss function of ${class} algorithm.

        Args:
            x(ndarray, Node): Output of model.
            y(ndarray, Node): Target array.

        Returns:
            (Node): Loss between x and y.

        Example:
            >>> builder = model.build_data()  # This will return a builder function.
            >>> x, buffer, y = builder(image_path_list, annotation_list)
            >>> z = model(x)
            >>> loss = model.loss(z, buffer, y)
        """
        LAMBDA_NO_OBJECT = 0.5
        LAMBDA_OBJECT = 5.0
        LAMBDA_COORD = 1.0
        LAMBDA_CLASS = 1.0

        batch, box, C, grid_h, grid_w = y.shape
        x = x.reshape(batch, box, C, grid_h, grid_w)

        anchors = np.array(self.anchor)

        cell_grid = self.get_cell_grid(grid_w, grid_h, batch, box)
        pred_box_xy, pred_box_wh, pred_box_conf, pred_box_class = self.adjust_scale_prediction(
            x, cell_grid, anchors)

        true_box_xy, true_box_wh, true_box_conf, true_box_class = self.extract_ground_truth(y)
        loss_xywh, coord_mask = self.calc_loss_xywh(
            true_box_conf, LAMBDA_COORD, true_box_xy, pred_box_xy, true_box_wh, pred_box_wh)
        loss_class = self.cal_loss_class(
            true_box_conf, LAMBDA_CLASS, true_box_class, pred_box_class)

        true_box_conf_IOU = self.calc_IOU_pred_true_assigned(
            true_box_conf, true_box_xy, true_box_wh, pred_box_xy, pred_box_wh)
        best_ious = self.calc_IOU_pred_true_best(pred_box_xy, pred_box_wh, buffer)
        conf_mask = self.get_conf_mask(best_ious, true_box_conf,
                                       true_box_conf_IOU, LAMBDA_NO_OBJECT, LAMBDA_OBJECT)
        loss_conf = self.calc_loss_conf(conf_mask, true_box_conf_IOU, pred_box_conf)

        loss = loss_class + loss_conf + loss_xywh

        return loss

    def fit(self, train_img_path_list, train_annotation_list,
            valid_img_path_list=None, valid_annotation_list=None,
            epoch=160, batch_size=16, optimizer=None, imsize_list=None, augmentation=None, callback_end_epoch=None):
        """
        This function performs training with the given data and hyperparameters.
        Yolov2 is trained using multiple scale images. Therefore, this function
        requires a list of image sizes. If this is not provided, the model will be trained
        using a fixed image size.

        Args:
            train_img_path_list (list): List of image paths.
            train_annotation_list (list): List of annotations.
            valid_img_path_list (list): List of image paths for validation.
            valid_annotation_list (list): List of annotations for validation.
            epoch (int): Number of training epochs.
            batch_size (int): Batch size.
            imsize_list (list): List of image sizes.
            augmentation (Augmentation): Augmentation object.
            callback_end_epoch (function): The given function will be called at the end of each epoch.

        Returns:
            (tuple): Training loss list and validation loss list.

        Example:
            >>> from renom_img.api.detection.yolo_v2 import Yolov2
            >>> train_img_path_list, train_annot_list = ... # Define train data.
            >>> valid_img_path_list, valid_annot_list = ...i # Define validation data.
            >>> class_map = ... # List of class names.
            >>> model = Yolov2(class_map)
            >>> model.fit(
            ...     # Feeds image and annotation data.
            ...     train_img_path_list,
            ...     train_annot_list,
            ...     valid_img_path_list,
            ...     valid_annot_list,
            ...     epoch=8,
            ...     batch_size=8)
            >>>

        The following arguments will be given to the function ``callback_end_epoch``.

        - **epoch** (int) - Current epoch number.
        - **model** (Model) - Yolov2 object.
        - **avg_train_loss_list** (list) - List of average train loss of each epoch.
        - **avg_valid_loss_list** (list) - List of average valid loss of each epoch.

        """

        if imsize_list is None:
            imsize_list = [self.imsize]
#           no need for checking here, cause it is already checked from init function.
        else:
            for ims in imsize_list:
                check_yolov2_init(ims)
        train_dist = ImageDistributor(
            train_img_path_list, train_annotation_list, augmentation=augmentation, num_worker=8)
        if valid_img_path_list is not None and valid_annotation_list is not None:
            valid_dist = ImageDistributor(valid_img_path_list, valid_annotation_list)
        else:
            valid_dist = None

        batch_loop = int(np.ceil(len(train_dist) / batch_size))
        avg_train_loss_list = []
        avg_valid_loss_list = []

        # optimizer settings
        if optimizer is None:
            opt = self.default_optimizer
        else:
            opt = optimizer
        if opt is None:
            raise InvalidOptimizerError(
                "Optimizer is not defined. Please define a valid optimizer.")

        if isinstance(opt, BaseOptimizer):
            opt.setup(batch_loop, epoch)

        for e in range(epoch):
            bar = tqdm(range(batch_loop))
            display_loss = 0
            for i, (train_x, buffers, train_y) in enumerate(train_dist.batch(batch_size, target_builder=self.build_data(imsize_list))):
                # This is for avoiding memory over flow.
                if is_cuda_active() and i % 10 == 0:
                    release_mem_pool()
                self.set_models(inference=False)

                if isinstance(opt, BaseOptimizer):
                    opt.set_information(i, e, avg_train_loss_list, avg_valid_loss_list)

                if (self._model.has_bn and len(train_x) > 1) or (not self._model.has_bn and len(train_x) > 0):
                    with self.train():
                        loss = self.loss(self(train_x), buffers, train_y)
                        reg_loss = loss + self.regularize()
                    reg_loss.grad().update(opt)
                    try:
                        loss = float(loss.as_ndarray()[0])
                    except:
                        loss = float(loss.as_ndarray())

                    display_loss += loss
                    bar.set_description("Epoch:{:03d} Train Loss:{:5.3f}".format(e, loss))
                bar.update(1)
            avg_train_loss = display_loss / (i + 1)

            avg_train_loss_list.append(avg_train_loss)

            if valid_dist is not None:
                if is_cuda_active():
                    release_mem_pool()
                bar.n = 0
                bar.total = int(np.ceil(len(valid_dist) / batch_size))
                display_loss = 0
                for i, (valid_x, buffer_data, valid_y) in enumerate(valid_dist.batch(batch_size, shuffle=False, target_builder=self.build_data())):
                    self.set_models(inference=True)
                    loss = self.loss(self(valid_x), buffer_data, valid_y)

                    try:
                        loss = float(loss.as_ndarray()[0])
                    except:
                        loss = float(loss.as_ndarray())
                    display_loss += loss
                    bar.set_description("Epoch:{:03d} Valid Loss:{:5.3f}".format(e, loss))
                    bar.update(1)
                avg_valid_loss = display_loss / (i + 1)
                avg_valid_loss_list.append(avg_valid_loss)
                bar.set_description("Epoch:{:03d} Avg Train Loss:{:5.3f} Avg Valid Loss:{:5.3f}".format(
                    e, avg_train_loss, avg_valid_loss))
            else:
                bar.set_description("Epoch:{:03d} Avg Train Loss:{:5.3f}".format(e, avg_train_loss))
            bar.close()
            if callback_end_epoch is not None:
                callback_end_epoch(e, self, avg_train_loss_list, avg_valid_loss_list)
        return avg_train_loss_list, avg_valid_loss_list
