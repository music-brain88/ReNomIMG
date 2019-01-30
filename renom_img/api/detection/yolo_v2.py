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
from renom_img.api.classification.darknet import Darknet19, DarknetConv2dBN
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.utility.box import calc_iou_xywh, transform2xy12
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.misc.download import download
from renom_img.api.utility.nms import nms


class AnchorYolov2(object):
    """
    This class contains anchors that will used by in Yolov2.

    Args:
      anchor(list): List of anchors.
      imsize(tuple): Image size.

    """

    def __init__(self, anchor, imsize):
        self.anchor = anchor
        self.imsize = imsize

    def __len__(self):
        return len(self.anchor)


def create_anchor(annotation_list, n_anchor=5, base_size=(416, 416)):
    """
    This function creates 'anchors' for yolo v2 algorithm using k-means clustering.

    Requires following annotation list.

    Perform k-means clustering using custom metric.
    We want to get only anchor's size so we don't have to consider coordinates.

    Args:
        annotation_list(list):
        n_anchor(int):
        base_size(int, list):

    Returns:
        (AnchorYolov2): Anchor list.
    """
    convergence = 0.005
    box_list = [(0, 0, an['box'][2] * base_size[0] / an['size'][0],
                 an['box'][3] * base_size[1] / an['size'][1])
                for an in chain.from_iterable(annotation_list)]

    centroid_index = np.random.permutation(len(box_list))[:n_anchor]
    centroid = [box_list[i] for i in centroid_index]

    def update(centroid, box_list):
        loss = 0
        group = [[] for _ in range(n_anchor)]
        new_centroid = [[0, 0, 0, 0] for _ in range(n_anchor)]

        def metric(x, center): return 1 - calc_iou_xywh(x, center)
        for box in box_list:
            minimum_distance = 100
            for c_ind, cent in enumerate(centroid):
                distance = metric(box, cent)
                if distance < minimum_distance:
                    minimum_distance = distance
                    group_index = c_ind
            group[group_index].append(box)
            new_centroid[group_index][2] += box[2]  # Sum up for calc mean.
            new_centroid[group_index][3] += box[3]
            loss += minimum_distance

        for n in range(n_anchor):
            if (len(group[n])) > 0:
                new_centroid[n][2] /= len(group[n])
                new_centroid[n][3] /= len(group[n])
        return new_centroid, loss

    # Perform k-means.
    new_centroids, old_loss = update(centroid, box_list)
    while True:
        new_centroids, loss = update(new_centroids, box_list)
        if np.abs(loss - old_loss) < convergence:
            break
        old_loss = loss

    # This depends on input image size.
    return AnchorYolov2([[cnt[2], cnt[3]] for cnt in new_centroids], base_size)


class Yolov2(Detection):
    """
    Yolov2 object detection algorithm.

    Args:
        class_map(list): List of class name.
        anchor(AnchorYolov2): Anchors.
        imsize(list): Image size.
            This can be both image size ex):(320, 320) and list of image size
            ex):[(288, 288), (320, 320)]. If list of image size is given,
            the prediction method uses the last image size of the list for prediction.
        load_pretrained_weight(bool, string):
        train_whole_network(bool):

    References:
        | Joseph Redmon, Ali Farhadi
        | **YOLO9000: Better, Faster, Stronger**
        | https://arxiv.org/abs/1612.08242
        |

    Note:
        If you save this model using 'save' method, anchor information(anchor list and base size of them) will be
        saved. So when you load your own saved model, you don't have to give the arguments 'anchor' and 'anchor_size'.
    """

    # Anchor information will be serialized by 'save' method.
    SERIALIZED = ("anchor", "num_anchor", "anchor_size",  *Base.SERIALIZED)
    WEIGHT_URL = "http://renom.jp/docs/downloads/weights/{}/detection/Yolov2.h5".format(__version__)

    def __init__(self, class_map=None, anchor=None,
                 imsize=(320, 320), load_pretrained_weight=False, train_whole_network=False):

        assert (imsize[0] / 32.) % 1 == 0 and (imsize[1] / 32.) % 1 == 0, \
            "Yolo v2 only accepts 'imsize' argument which is list of multiple of 32. \
              exp),imsize=(320, 320)."

        self.flag = False  # This is used for modify loss function.
        self.global_counter = 0
        self.anchor = [] if not isinstance(anchor, AnchorYolov2) else anchor.anchor
        self.anchor_size = imsize if not isinstance(anchor, AnchorYolov2) else anchor.imsize
        self.num_anchor = 0 if anchor is None else len(anchor)

        darknet = Darknet19(1)
        self._opt = rm.Sgd(0.001, 0.9)

        super(Yolov2, self).__init__(class_map, imsize,
                                     load_pretrained_weight, train_whole_network, darknet)

        # Initialize trainable layers.
        last_channel = (self.num_class + 5) * self.num_anchor
        self._conv1 = rm.Sequential([
            DarknetConv2dBN(channel=1024, prev_ch=1024),
            DarknetConv2dBN(channel=1024, prev_ch=1024),
        ])
        self._conv21 = DarknetConv2dBN(channel=64, prev_ch=512, filter=1)
        self._conv2 = DarknetConv2dBN(channel=1024, prev_ch=1024 + 256)
        self._last = rm.Conv2d(channel=last_channel, filter=1)
        self._freezed_network = darknet._base

        for model in [self._conv21, self._conv1, self._conv2]:
            for layer in model.iter_models():
                if not layer.params:
                    continue
                if isinstance(layer, rm.Conv2d):
                    layer.params = {
                        "w": rm.Variable(layer._initializer(layer.params.w.shape), auto_update=True),
                        "b": rm.Variable(np.zeros_like(layer.params.b), auto_update=False),
                    }
                elif isinstance(layer, rm.BatchNormalize):
                    layer.params = {
                        "w": rm.Variable(layer._initializer(layer.params.w.shape), auto_update=True),
                        "b": rm.Variable(np.zeros_like(layer.params.b), auto_update=True),
                    }

    def set_last_layer_unit(self, unit_size):
        # Last layer setting is done in __init__.
        pass

    def get_optimizer(self, current_loss=None, current_epoch=None,
                      total_epoch=None, current_batch=None, total_batch=None, avg_valid_loss_list=None):
        """Returns an instance of Optimizer for training Yolov2 algorithm.

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
            self.global_counter += 1
            if self.global_counter > int(0.3 * (total_epoch * total_batch)):
                self.flag = False
            if current_loss is not None and current_loss > 50:
                self._opt._lr *= 0.1
                return self._opt
            ind0 = int(total_epoch * 1 / 16.)
            ind1 = int(total_epoch * 5 / 16.)
            ind2 = int(total_epoch * 3 / 16.)
            ind3 = total_epoch - ind1 - ind2
            lr_list = [0] + [0.01] * ind0 + [0.001] * ind1 + [0.0001] * ind2 + [0.00001] * ind3

            if current_epoch == 0:
                lr = 0.0001 + (0.001 - 0.0001) / float(total_batch) * current_batch
            else:
                lr = lr_list[current_epoch]
            self._opt._lr = lr

            return self._opt

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
            >>> from renom_img.api.detection.yolo_v2 import Yolov2
            >>>
            >>> x = np.random.rand(1, 3, 224, 224)
            >>> class_map = ["dog", "cat"]
            >>> model = Yolov2(class_map)
            >>> y = model.forward(x) # Forward propagation.
            >>> y = model(x)  # Same as above result.
            >>>
            >>> bbox = model.get_bbox(y) # The output can be reformed using get_bbox method.

        """

        assert len(self.class_map) > 0, \
            "Class map is empty. Please set the attribute class_map when instantiate model class. " +\
            "Or, please load already trained model using the method 'load()'."

        self._freezed_network.set_auto_update(self.train_whole_network)
        self._freezed_network.set_models(inference=(
            not self.train_whole_network or getattr(self, 'inference', False)))

        h, f = self._freezed_network(x)
        f = self._conv21(f)
        h = self._conv1(h)

        h = self._conv2(rm.concat(h,
                                  rm.concat([f[:, :, i::2, j::2] for i in range(2) for j in range(2)])))

        out = self._last(h)
        # Create yolo format.
        N, C, H, W = h.shape

        reshaped = out.reshape(N, self.num_anchor, -1, W * H)
        conf = rm.sigmoid(reshaped[:, :, 0:1]).transpose(0, 2, 1, 3)
        px = rm.sigmoid(reshaped[:, :, 1:2]).transpose(0, 2, 1, 3)
        py = rm.sigmoid(reshaped[:, :, 2:3]).transpose(0, 2, 1, 3)
        pw = rm.exp(reshaped[:, :, 3:4]).transpose(0, 2, 1, 3)
        ph = rm.exp(reshaped[:, :, 4:5]).transpose(0, 2, 1, 3)
        cl = rm.softmax(reshaped[:, :, 5:].transpose(0, 2, 1, 3))
        return rm.concat(conf, px, py, pw, ph, cl).transpose(0, 2, 1, 3).reshape(N, -1, H, W)

    def regularize(self):
        """Regularize term. You can use this function to add regularize term to
        loss function.

        In Yolo v2, weight decay of 0.0005 will be added.

        Example:
            >>> import numpy as np
            >>> from renom_img.api.detection.yolo_v2 import Yolov2
            >>> x = np.random.rand(1, 3, 224, 224)
            >>> y = np.random.rand(1, (5*2+20)*7*7)
            >>> model = Yolov2()
            >>> loss = model.loss(x, y)
            >>> reg_loss = loss + model.regularize() # Adding weight decay term.
        """
        reg = 0
        for layer in self.iter_models():
            if hasattr(layer, "params") and hasattr(layer.params, "w") and isinstance(layer, rm.Conv2d):
                reg += rm.sum(layer.params.w * layer.params.w)
        return (0.0005 / 2.) * reg

    def get_bbox(self, z, score_threshold=0.3, nms_threshold=0.4):
        """
        Example:
            >>> z = model(x)
            >>> model.get_bbox(z)
            [[{'box': [0.21, 0.44, 0.11, 0.32], 'score':0.823, 'class':1, 'name':'dog'}],
             [{'box': [0.87, 0.38, 0.84, 0.22], 'score':0.423, 'class':0, 'name':'cat'}]]

        Args:
            z (ndarray): Output array of neural network. The shape of array

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

        Note:
            Box coordinate and size will be returned as ratio to the original image size.
            Therefore the range of 'box' is [0 ~ 1].

        """

        if hasattr(z, 'as_ndarray'):
            z = z.as_ndarray()

        imsize = self.imsize
        asw = imsize[0] / self.anchor_size[0]
        ash = imsize[1] / self.anchor_size[1]
        anchor = [[an[0] * asw, an[1] * ash] for an in self.anchor]

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
            a_box[:, 0] *= 32
            a_box[:, 1] *= 32
            a_box[:, 2] *= anc[0]
            a_box[:, 3] *= anc[1]
            a_box[:, 0::2] = a_box[:, 0::2] / imsize[0]
            a_box[:, 1::2] = a_box[:, 1::2] / imsize[1]

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
                        "name": self.class_map[int(ind_c)].decode('utf-8'),
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
        else:
            for ims in imsize_list:
                assert (ims[0] / 32.) % 1 == 0 and (ims[1] / 32.) % 1 == 0, \
                    "Yolo v2 only accepts 'imsize' argument which is list of multiple of 32. \
                    exp),imsize=[(288, 288), (320, 320)]."

        size_N = len(imsize_list)
        perm = np.random.permutation(size_N)

        def builder(img_path_list, annotation_list, augmentation=None, nth=0, **kwargs):
            """
            These parameters will be given by distributor.
            Args:
                img_path_list (list): List of input image.
                annotation_list (list): List of detection annotation.
                augmentation (Augmentation): Augmentation object.
                nth (int): Current batch index.
            """
            N = len(img_path_list)
            # This ratio is specific to Darknet19.
            ratio_w = 32.
            ratio_h = 32.
            img_list = []
            num_class = self.num_class
            channel = num_class + 5
            offset = channel

            if nth % 10 == 0:
                perm[:] = np.random.permutation(size_N)
            size_index = perm[0]

            label = np.zeros(
                (N, channel, imsize_list[size_index][1] // 32, imsize_list[size_index][0] // 32))
            img_list, label_list = prepare_detection_data(
                img_path_list, annotation_list, imsize_list[size_index])

            if augmentation is not None:
                img_list, label_list = augmentation(img_list, label_list, mode="detection")

            for n, annotation in enumerate(label_list):
                # This returns resized image.
                # Target processing
                boxces = np.array([a['box'] for a in annotation])
                classes = np.array([[0] * a["class"] + [1] + [0] * (num_class - a["class"] - 1)
                                    for a in annotation])
                if len(boxces.shape) < 2:
                    continue
                # x, y
                cell_x = (boxces[:, 0] // ratio_w).astype(np.int)
                cell_y = (boxces[:, 1] // ratio_h).astype(np.int)
                for i, (cx, cy) in enumerate(zip(cell_x, cell_y)):
                    label[n, 1, cy, cx] = boxces[i, 0]
                    label[n, 2, cy, cx] = boxces[i, 1]

                    # w, h
                    label[n, 3, cy, cx] = boxces[i, 2]
                    label[n, 4, cy, cx] = boxces[i, 3]

                    # Conf
                    label[n, 0, cy, cx] = 1
                    label[n, 5:, cy, cx] = classes[i].reshape(-1, 1, num_class)
            return self.preprocess(img_list), label

        return builder

    def loss(self, x, y):
        """Loss function specified for yolov2.

        Args:
            x(Node, ndarray): Output data of neural network.
            y(Node, ndarray): Target data.

        Returns:
            (Node): Loss between x and y.

        Example:
            >>> z = model(x)
            >>> model.loss(z, y)
        """
        N, C, H, W = x.shape
        nd_x = x.as_ndarray()
        asw = W * 32 / self.anchor_size[0]
        ash = H * 32 / self.anchor_size[1]
        anchor = [[an[0] * asw, an[1] * ash] for an in self.anchor]
        num_anchor = self.num_anchor
        mask = np.zeros((N, C, H, W), dtype=np.float32)
        mask = mask.reshape(N, num_anchor, 5 + self.num_class, H, W)
        if self.inference == False:
            if self.flag:
                mask[:, :, 1:3, ...] = 1.0
                mask[:, :, 3:5, ...] = 0.0
            else:
                mask[:, :, 1:5, ...] = 0.0
        else:
            mask[:, :, 1:5, ...] = 0.0
        mask = mask.reshape(N, C, H, W)

        target = np.zeros((N, C, H, W), dtype=np.float32)
        target = target.reshape(N, num_anchor, 5 + self.num_class, H, W)

        if self.inference == False:
            if self.flag:
                target[:, :, 1:3, ...] = 0.5
                target[:, :, 3:5, ...] = 0.0
            else:
                target[:, :, 1:5, ...] = 0.0
        else:
            target[:, :, 1:5, ...] = 0.0
        target = target.reshape(N, C, H, W)

        low_thresh = 0.6
        im_w, im_h = (W * 32, H * 32)
        offset = 5 + self.num_class

        # Calc iou and get best matched prediction.
        best_anchor_ious = np.zeros((N, 1, H, W), dtype=np.float32)
        for n in range(N):
            gt_index = np.where(y[n, 0] > 0)

            # Create mask for prediction that
            for ind in np.ndindex((num_anchor, H, W)):
                max_iou = -1
                px = (nd_x[n, 1 + ind[0] * offset, ind[1], ind[2]] + ind[2]) * im_w / W
                py = (nd_x[n, 2 + ind[0] * offset, ind[1], ind[2]] + ind[1]) * im_h / H
                pw = nd_x[n, 3 + ind[0] * offset, ind[1], ind[2]] * anchor[ind[0]][0]
                ph = nd_x[n, 4 + ind[0] * offset, ind[1], ind[2]] * anchor[ind[0]][1]
                for h, w in zip(*gt_index):
                    tx = y[n, 1, h, w]
                    ty = y[n, 2, h, w]
                    tw = y[n, 3, h, w]
                    th = y[n, 4, h, w]
                    iou = calc_iou_xywh((px, py, pw, ph), (tx, ty, tw, th))
                    if iou > max_iou:
                        max_iou = iou

                # scale of noobject iou
                if max_iou <= low_thresh:
                    mask[n, ind[0] * offset, ind[1], ind[2]] = 1.
#                     mask[n, ind[0] * offset, ind[1], ind[2]] = nd_x[n,
#                                                                     ind[0] * offset, ind[1], ind[2]] * 1

            # Create target and mask for cell that contains obj.
            for h, w in zip(*gt_index):
                max_anc_iou = -1
                best_anc_ind = None

                tx = y[n, 1, h, w]
                ty = y[n, 2, h, w]
                tw = y[n, 3, h, w]
                th = y[n, 4, h, w]

                for ind, anc in enumerate(anchor):
                    aw = anc[0]
                    ah = anc[1]
                    anc_iou = calc_iou_xywh((0, 0, aw, ah), (0, 0, tw, th))
                    if anc_iou > max_anc_iou:
                        max_anc_iou = anc_iou
                        best_anc_ind = ind

                # target of coordinate
                target[n, 1 + best_anc_ind * offset, h, w] = (tx / 32.) % 1
                target[n, 2 + best_anc_ind * offset, h, w] = (ty / 32.) % 1

                # Don't need to divide by 32 because anchor is already rescaled to input image size.
                target[n, 3 + best_anc_ind * offset, h, w] = tw / anchor[best_anc_ind][0]
                target[n, 4 + best_anc_ind * offset, h, w] = th / anchor[best_anc_ind][1]

                # target of class
                target[n, 5 + best_anc_ind * offset:(best_anc_ind + 1) * offset, h, w] = \
                    y[n, 5:offset, h, w]

                # target of iou.
                px = (nd_x[n, 1 + best_anc_ind * offset, h, w] + w) * 32
                py = (nd_x[n, 2 + best_anc_ind * offset, h, w] + h) * 32
                pw = nd_x[n, 3 + best_anc_ind * offset, h, w] * anchor[best_anc_ind][0]
                ph = nd_x[n, 4 + best_anc_ind * offset, h, w] * anchor[best_anc_ind][1]

                target[n, 0 + best_anc_ind * offset, h, w] = \
                    calc_iou_xywh([px, py, pw, ph], [tx, ty, tw, th])

                # scale of obj iou
                mask[n, 0 + best_anc_ind * offset, h, w] = 5.
#                 mask[n, 0 + best_anc_ind * offset, h,
#                      w] = (1 - nd_x[n, best_anc_ind * offset, h, w]) * 5

                # scale of coordinate
                mask[n, 1 + best_anc_ind * offset, h, w] = 1
                mask[n, 2 + best_anc_ind * offset, h, w] = 1
                mask[n, 3 + best_anc_ind * offset, h, w] = 1
                mask[n, 4 + best_anc_ind * offset, h, w] = 1

                # scale of class
                mask[n, 5 + best_anc_ind * offset:(best_anc_ind + 1) * offset, h, w] = 1

        diff = x - target
        N = np.sum(y[:, 0] > 0)
        mask = np.abs(mask)
        return rm.sum(mask * diff * diff) / N

    def fit(self, train_img_path_list, train_annotation_list,
            valid_img_path_list=None, valid_annotation_list=None,
            epoch=160, batch_size=16, imsize_list=None, augmentation=None, callback_end_epoch=None):
        """
        This function performs training with given data and hyper parameters.
        Yolov2 is trained using multiple scale images. Therefore, this function
        requires list of image size. If it is not given, the model will be trained
        using fixed image size.

        Args:
            train_img_path_list(list): List of image path.
            train_annotation_list(list): List of annotations.
            valid_img_path_list(list): List of image path for validation.
            valid_annotation_list(list): List of annotations for validation.
            epoch(int): Number of training epoch.
            batch_size(int): Number of batch size.
            imsize_list(list): List of image size.
            augmentation(Augmentation): Augmentation object.
            callback_end_epoch(function): Given function will be called at the end of each epoch.

        Returns:
            (tuple): Training loss list and validation loss list.

        Example:
            >>> from renom_img.api.detection.yolo_v2 import Yolov2
            >>> train_img_path_list, train_annot_list = ... # Define own data.
            >>> valid_img_path_list, valid_annot_list = ...
            >>> model = Yolov2()
            >>> model.fit(
            ...     # Feeds image and annotation data.
            ...     train_img_path_list,
            ...     train_annot_list,
            ...     valid_img_path_list,
            ...     valid_annot_list,
            ...     epoch=8,
            ...     batch_size=8)
            >>>

        Following arguments will be given to the function ``callback_end_epoch``.

        - **epoch** (int) - Number of current epoch.
        - **model** (Model) - Yolo2 object.
        - **avg_train_loss_list** (list) - List of average train loss of each epoch.
        - **avg_valid_loss_list** (list) - List of average valid loss of each epoch.

        """

        if imsize_list is None:
            imsize_list = [self.imsize]
        else:
            for ims in imsize_list:
                assert (ims[0] / 32.) % 1 == 0 and (ims[1] / 32.) % 1 == 0, \
                    "Yolo v2 only accepts 'imsize' argument which is list of multiple of 32. \
                    exp),imsize=[(288, 288), (320, 320)]."

        train_dist = ImageDistributor(
            train_img_path_list, train_annotation_list, augmentation=augmentation, num_worker=8)
        if valid_img_path_list is not None and valid_annotation_list is not None:
            valid_dist = ImageDistributor(valid_img_path_list, valid_annotation_list)
        else:
            valid_dist = None

        batch_loop = int(np.ceil(len(train_dist) / batch_size))
        avg_train_loss_list = []
        avg_valid_loss_list = []

        for e in range(epoch):
            bar = tqdm(range(batch_loop))
            display_loss = 0
            for i, (train_x, train_y) in enumerate(train_dist.batch(batch_size, shuffle=True, target_builder=self.build_data(imsize_list))):
                # This is for avoiding memory over flow.
                if is_cuda_active() and i % 10 == 0:
                    release_mem_pool()
                self.set_models(inference=False)
                with self.train():
                    loss = self.loss(self(train_x), train_y)
                    reg_loss = loss + self.regularize()
                reg_loss.grad().update(self.get_optimizer(loss.as_ndarray(), e, epoch, i, batch_loop))

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
                for i, (valid_x, valid_y) in enumerate(valid_dist.batch(batch_size, shuffle=False, target_builder=self.build_data())):
                    self.set_models(inference=True)
                    loss = self.loss(self(valid_x), valid_y)

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
