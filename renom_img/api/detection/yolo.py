#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from renom.core import Node, to_value, get_gpu


def box_iou(box1, box2):
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


def make_box(box):
    x1 = box[:, :, :, 0] - box[:, :, :, 2] / 2
    y1 = box[:, :, :, 1] - box[:, :, :, 3] / 2
    x2 = box[:, :, :, 0] + box[:, :, :, 2] / 2
    y2 = box[:, :, :, 1] + box[:, :, :, 3] / 2
    return [x1, y1, x2, y2]


class yolo_loss(Node):
    u"""Loss function for Yolo detection.
    Last layer of the network needs to be following size:
    cells*cells*(bbox*5+classes)
    5 is because every bounding box gets 1 score and 4 locations (x, y, w, h)
    Ex:
    Prediction: 2 bbox per cell, 7*7 cells per image, 5 classes
    X[0,0,0] = S  X  Y  W  H  S  X  Y  W  H  0 0 0 1 0
              |---1st bbox--||---2nd bbox--||-classes-|
    """

    def __new__(cls, x, y, cells, bbox, classes):
        return cls.calc_value(x, y, cells, bbox, classes)

    @classmethod
    def _oper_cpu(cls, x, y, cells, bbox, classes):
        x.to_cpu()
        noobj_scale = 0.5
        obj_scale = 5
        N = x.shape[0]  # np.prod(x.shape)
        raw_x = x
        x = x.as_ndarray()
        x = x.reshape(-1, cells, cells, (5 * bbox) + classes)
        y = y.reshape(-1, cells, cells, 5 + classes)
        deltas = np.zeros_like(x)
        loss = 0
        # Case: there's no object in the cell
        bg_ind = (y[:, :, :, 0] == 0)
        # Case: there's an object
        obj_ind = (y[:, :, :, 0] == 1)
        # add 5th part of the equation
        deltas[obj_ind, bbox * 5:] = x[obj_ind, bbox * 5:] - y[obj_ind, 5:]
        loss += np.sum(np.square(x[obj_ind, bbox * 5:] - y[obj_ind, 5:]))
        # search for the best predicted bounding box
        truth_box = make_box(y[:, :, :, 1:5])
        ious = np.zeros((y.shape[0], y.shape[1], y.shape[2], bbox))

        for b in range(bbox):
            # add 4th part of the equation
            deltas[bg_ind, b * 5] = noobj_scale * x[bg_ind, b * 5]
            loss += noobj_scale * np.sum(np.square(x[bg_ind, b * 5]))
            # get ious for current box
            box = x[:, :, :, 5 * b + 1:5 * b + 5]
            predict_box = make_box(box)
            ious[:, :, :, b] = box_iou(truth_box, predict_box)
        best_ind = np.argmax(ious, axis=3)

        for b in range(bbox):
            update_ind = (b == best_ind) & obj_ind
            # add 3rd part of the equation
            loss += np.sum(np.square(x[update_ind, 5 * b] - 1))
            deltas[update_ind, 5 * b] = (x[update_ind, 5 * b] - 1)

            # add 1st-2nd part of the equation
            loss += obj_scale * \
                np.sum(np.square(x[update_ind, 5 * b +
                                   1:5 * b + 5] - y[update_ind, 1:5]))

            deltas[update_ind, 5 * b + 1:5 * b + 5] = obj_scale * \
                (x[update_ind, 5 * b + 1:5 * b + 5] - y[update_ind, 1:5])

        divs = N * cells * cells * bbox / (7 * 7 * 2)

        loss = loss / 2 / divs
        deltas = deltas.reshape(-1, cells * cells *
                                (5 * bbox + classes)) / divs
        ret = cls._create_node(loss)
        ret.attrs._x = raw_x
        ret.attrs._deltas = deltas
        return ret

    @classmethod
    def _oper_gpu(cls, x, y, cells, bbox, classes):
        return cls._oper_cpu(x, y, cells, bbox, classes)

    def _backward_cpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._x, Node):
            self.attrs._x._update_diff(context, self.attrs._deltas * dy)

    def _backward_gpu(self, context, dy, **kwargs):
        if isinstance(self.attrs._x, Node):
            self.attrs._x._update_diff(
                context, get_gpu(self.attrs._deltas) * dy)


class YoloLoss(object):

    def __init__(self, cells=7, bbox=2, classes=20):
        self._cells = cells
        self._bbox = bbox
        self._classes = classes

    def __call__(self, x, y):
        return yolo_loss(x, y, self._cells, self._bbox, self._classes)
