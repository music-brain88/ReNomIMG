import os
import sys
import numpy as np
import renom as rm
from renom_img.api.detection.yolo import Yolo
from renom_img.api.model.darknet import Darknet
from renom_img.api.utility.nms import transform2xy12 as xy12
from renom_img.api.utility.nms import calc_iou
from renom_img.server.model_wrapper.wrapper_base import Wrapper


class WrapperYoloDarknet(Wrapper):

    def __init__(self, num_class, cells, bbox, img_size):
        super(WrapperYoloDarknet, self).__init__(l2reg=0.0005)

        if isinstance(cells, (tuple, list)):
            cell_w = cells[0]
            cell_h = cells[1]
        elif isinstance(cells, int):
            cell_w = cells
            cell_h = cells
        self._bbox = bbox
        self._cell = (cell_w, cell_h)
        last_dense_size = cell_w * cell_h * (5 * bbox + num_class)
        model = Darknet(last_dense_size, load_weight=True)
        self._prob_thresh = 0.2
        self._nms_thresh = 0.4
        self._num_class = num_class
        self._img_size = img_size
        self._freezed_model = rm.Sequential(model[:-7])
        self._learnable_model = rm.Sequential(model[-7:])

        for layer in self._learnable_model.iter_models():
            layer.params = {}

        self._optimizer = rm.Sgd(momentum=0.9)
        self._yolo_detector_loss = Yolo(cell_w, bbox, num_class)

    def load(self, path):
        super(WrapperYoloDarknet, self).load(path)
        bbox = self._bbox
        cell = self._cell[0]
        last_size = self._learnable_model[-1].params.w.shape[1] / (cell**2) - 5 * bbox
        self._num_class = int(last_size)

    def get_bbox(self, model_original_formatted_out):
        assert len(model_original_formatted_out.shape) == 2
        N = len(model_original_formatted_out)
        cell = self._cell[0]
        bbox = self._bbox
        probs = np.zeros((N, cell, cell, bbox, self._num_class))
        boxes = np.zeros((N, cell, cell, bbox, 4))
        yolo_format_out = model_original_formatted_out.reshape(
            N, cell, cell, bbox * 5 + self._num_class)
        offset = np.vstack([np.arange(cell) for c in range(cell)])

        for b in range(bbox):
            prob = yolo_format_out[:, :, :, b * 5][..., None] * yolo_format_out[:, :, :, bbox * 5:]
            probs[:, :, :, b, :] = prob
            boxes[:, :, :, b, :] = yolo_format_out[:, :, :, b * 5 + 1:b * 5 + 5]
            boxes[:, :, :, b, 0] += offset
            boxes[:, :, :, b, 1] += offset.T
        boxes[:, :, :, :, 0:2] /= float(cell)
        probs = probs.reshape(N, -1, self._num_class)
        boxes = boxes.reshape(N, -1, 4)

        probs[probs < self._prob_thresh] = 0
        # Perform NMS
        argsort = np.argsort(probs, axis=1)[:, ::-1]
        for n in range(N):
            for cl in range(self._num_class):
                for b in range(len(boxes[n])):
                    if probs[n, argsort[n, b, cl], cl] == 0:
                        continue
                    b1 = xy12(boxes[n, argsort[n, b, cl], :])
                    for comp in range(b + 1, len(boxes[n])):
                        b2 = xy12(boxes[n, argsort[n, comp, cl], :])
                        if calc_iou(b1, b2) > self._nms_thresh:
                            probs[n, argsort[n, comp, cl], cl] = 0

        result = [[] for _ in range(N)]
        max_class = np.argmax(probs, axis=2)
        max_probs = np.max(probs, axis=2)
        indexes = np.nonzero(np.clip(max_probs, 0, 1))
        for i in range(len(indexes[0])):
            # Note: Take care types.
            result[indexes[0][i]].append({
                "class": int(max_class[indexes[0][i], indexes[1][i]]),
                "box": boxes[indexes[0][i], indexes[1][i]].astype(np.float64).tolist(),
                "score": float(max_probs[indexes[0][i], indexes[1][i]])
            })
        return result


    def optimizer(self, nth_epoch, nth_batch, total_epoch, total_batch_loop):
        lr_list = [0.001] \
            + [0.01] * int(total_epoch * 0.5) \
            + [0.001] * int(total_epoch * 0.25) \
            + [0.0001] * int(total_epoch * 0.25)
        if len(lr_list) < total_epoch:
            lr_list += [0.0001] * (len(total_epoch) - len(lr_list))
        lr_list = lr_list[:total_epoch]

        if nth_epoch == 0:
            lr = nth_batch * ((0.01 - 0.001) /
                              total_batch_loop) + lr_list[nth_epoch]
        else:
            lr = lr_list[nth_epoch]

        self._optimizer._lr = lr
        return self._optimizer


    def loss_func(self, x, y):
        return self._yolo_detector_loss(x, y)
