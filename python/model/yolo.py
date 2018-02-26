import os
import sys
import urllib.request as request
import numpy as np
import renom as rm
from .yolo_detector import Yolo, build_truth, apply_nms


class YoloBase(rm.Model):

    def __init__(self, class_num, cell, bbox, img_size):
        self._class_num = class_num
        self._optimizer = rm.Sgd(momentum=0.9)
        self._cells = cell
        self._bbox = bbox
        self._img_size = img_size
        self.loss_func = Yolo(cell, bbox, class_num)

    def transform_label_format(self, label):
        yolo_format = []
        for l in label:
            yolo_format.append(build_truth(
                l.reshape(1, -1), self._img_size[0], self._img_size[1], self._cells, self._class_num).flatten())
        return np.array(yolo_format)

    def optimizer(self, epoch, batch_loop, num_epoch, num_batch_loop):
        lr_list = [0.001] \
                + [0.01] * int(num_epoch*0.5) \
                + [0.001] * int(num_epoch*0.25) \
                + [0.0001] * int(num_epoch*0.25)
        if len(lr_list) < num_epoch:
            lr_list += [0.0001] * (len(num_epoch) - len(lr_list))
        lr_list = lr_list[:num_epoch]

        if epoch == 0:
            lr = batch_loop * ((0.01 - 0.001) / num_batch_loop) + lr_list[epoch]
        else:
            lr = lr_list[epoch]

        self._optimizer._lr = lr
        return self._optimizer

    def freezed_forward(self, x):
        z = self._upper_network(x)
        return z

    def forward(self, x):
        z = self._detector_network(x)
        return z

    def get_bbox(self, model_original_formatted_out):
        obj_list = []
        data = model_original_formatted_out
        for d in data:
            objs = apply_nms(d.reshape(self._cells, self._cells, 5 * self._bbox + self._class_num),
                             self._cells, self._bbox, self._class_num,
                             image_size=None, thresh=0.2, iou_thresh=0.4)
            obj_list.append(objs)
        return obj_list

    def weight_decay(self):
        wd = 0
        for m in self._detector_network:
            if hasattr(m, "params"):
                w = m.params.get("w", None)
                if w is not None:
                    wd += rm.sum(w**2)
        return wd * 0.0005


class YoloDarknet(YoloBase):

    def __init__(self, class_num, cell, bbox, img_size):
        super(YoloDarknet, self).__init__(class_num, cell, bbox, img_size)
        last_dense_size = cell * cell * (5 * bbox + class_num)
        model = rm.Sequential([
            # 1st Block
            rm.Conv2d(channel=64, filter=7, stride=2, padding=3),
            rm.LeakyRelu(slope=0.1),
            rm.MaxPool2d(stride=2, filter=2),

            # 2nd Block
            rm.Conv2d(channel=192, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.MaxPool2d(stride=2, filter=2),

            # 3rd Block
            rm.Conv2d(channel=128, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=256, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=256, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.MaxPool2d(stride=2, filter=2),

            # 4th Block
            rm.Conv2d(channel=256, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=256, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=256, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=256, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.MaxPool2d(stride=2, filter=2),

            # 5th Block
            rm.Conv2d(channel=512, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=512, filter=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, stride=2, padding=1),
            rm.LeakyRelu(slope=0.1),

            # 6th Block
            rm.Conv2d(channel=1024, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),
            rm.Conv2d(channel=1024, filter=3, padding=1),
            rm.LeakyRelu(slope=0.1),

            # 7th Block
            rm.Flatten(),
            rm.Dense(1024),
            rm.LeakyRelu(slope=0.1),
            rm.Dense(4096),
            rm.LeakyRelu(slope=0.1),
            rm.Dropout(0.5),

            # 8th Block
            rm.Dense(last_dense_size),
        ])

        if not os.path.exists("yolo.h5"):
            print("Weight parameters will be downloaded.")
            url = "http://docs.renom.jp/downloads/weights/yolo.h5"
            request.urlretrieve(url, "yolo.h5")

        model.load('yolo.h5')
        self._upper_network = rm.Sequential(model[:-7])
        self._detector_network = rm.Sequential(model[-7:])

        for layer in self._detector_network:
            if hasattr(layer, "params"):
                layer.params = {}
