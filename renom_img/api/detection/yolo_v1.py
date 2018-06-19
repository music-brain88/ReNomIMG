import numpy as np
import renom as rm
from PIL import Image
from renom_img.api.model.darknet import Darknet
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.utility.nms import transform2xy12, nms


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


class Yolov1(rm.Model):

    WEIGHT_URL = Darknet.WEIGHT_URL

    def __init__(self, num_class, cells, bbox, imsize=(224, 224), load_weight_path=None):
        assert load_weight_path is None or isinstance(load_weight_path, str)

        if not hasattr(cells, "__getitem__"):
            cells = (cells, cells)

        self._num_class = num_class
        self._cells = cells
        self._bbox = bbox
        self._last_dense_size = (num_class + 5 * bbox) * cells[0] * cells[1]
        model = Darknet(self._last_dense_size, load_weight_path=load_weight_path)

        self.imsize = imsize
        self._freezed_network = rm.Sequential(model[:-7])
        self._network = rm.Sequential(model[-7:])

        self._opt = rm.Sgd(0.01, 0.9)

        for layer in self._network.iter_models():
            layer.params = {}

    @property
    def freezed_network(self):
        return self._freezed_network

    @freezed_network.setter
    def freezed_network(self, new_network):
        assert isinstance(new_network, rm.Model)
        self._freezed_network = new_network

    @property
    def network(self):
        return self._network

    @network.setter
    def network(self, new_network):
        assert isinstance(new_network, rm.Model)
        self._network = new_network

    def get_optimizer(self, current_epoch=None, total_epoch=None, current_batch=None, total_batch=None):
        if any([num is None for num in [current_epoch, total_epoch, current_batch, total_batch]]):
            return self._opt
        else:
            ind1 = int(total_epoch * 0.5)
            ind2 = int(total_epoch * 0.3)
            ind3 = total_epoch - (ind1 + ind2 + 1)
            lr_list = [0] + [0.01] * ind1 + [0.001] * ind2 + [0.0001] * ind3
            if current_epoch == 0:
                lr = 0.0001 + (0.01 - 0.0001) / float(total_batch) * current_batch
            else:
                lr = lr_list[current_epoch]
            self._opt._lr = lr
            return self._opt

    def preprocess(self, x):
        return x / 255. * 2 - 1

    def forward(self, x):
        self.freezed_network.set_auto_update(False)
        return self.network(self.freezed_network(x).as_ndarray())

    def regularize(self):
        reg = 0
        for layer in self.network.iter_models():
            if hasattr(layer, "params") and hasattr(layer.params, "w"):
                reg += rm.sum(layer.params.w * layer.params.w)
        return 0.0005 * reg

    def get_bbox(self, z):
        if hasattr(z, 'as_ndarray'):
            z = z.as_ndarray()

        N = len(z)
        cell = self._cells[0]
        bbox = self._bbox
        probs = np.zeros((N, cell, cell, bbox, self._num_class))
        boxes = np.zeros((N, cell, cell, bbox, 4))
        yolo_format_out = z.reshape(
            N, cell, cell, bbox * 5 + self._num_class)
        offset = np.vstack([np.arange(cell) for c in range(cell)])

        for b in range(bbox):
            prob = yolo_format_out[:, :, :, b * 5][..., None] * yolo_format_out[:, :, :, bbox * 5:]
            probs[:, :, :, b, :] = prob
            boxes[:, :, :, b, :] = yolo_format_out[:, :, :, b * 5 + 1:b * 5 + 5]
            boxes[:, :, :, b, 0] += offset
            boxes[:, :, :, b, 1] += offset.T
            boxes[:, :, :, b, 2] = boxes[:, :, :, b, 2]**2
            boxes[:, :, :, b, 3] = boxes[:, :, :, b, 3]**2
        boxes[:, :, :, :, 0:2] /= float(cell)
        probs = probs.reshape(N, -1, self._num_class)
        boxes = boxes.reshape(N, -1, 4)

        probs[probs < 0.3] = 0
        # Perform NMS

        argsort = np.argsort(probs, axis=1)[:, ::-1]
        for n in range(N):
            for cl in range(self._num_class):
                for b in range(len(boxes[n])):
                    if probs[n, argsort[n, b, cl], cl] == 0:
                        continue
                    b1 = transform2xy12(boxes[n, argsort[n, b, cl], :])
                    for comp in range(b + 1, len(boxes[n])):
                        b2 = transform2xy12(boxes[n, argsort[n, comp, cl], :])
                        if calc_iou(b1, b2) > 0.4:
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

    def predict(self, img_list):
        self.set_models(inference=True)
        if isinstance(img_list, (list, str)):
            if isinstance(img_list, (tuple, list)):
                img_array = np.vstack([load_img(path, self.imsize)[None] for path in img_list])
            else:
                img_array = load_img(img_list, self.imsize)[None]
            img_array = self.preprocess(img_array)
        else:
            img_array = img_list
        pred = self(img_array).as_ndarray()
        return self.get_bbox(pred)

    def build_data(self, img_path_list, annotation_list, augmentation=None):
        """
        Args:
            x: Image path list.
            y: Detection formatted label.
        """
        N = len(img_path_list)
        num_bbox = self._bbox
        cell_w, cell_h = self._cells
        target = np.zeros((N, self._cells[0], self._cells[1], 5 * num_bbox + self._num_class))

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
                one_hot = [0] * obj["class"] + [1] + [0] * (self._num_class - obj["class"] - 1)
                target[n, int(ty), int(tx)] = \
                    np.concatenate(([1, tx % 1, ty % 1, tw, th] * num_bbox, one_hot))
        return self.preprocess(img_data), target.reshape(N, -1)

    def loss(self, x, y):
        N = len(x)
        nd_x = x.as_ndarray()
        num_bbox = self._bbox
        target = y.reshape(N, self._cells[0], self._cells[1], 5 * num_bbox + self._num_class)
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

        diff = x - y
        return rm.sum(diff * diff * mask.reshape(N, -1)) / N / 2.
