from itertools import chain
import numpy as np
import renom as rm
from PIL import Image

from renom_img.api.model.darknet import Darknet19Base, DarknetConv2dBN
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.utility.box import calc_iou_xywh, transform2xy12


def make_box(box):
    x1 = box[:, :, :, 0] - box[:, :, :, 2] / 2.
    y1 = box[:, :, :, 1] - box[:, :, :, 3] / 2.
    x2 = box[:, :, :, 0] + box[:, :, :, 2] / 2.
    y2 = box[:, :, :, 1] + box[:, :, :, 3] / 2.
    return [x1, y1, x2, y2]


def create_anchor(annotation_list, n_anchor=5, base_size=(416, 416)):
    """
    Requires following annotation list.



    Perform k-means clustering using custom metric.
    We want to get only anchor's size so we don't have to consider coordinates.

    Args:
        box_list: 
    """
    convergence = 0.01
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
            new_centroid[n][2] /= len(group[n])
            new_centroid[n][3] /= len(group[n])
        return new_centroid, loss

    # Perform s-means.
    new_centroids, old_loss = update(centroid, box_list)
    while True:
        new_centroids, loss = update(new_centroids, box_list)
        if np.abs(loss - old_loss) < convergence:
            break
        old_loss = loss

    # This depends on input image size.
    return [[cnt[2], cnt[3]] for cnt in new_centroids]


class Yolov2(rm.Model):
    """
    Args:
        num_class(int):
        anchor(list):
        anchor_size(list):
        imsize(lit):
        load_weight_path(string):
        train_whole_network(bool):
    """

    WEIGHT_URL = "Yolov2.h5"

    def __init__(self, num_class, anchor, anchor_size,
            imsize=(224, 224), load_weight_path=None, train_whole_network=False):
        assert (imsize[0] / 32.) % 1 == 0 and (imsize[1] / 32.) % 1 == 0, \
            "Yolo v2 only accepts 'imsize' argument which is list of multiple of 32. \
            exp),imsize=(320, 320)."
        self.imsize = imsize
        self.anchor_size = anchor_size
        self._freezed_network = Darknet19Base()
        self.anchor = anchor
        self.num_anchor = 0 if anchor is None else len(anchor)
        self.cn = num_class
        last_channel = (num_class + 5) * self.num_anchor
        self._base = Darknet19Base()
        self._conv1 = rm.Sequential([
            DarknetConv2dBN(channel=1024),
            DarknetConv2dBN(channel=1024),
        ])
        self._conv2 = DarknetConv2dBN(channel=1024)
        self._last = rm.Conv2d(channel=last_channel, filter=1)
        self._opt = rm.Sgd(0.01, 0.9)
        self._train_whole_network = train_whole_network

        # Load weight here.
        # self.load()
        if load_weight_path is not None:
            self.load(load_weight_path)
            # self._conv1[0].params = {}
            # self._conv1[1].params = {}
            # self._conv2.params = {}
            self._last.params = {}

    @property
    def freezed_network(self):
        return self._freezed_network

    def get_optimizer(self, current_epoch=None, total_epoch=None, current_batch=None, total_batch=None):
        if any([num is None for num in [current_epoch, total_epoch, current_batch, total_batch]]):
            return self._opt
        else:
            ind1 = int(total_epoch * 0.5)
            ind2 = total_epoch - ind1
            lr_list = [0] + [0.001] * ind1 + [0.001] * ind2
            if current_epoch == 0:
                lr = 0.00001 + (0.001 - 0.00001) / float(total_batch) * current_batch
            else:
                lr = lr_list[current_epoch]
            self._opt._lr = lr
            return self._opt

    def preprocess(self, x):
        return x / 255. * 2 - 1

    def forward(self, x):
        self.freezed_network.set_auto_update(self._train_whole_network)
        self.freezed_network.set_models(inference= \
            (not self._train_whole_network or getattr(self, 'inference', False)))
        h, f = self.freezed_network(x)
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
        out = rm.concat(conf, px, py, pw, ph, cl).transpose(0, 2, 1, 3).reshape(N, -1, H, W)
        return out

    def regularize(self):
        reg = 0
        for layer in self.iter_models():
            if hasattr(layer, "params") and hasattr(layer.params, "w"):
                reg += rm.sum(layer.params.w * layer.params.w)
        return 0.0005 * reg

    def get_bbox(self, z):
        if hasattr(z, 'as_ndarray'):
            z = z.as_ndarray()

        asw = self.imsize[0] / self.anchor_size[0]
        ash = self.imsize[1] / self.anchor_size[1]
        anchor = [[an[0] * asw, an[1] * ash] for an in self.anchor]

        num_anchor = len(anchor)
        N, C, H, W = z.shape
        offset = self.cn + 5
        FW, FH = self.imsize[0] // 32, self.imsize[1] // 32
        box_list = [[] for n in range(N)]
        score_list = [[] for n in range(N)]

        for ind_a, anc in enumerate(anchor):
            a_pred = z[:, ind_a * offset:(ind_a + 1) * offset]
            score = a_pred[:, 0].reshape(N, 1, H, W)
            cls_score = a_pred[:, 5:]
            score = score * cls_score
            max_index = np.argmax(score, axis=1)
            max_conf = np.max(score, axis=1)
            max_conf[max_conf < 0.05] = 0
            a_box = a_pred[:, 1:5]
            a_box[:, 0] += np.arange(FW).reshape(1, 1, FW)
            a_box[:, 1] += np.arange(FH).reshape(1, FH, 1)
            a_box[:, 0] *= 32
            a_box[:, 1] *= 32
            a_box[:, 2] *= anc[0]
            a_box[:, 3] *= anc[1]
            a_box[:, 0::2] = np.clip(a_box[:, 0::2], 0, self.imsize[0]) / self.imsize[0]
            a_box[:, 1::2] = np.clip(a_box[:, 1::2], 0, self.imsize[1]) / self.imsize[1]
            keep = np.where(max_conf > 0)
            for i, (b, s, c) in enumerate(zip(a_box[keep[0], :, keep[1], keep[2]],
                                              max_conf[keep[0], keep[1], keep[2]],
                                              score[keep[0], :, keep[1], keep[2]])):
                box_list[keep[0][i]].append(b)
                score_list[keep[0][i]].append((s, np.argmax(c)))

        # NMS
        for n in range(N):
            sorted_ind = np.argsort([s[0] for s in score_list[n]])[::-1]
            keep = np.ones((len(score_list[n]),), dtype=np.bool)
            for i, ind1 in enumerate(sorted_ind):
                if not keep[i]:
                    continue
                box1 = box_list[n][ind1]
                for j, ind2 in enumerate(sorted_ind[i + 1:]):
                    box2 = box_list[n][ind2]
                    if keep[j] and score_list[n][ind1][1] == score_list[n][ind2][1]:
                        keep[j] = calc_iou_xywh(box1, box2) < 0.4

            box_list[n] = [{
                "box": box_list[n][i],
                "score": score_list[n][i][0],
                "class": score_list[n][i][1],
            } for i, k in enumerate(keep) if k]

        return box_list

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
        # This ratio is specific to Darknet19.
        N = len(img_path_list)
        ratio_w = 32.
        ratio_h = 32.
        img_list = []
        num_class = self.cn
        channel = num_class + 5
        offset = channel

        label = np.zeros((N, channel, self.imsize[1] // 32, self.imsize[0] // 32))
        img_list, label_list = prepare_detection_data(img_path_list, annotation_list, self.imsize)

        for n, annotation in enumerate(label_list):
            # This returns resized image.

            # Target processing
            boxces = np.array([a['box'] for a in annotation])
            classes = np.array([[0] * a["class"] + [1] + [0] * (num_class - a["class"] - 1)
                                for a in annotation])

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

    def loss(self, x, y):
        """
        Returns mask.
        Args: 
            x: Yolo output. (N, C(anc*(5+class)), H, W)
            y: (N, C(5+class), H(feature), W(feature))
        """
        N, C, H, W = x.shape
        nd_x = x.as_ndarray()
        asw = self.imsize[0] / self.anchor_size[0]
        ash = self.imsize[1] / self.anchor_size[1]
        anchor = [[an[0] * asw, an[1] * ash] for an in self.anchor]

        num_anchor = self.num_anchor
        mask = np.zeros((N, C, H, W), dtype=np.float32)
        mask = mask.reshape(N, num_anchor, 5 + self.cn, H, W)
        mask[:, :, 1:5, ...] += 0.1
        mask = mask.reshape(N, C, H, W)

        target = np.zeros((N, C, H, W), dtype=np.float32)
        target = target.reshape(N, num_anchor, 5 + self.cn, H, W)
        target[:, :, 1:3, ...] = 0.5
        target[:, :, 3:5, ...] = 1.0
        target = target.reshape(N, C, H, W)

        low_thresh = 0.6
        im_w, im_h = self.imsize
        offset = 5 + self.cn

        # Calc iou and get best matched prediction.
        best_ious = np.zeros((N, num_anchor, H, W), dtype=np.float32)
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
                best_ious[n, ind[0], ind[1], ind[2]] = max_iou

                # scale of noobject iou
                if max_iou <= low_thresh:
                    mask[n, ind[0] * offset, ind[1], ind[2]] = \
                        (0 - x[n, ind[0] * offset, ind[1], ind[2]]) * 1
                #     mask[n, ind[0]*offset, ind[1], ind[2]] = 1

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
                target[n, 0 + best_anc_ind * offset, h, w] = \
                    best_ious[n, best_anc_ind, h, w]

                # scale of obj iou
                mask[n, 0 + best_anc_ind * offset, h, w] = \
                    (1 - best_ious[n, best_anc_ind, h, w]) * 5.
                # mask[n, 0+best_anc_ind*offset, h, w] = 5.

                # scale of coordinate
                mask[n, 1 + best_anc_ind * offset, h, w] = 1
                mask[n, 2 + best_anc_ind * offset, h, w] = 1
                mask[n, 3 + best_anc_ind * offset, h, w] = 1
                mask[n, 4 + best_anc_ind * offset, h, w] = 1

                # scale of class
                mask[n, 5 + best_anc_ind * offset:(best_anc_ind + 1) * offset, h, w] = 1
        diff = (x - target)
        return rm.sum(diff * diff * mask) / np.sum(y[:, 0] > 0) / 2.
