import os
from itertools import chain
import numpy as np
import renom as rm
from tqdm import tqdm
from PIL import Image

from renom_img.api.model.darknet import Darknet19Base, DarknetConv2dBN
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.utility.box import calc_iou_xywh, transform2xy12
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.misc.download import download


class AnchorYolov2(object):

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


class Yolov2(rm.Model):
    """

    Args:
        num_class(int):
        anchor(list):
        imsize(lit):
        load_pretrained_weight(bool, string):
        train_whole_network(bool):

    Note:
        If you save this model using 'save' method, anchor information(anchor list and base size of them) will be
        saved. So when you load your own saved model, you don't have to give the arguments 'anchor' and 'anchor_size'.
    """

    # Anchor information will be serialized by 'save' method.
    SERIALIZED = ("anchor", "num_anchor", "anchor_size", "_class_map", "num_class")
    WEIGHT_URL = "http://docs.renom.jp/downloads/weights/Yolov2.h5"

    def __init__(self, class_map, anchor,
                 imsize=(320, 320), load_pretrained_weight=False, train_whole_network=False):
        assert (imsize[0] / 32.) % 1 == 0 and (imsize[1] / 32.) % 1 == 0, \
            "Yolo v2 only accepts 'imsize' argument which is list of multiple of 32. \
            exp),imsize=(320, 320)."

        num_class = len(class_map)
        self._class_map = [k for k, v in sorted(
            class_map.items(), key=lambda x:x[1])] if isinstance(class_map, dict) else class_map
        self._class_map = [c.encode("ascii", "ignore") for c in self._class_map]
        self.imsize = imsize
        self._freezed_network = Darknet19Base()
        self.anchor = [] if not isinstance(anchor, AnchorYolov2) else anchor.anchor
        self.anchor_size = imsize if not isinstance(anchor, AnchorYolov2) else anchor.imsize
        self.num_anchor = 0 if anchor is None else len(anchor)
        self.num_class = num_class
        last_channel = (num_class + 5) * self.num_anchor
        self._base = Darknet19Base()
        self._conv1 = rm.Sequential([
            DarknetConv2dBN(channel=1024, prev_ch=1024),
            DarknetConv2dBN(channel=1024, prev_ch=1024),
        ])
        self._conv2 = DarknetConv2dBN(channel=1024, prev_ch=1024 * 3)
        self._last = rm.Conv2d(channel=last_channel, filter=1)
        self._last.params = {
            "w": rm.Variable(self._last._initializer((last_channel, 1024, 1, 1)), auto_update=True),
            "b": rm.Variable(np.zeros((1, last_channel, 1, 1), dtype=np.float32), auto_update=False),
        }

        self._opt = rm.Sgd(0.001, 0.9)
        self._train_whole_network = train_whole_network

        # Load weight here.
        if load_pretrained_weight:
            if isinstance(load_pretrained_weight, bool):
                load_pretrained_weight = self.__class__.__name__ + '.h5'

            if not os.path.exists(load_pretrained_weight):
                download(self.WEIGHT_URL, load_pretrained_weight)
            self.load(load_pretrained_weight)

            for model in [self._conv1, self._conv2, self._last]:
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

    @property
    def freezed_network(self):
        return self._freezed_network

    def get_optimizer(self, current_epoch=None, total_epoch=None, current_batch=None, total_batch=None):
        """
        This returns optimizer whose learning rate is modified according to epoch.

        Args:
            current_batch:
            total_epoch:
            current_batch:
            total_batch:

        Returns:
            (Optimizer): 
        """
        if any([num is None for num in [current_epoch, total_epoch, current_batch, total_batch]]):
            return self._opt
        else:
            ind1 = int(total_epoch * 6 / 16.)
            ind2 = int(total_epoch * 3 / 16.)
            ind3 = total_epoch - ind1 - ind2
            lr_list = [0] + [0.001] * ind1 + [0.0001] * ind2 + [0.00001] * ind3
            if current_epoch == 0:
                lr = 0.0001 + (0.001 - 0.0001) / float(total_batch) * current_batch
            else:
                lr = lr_list[current_epoch]
            self._opt._lr = lr
            return self._opt

    def preprocess(self, x):
        """
        This performs preprocess for given image.

        Args:
            x(ndarray):

        Returns:
            (ndarray): Preprocessed array.
        """
        return x / 255. * 2 - 1

    def forward(self, x):
        self.freezed_network.set_auto_update(self._train_whole_network)
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
        return rm.concat(conf, px, py, pw, ph, cl).transpose(0, 2, 1, 3).reshape(N, -1, H, W)

    def regularize(self):
        """
        Regularize term of 
        """
        reg = 0
        for layer in self.iter_models():
            if hasattr(layer, "params") and hasattr(layer.params, "w"):# and isinstance(layer, rm.Conv2d):
                reg += rm.sum(layer.params.w * layer.params.w)
        return 0.0005 * reg

    def get_bbox(self, z):
        """
        This method reforms network output to list of bounding box.

        Args:
            z(Variable, ndarray):

        Returns:
            (list):
        """
        if hasattr(z, 'as_ndarray'):
            z = z.as_ndarray()

        asw = self.imsize[0] / self.anchor_size[0]
        ash = self.imsize[1] / self.anchor_size[1]
        anchor = [[an[0] * asw, an[1] * ash] for an in self.anchor]

        num_anchor = len(anchor)
        N, C, H, W = z.shape
        offset = self.num_class + 5
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
            a_box[:, 0::2] = a_box[:, 0::2] / self.imsize[0]
            a_box[:, 1::2] = a_box[:, 1::2] / self.imsize[1]

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

            keep = np.where(max_conf > 0.3)
            for i, (b, s, c) in enumerate(zip(a_box[keep[0], :, keep[1], keep[2]],
                                              max_conf[keep[0], keep[1], keep[2]],
                                              score[keep[0], :, keep[1], keep[2]])):
                b = b if isinstance(b, list) else b.tolist()
                box_list[keep[0][i]].append(b)
                score_list[keep[0][i]].append((float(s), int(np.argmax(c))))

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
                "name": self._class_map[score_list[n][i][1]].decode('utf-8'),
                "score": score_list[n][i][0],
                "class": score_list[n][i][1],
            } for i, k in enumerate(keep) if k]

        return box_list

    def predict(self, img_list):
        """
        This method performs prediction.

        Args:
            img_list(list, ndarray, string):

        Returns:
            (list):
        """
        self.set_models(inference=True)
        if isinstance(img_list, (list, str)):
            if isinstance(img_list, (tuple, list)):
                img_array = np.vstack([load_img(path, self.imsize)[None] for path in img_list])
                img_array = self.preprocess(img_array)
            else:
                img_array = load_img(img_list, self.imsize)[None]
                img_array = self.preprocess(img_array)
                return self.get_bbox(self(img_array).as_ndarray())[0]
        else:
            img_array = img_list
        return self.get_bbox(self(img_array).as_ndarray())

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
        num_class = self.num_class
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
        mask = mask.reshape(N, num_anchor, 5 + self.num_class, H, W)
        mask[:, :, 1:5, ...] = 0.1
        mask = mask.reshape(N, C, H, W)

        target = np.zeros((N, C, H, W), dtype=np.float32)
        target = target.reshape(N, num_anchor, 5 + self.num_class, H, W)
        target[:, :, 1:3, ...] = 0.5
        target[:, :, 3:5, ...] = 1.0
        target = target.reshape(N, C, H, W)

        low_thresh = 0.6
        im_w, im_h = self.imsize
        offset = 5 + self.num_class

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
                    mask[n, ind[0] * offset, ind[1], ind[2]] = 1.

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
                #    best_ious[n, best_anc_ind, h, w]

                # scale of obj iou
                mask[n, 0 + best_anc_ind * offset, h, w] = 5.

                # scale of coordinate
                mask[n, 1 + best_anc_ind * offset, h, w] = 1
                mask[n, 2 + best_anc_ind * offset, h, w] = 1
                mask[n, 3 + best_anc_ind * offset, h, w] = 1
                mask[n, 4 + best_anc_ind * offset, h, w] = 1

                # scale of class
                mask[n, 5 + best_anc_ind * offset:(best_anc_ind + 1) * offset, h, w] = 1
        diff = (x - target)
        # N = np.sum(y[:, 0] > 0)
        return rm.sum(diff * diff * mask) / N / 2.

    def fit(self, train_img_path_list=None, train_annotation_list=None,
            valid_img_path_list=None, valid_annotation_list=None,
            epoch=160, batch_size=16, augmentation=None, callback_end_epoch=None):
        """
        This function performs training with given data and hyper parameters.

        Args:
            train_img_path_list:
            train_annotation_list:
            train_image_distributor:
            valid_img_path_list:
            valid_annotation_list:
            valid_image_distributor:
            epoch:
            batch_size:
            callback_end_epoch:

        Returns:
            (tuple): Training loss list and validation loss list.
        """

        train_dist = ImageDistributor(
            train_img_path_list, train_annotation_list, augmentation=augmentation)
        valid_dist = ImageDistributor(valid_img_path_list, valid_annotation_list)

        batch_loop = int(np.ceil(len(train_dist) / batch_size))
        avg_train_loss_list = []
        avg_valid_loss_list = []
        for e in range(epoch):
            bar = tqdm(range(batch_loop))
            display_loss = 0
            for i, (train_x, train_y) in enumerate(train_dist.batch(batch_size, shuffle=True, target_builder=self.build_data)):
                self.set_models(inference=False)
                with self.train():
                    loss = self.loss(self(train_x), train_y)
                    reg_loss = loss + self.regularize()
                reg_loss.grad().update(self.get_optimizer(e, epoch, i, batch_loop))
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
                bar.n = 0
                bar.total = int(np.ceil(len(train_dist) / batch_size))
                display_loss = 0
                for i, (valid_x, valid_y) in enumerate(valid_dist.batch(batch_size, shuffle=False, target_builder=self.build_data)):
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
