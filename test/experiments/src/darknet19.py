import numpy as np
import renom as rm

from utils import calc_iou_xywh


class DarknetConv2d(rm.Model):

    def __init__(self, channel, filter=3):
        pad = int((filter - 1) / 2)
        self._conv = rm.Conv2d(channel=channel, filter=filter, padding=pad)

    def forward(self, x):
        return rm.leaky_relu(self._conv(x), 0.1)


class DarknetConv2dBN(rm.Model):

    def __init__(self, channel, filter=3, conv_weight=None, bn_weight=None, bn_bias=None, bn_mean=None, bn_var=None):
        pad = int((filter - 1) / 2)
        self._conv = rm.Conv2d(channel=channel, filter=filter, padding=pad, ignore_bias=True)
        self._bn = rm.BatchNormalize(mode='feature', momentum=0.01)

    def forward(self, x):
        return rm.leaky_relu(self._bn(self._conv(x)), 0.1)

class Darknet19Base(rm.Model):

    def __init__(self):
        self.block1 = rm.Sequential([
            DarknetConv2dBN(32),
            rm.MaxPool2d(filter=2, stride=2)
        ])
        self.block2 = rm.Sequential([
            DarknetConv2dBN(64),
            rm.MaxPool2d(filter=2, stride=2)
        ])
        self.block3 = rm.Sequential([
            DarknetConv2dBN(128),
            DarknetConv2dBN(64, filter=1),
            DarknetConv2dBN(128),
            rm.MaxPool2d(filter=2, stride=2)
        ])
        self.block4 = rm.Sequential([
            DarknetConv2dBN(256),
            DarknetConv2dBN(128, filter=1),
            DarknetConv2dBN(256),
            rm.MaxPool2d(filter=2, stride=2)
        ])
        self.block5 = rm.Sequential([
            DarknetConv2dBN(512),
            DarknetConv2dBN(256, filter=1),
            DarknetConv2dBN(512),
            DarknetConv2dBN(256, filter=1),

            DarknetConv2dBN(512),
        ])

        self.block6 = rm.Sequential([
            # For concatenation.
            rm.MaxPool2d(filter=2, stride=2),
            DarknetConv2dBN(1024),
            DarknetConv2dBN(512, filter=1),
            DarknetConv2dBN(1024),
            DarknetConv2dBN(512, filter=1),
            DarknetConv2dBN(1024),
        ])

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        h = self.block3(h)
        h = self.block4(h)
        f = self.block5(h)
        h = self.block6(f)
        return h, f

class Darknet19Classification(rm.Model):

    def __init__(self):
        self._base = Darknet19Base()
        self._last = rm.Conv2d(1000, filter=1, ignore_bias=True)

    def forward(self, x):
        N = len(x)
        h, _ = self._base(x)
        #h = rm.average_pool2d(h, h.shape[2])
        #h = rm.flatten(h)
        D = h.shape[2] * h.shape[3]
        h = rm.sum(self._last(h).reshape(N, 1000, -1), axis=2)
        h /= D
        return h

class Darknet19Detection(rm.Model):

    def __init__(self, num_class=20, num_anchor=5):
        self.num_anchor = num_anchor
        self.cn = num_class

        last_channel = (num_class + 5) * self.num_anchor
        self._base = Darknet19Base()
        self._conv1 = rm.Sequential([
            DarknetConv2dBN(channel=1024),
            DarknetConv2dBN(channel=1024),
        ])
        self._conv2 = DarknetConv2dBN(channel=1024,)
        self._last = rm.Conv2d(channel=last_channel, filter=1)
        # Load base weight here.

    def forward(self, x):
        h, f = self._base(x)
        h = self._conv1(h)
        # f = f.as_ndarray()
        # h = h.as_ndarray()
        h = self._conv2(rm.concat(h,
            rm.concat([f[:, :, i::2, j::2] for i in range(2) for j in range(2)])))
        out = self._last(h)

        # Create yolo format.
        N, C, H, W = h.shape
        reshaped = out.reshape(N, self.num_anchor, -1, W*H)
        conf = rm.sigmoid(reshaped[:, :, 0:1]).transpose(0, 2, 1, 3)
        px = rm.sigmoid(reshaped[:, :, 1:2]).transpose(0, 2, 1, 3)
        py = rm.sigmoid(reshaped[:, :, 2:3]).transpose(0, 2, 1, 3)
        pw = rm.exp(reshaped[:, :, 3:4]).transpose(0, 2, 1, 3)
        ph = rm.exp(reshaped[:, :, 4:5]).transpose(0, 2, 1, 3)
        cl = rm.softmax(reshaped[:, :, 5:].transpose(0, 2, 1, 3))
        out = rm.concat(conf, px, py, pw, ph, cl).transpose(0, 2, 1, 3).reshape(N, -1, H, W)
        return out

    def weight_decay(self):
        reg = 0
        for layer in self.iter_models():
            if hasattr(layer.params, "w") and isinstance(layer, rm.Conv2d):
                reg += rm.sum(layer.params.w * layer.params.w)
        return 0.0005*reg

    def build_target(self, x, y, anchor, img_size):
        """
        Returns mask.
        Args: 
            x: Yolo output. (N, C(anc*(5+class)), H, W)
            y: (N, C(5+class), H(feature), W(feature))
        """
        N, C, H, W = x.shape
        num_anchor = self.num_anchor
        mask = np.zeros((N, C, H, W), dtype=np.float32)
        mask = mask.reshape(N, num_anchor, 5+self.cn, H, W)
        mask[:, :, 1:5, ...] += 0.1
        mask = mask.reshape(N, C, H, W)

        target = np.zeros((N, C, H, W), dtype=np.float32)
        target = target.reshape(N, num_anchor, 5+self.cn, H, W)
        target[:, :, 1:3, ...] = 0.5
        target[:, :, 3:5, ...] = 1.0
        target = target.reshape(N, C, H, W)

        low_thresh = 0.6
        im_w, im_h = img_size
        offset = 5 + self.cn

        # Calc iou and get best matched prediction.
        best_ious = np.zeros((N, num_anchor, H, W), dtype=np.float32)
        best_anchor_ious = np.zeros((N, 1, H, W), dtype=np.float32)
        for n in range(N):
            gt_index = np.where(y[n, 0] > 0)

            # Create mask for prediction that 
            for ind in np.ndindex((num_anchor, H, W)):
                max_iou = -1
                px = (x[n, 1+ind[0]*offset, ind[1], ind[2]] + ind[2]) * im_w/W
                py = (x[n, 2+ind[0]*offset, ind[1], ind[2]] + ind[1]) * im_h/H
                pw = x[n, 3+ind[0]*offset, ind[1], ind[2]] * anchor[ind[0]][0] #*im_w/W
                ph = x[n, 4+ind[0]*offset, ind[1], ind[2]] * anchor[ind[0]][1] #*im_h/H
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
                #     mask[n, ind[0]*offset, ind[1], ind[2]] = \
                #             (0 - x[n, ind[0]*offset, ind[1], ind[2]])*1
                    mask[n, ind[0]*offset, ind[1], ind[2]] = 1

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
                target[n, 1+best_anc_ind*offset, h, w] = (tx/32.)%1
                target[n, 2+best_anc_ind*offset, h, w] = (ty/32.)%1

                # Don't need to divide by 32 because anchor is already rescaled to input image size.
                target[n, 3+best_anc_ind*offset, h, w] = tw/anchor[best_anc_ind][0]
                target[n, 4+best_anc_ind*offset, h, w] = th/anchor[best_anc_ind][1]

                # target of class
                target[n, 5+best_anc_ind*offset:(best_anc_ind+1)*offset, h, w] = \
                        y[n, 5:offset, h, w]

                # target of iou.
                target[n, 0+best_anc_ind*offset, h, w] = \
                        best_ious[n, best_anc_ind, h, w]

                # scale of obj iou
                # mask[n, 0+best_anc_ind*offset, h, w] = \
                #         (1 - best_ious[n, best_anc_ind, h, w])*5.
                mask[n, 0+best_anc_ind*offset, h, w] = 5.

                # scale of coordinate
                mask[n, 1+best_anc_ind*offset, h, w] = 1
                mask[n, 2+best_anc_ind*offset, h, w] = 1
                mask[n, 3+best_anc_ind*offset, h, w] = 1
                mask[n, 4+best_anc_ind*offset, h, w] = 1

                # scale of class
                mask[n, 5+best_anc_ind*offset:(best_anc_ind+1)*offset, h, w] = 1

        return target, mask

    def transform_to_original_scale(self, pred, anchor, img_size):
        """
        Returns:
            Original Scaled bbox(x1y1x2y2)
        """
        # Anchors must be resized.
        num_anchor = len(anchor)
        N, C, H, W = pred.shape
        offset = self.cn + 5
        FW, FH = img_size[0]//32, img_size[1]//32
        box_list = [[] for n in range(N)]
        score_list = [[] for n in range(N)]

        for ind_a, anc in enumerate(anchor):
            a_pred = pred[:, ind_a*offset:(ind_a+1)*offset]
            score = a_pred[:, 0].reshape(N, 1, H, W)
            cls_score = a_pred[:, 5:]
            score = score*cls_score
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
            keep = np.where(max_conf > 0)
            for i, (b, s, c) in enumerate(zip(a_box[keep[0], :, keep[1], keep[2]], 
                                max_conf[keep[0], keep[1], keep[2]],
                                score[keep[0], :, keep[1], keep[2]])):
                box_list[keep[0][i]].append(b)
                score_list[keep[0][i]].append((s, np.argmax(c)))

        ### NMS
        for n in range(N):
            sorted_ind = np.argsort([s[0] for s in score_list[n]])[::-1]
            keep = np.ones((len(score_list[n]),), dtype=np.bool)
            for i, ind1 in enumerate(sorted_ind):
                if not keep[i]: continue
                box1 = box_list[n][ind1]
                for j, ind2 in enumerate(sorted_ind[i+1:]):
                    box2 = box_list[n][ind2]
                    if keep[j] and score_list[n][ind1][1] == score_list[n][ind2][1]:
                        keep[j] = calc_iou_xywh(box1, box2) < 0.4

            box_list[n] = [box_list[n][i] for i, k in enumerate(keep) if k]
            score_list[n] = [score_list[n][i] for i, k in enumerate(keep) if k]

        return box_list, score_list
