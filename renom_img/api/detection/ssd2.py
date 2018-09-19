import time
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

from itertools import product as product
import numpy as np
import renom as rm

from renom_img.api.classification.vgg import VGG16
from renom_img.api.utility.load import prepare_detection_data
from renom_img.api.utility.box import transform2xy12, calc_iou_xywh


def calc_iou(prior, box):
    upleft = np.maximum(prior[:, :2], box[:2])
    bottom_right = np.minimum(prior[:, 2:4], box[2:])
    wh = bottom_right - upleft
    wh = np.maximum(wh, 0)
    inter = wh[:, 0] * wh[:, 1]
    # xmin ymin xmax ymax
    area_pred = (box[2] - box[0]) * (box[3] - box[1])
    area_gt = (prior[:, 2] - prior[:, 0]) * (prior[:, 3] - prior[:, 1])
    union = area_gt + area_pred - inter
    iou = inter / union
    return iou


class PriorBox(object):

    def __init__(self):
        self.clip = True
        self.image_size = 300
        self.feature_maps = [38, 19, 10, 5, 3, 1]
        self.min_sizes = [30, 60, 111, 162, 213, 264]
        self.max_sizes = [60, 111, 162, 213, 264, 315]
        self.steps = [8, 16, 32, 64, 100, 300]
        self.aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
        self.variance = [0.1, 0.2]

    def create(self):
        mean_boxes = []
        for k, f in enumerate(self.feature_maps):
            count = 0
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                s_k = self.min_sizes[k] / self.image_size
                mean_boxes.append([cx, cy, s_k, s_k])
                count += 1

                s_k_prime = np.sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean_boxes.append([cx, cy, s_k_prime, s_k_prime])
                count += 1

                for ar in self.aspect_ratios[k]:
                    mean_boxes.append([cx, cy, s_k * np.sqrt(ar), s_k / np.sqrt(ar)])
                    mean_boxes.append([cx, cy, s_k / np.sqrt(ar), s_k * np.sqrt(ar)])
                    count += 1
                    count += 1
            print(count/f/f, f, f)


        output = np.array(mean_boxes)
        if self.clip:
            output = np.clip(output, 0, 1)
            if False:
                plt.clf()
                img = Image.new('RGB', (self.image_size, self.image_size), (255, 255, 255))
                draw = ImageDraw.Draw(img)
                for out in output:
                    draw.rectangle(
                        (
                            int((out[0]) * self.image_size - (out[2] * self.image_size) / 2.),
                            int((out[1]) * self.image_size - (out[3] * self.image_size) / 2.),
                            int((out[0]) * self.image_size + (out[2] * self.image_size) / 2.),
                            int((out[1]) * self.image_size + (out[3] * self.image_size) / 2.),
                        ),
                        outline=(0, 0, 0)
                    )
                plt.imshow(img)
                plt.show()
        # Boxes are scaled to [0 ~ 1]

        # Change output to enter offset format.
        min_xy = output[:, :2] - output[:, 2:]/2.
        max_xy = output[:, :2] + output[:, 2:]/2.
        output = np.concatenate([min_xy, max_xy], axis=1)
        return output


class DetectorNetwork(rm.Model):

    def __init__(self, num_class, vgg):
        self.num_class = num_class
        block3 = vgg._model.block3
        self.conv3_1 = block3._layers[0]
        self.conv3_2 = block3._layers[2]
        self.conv3_3 = block3._layers[4]
        self.pool3 = rm.MaxPool2d(filter=2, stride=2, padding=1)

        self.norm = rm.L2Norm(20)

        block4 = vgg._model.block4
        self.conv4_1 = block4._layers[0]
        self.conv4_2 = block4._layers[2]
        self.conv4_3 = block4._layers[4]
        self.pool4 = rm.MaxPool2d(filter=2, stride=2)

        block5 = vgg._model.block5
        self.conv5_1 = block5._layers[0]
        self.conv5_2 = block5._layers[2]
        self.conv5_3 = block5._layers[4]
        self.pool5 = rm.MaxPool2d(filter=3, stride=1, padding=1)

        #=================================================
        # THOSE ARE USED AFTER OUTPUS ARE NORMALIZED
        self.fc6 = rm.Conv2d(channel=1024, filter=3, padding=6, dilation=6)  # relu
        self.fc7 = rm.Conv2d(channel=1024, filter=1, padding=0)

        self.conv8_1 = rm.Conv2d(channel=256, filter=1)
        self.conv8_2 = rm.Conv2d(channel=512, stride=2, filter=3, padding=1)

        self.conv9_1 = rm.Conv2d(channel=128, filter=1)
        self.conv9_2 = rm.Conv2d(channel=256, stride=2, filter=3, padding=1)

        self.conv10_1 = rm.Conv2d(channel=128, filter=1, padding=0)
        self.conv10_2 = rm.Conv2d(channel=256, padding=0, stride=1, filter=3)

        self.conv11_1 = rm.Conv2d(channel=128, filter=1, padding=0)
        self.conv11_2 = rm.Conv2d(channel=256, padding=0, stride=1, filter=3)

        num_priors = 4
        self.conv4_3_mbox_loc = rm.Conv2d(num_priors * 4, padding=1, filter=3)
        self.conv4_3_mbox_conf = rm.Conv2d(num_priors * num_class, padding=1, filter=3)
        #=================================================

        #=================================================
        num_priors = 6
        self.fc7_mbox_loc = rm.Conv2d(num_priors * 4, padding=1)
        self.fc7_mbox_conf = rm.Conv2d(num_priors * num_class, padding=1, filter=3)
        #=================================================

        #=================================================
        self.conv8_2_mbox_loc = rm.Conv2d(num_priors * 4, padding=1, filter=3)
        self.conv8_2_mbox_conf = rm.Conv2d(num_priors * num_class, padding=1, filter=3)
        #=================================================

        #=================================================
        self.conv9_2_mbox_loc = rm.Conv2d(num_priors * 4, padding=1)
        self.conv9_2_mbox_conf = rm.Conv2d(num_priors * num_class, padding=1, filter=3)
        #=================================================

        #=================================================
        num_priors = 4
        self.conv10_2_mbox_loc = rm.Conv2d(num_priors * 4, padding=1)
        self.conv10_2_mbox_conf = rm.Conv2d(num_priors * num_class, padding=1, filter=3)
        #=================================================

        #=================================================
        num_priors = 4
        self.conv11_2_mbox_loc = rm.Conv2d(num_priors * 4, padding=1)
        self.conv11_2_mbox_conf = rm.Conv2d(num_priors * num_class, padding=1, filter=3)
        #=================================================


    def forward(self, x):
        n = x.shape[0]
        t = x
        # Vgg 3rd Block
        t = rm.relu(self.conv3_1(t))
        t = rm.relu(self.conv3_2(t))
        t = rm.relu(self.conv3_3(t))
        t = self.pool3(t)

        # Vgg 4th Block
        t = rm.relu(self.conv4_1(t))
        t = rm.relu(self.conv4_2(t))
        t = rm.relu(self.conv4_3(t))

        # Normalize and compute location, confidence and priorbox aspect ratio
        conv4_norm = self.norm(t)

        conv4_norm_loc = self.conv4_3_mbox_loc(conv4_norm)
        conv4_norm_loc_flat = rm.flatten(conv4_norm_loc)
        conv4_norm_conf = self.conv4_3_mbox_conf(conv4_norm)
        conv4_norm_conf_flat = rm.flatten(conv4_norm_conf)

        # print("Norm4", conv4_norm_conf.shape, conv4_norm_loc.shape)

        t = self.pool4(t)

        # Vgg 5th Block
        t = rm.relu(self.conv5_1(t))
        t = rm.relu(self.conv5_2(t))
        t = rm.relu(self.conv5_3(t))
        t = self.pool5(t)

        # Vgg 6, 7th Block
        t = rm.relu(self.fc6(t))
        t = rm.relu(self.fc7(t))
        # Confirmed here.

        # Normalize and compute location, confidence and priorbox aspect ratio
        fc7_mbox_loc = self.fc7_mbox_loc(t)
        fc7_mbox_loc_flat = rm.flatten(fc7_mbox_loc)

        fc7_mbox_conf = self.fc7_mbox_conf(t)
        fc7_mbox_conf_flat = rm.flatten(fc7_mbox_conf)

        # print("FC7", fc7_mbox_conf.shape, fc7_mbox_loc.shape)

        t = rm.relu(self.conv8_1(t))
        t = rm.relu(self.conv8_2(t))
        # Normalize and compute location, confidence and priorbox aspect ratio
        conv8_mbox_loc = self.conv8_2_mbox_loc(t)
        conv8_mbox_loc_flat = rm.flatten(conv8_mbox_loc)

        conv8_mbox_conf = self.conv8_2_mbox_conf(t)
        conv8_mbox_conf_flat = rm.flatten(conv8_mbox_conf)

        # print("Conv8", conv8_mbox_conf.shape, conv8_mbox_loc.shape)

        t = rm.relu(self.conv9_1(t))
        t = rm.relu(self.conv9_2(t))
        # Normalize and compute location, confidence and priorbox aspect ratio
        conv9_mbox_loc = self.conv9_2_mbox_loc(t)
        conv9_mbox_loc_flat = rm.flatten(conv9_mbox_loc)

        conv9_mbox_conf = self.conv9_2_mbox_conf(t)
        conv9_mbox_conf_flat = rm.flatten(conv9_mbox_conf)

        # print("Conv9", conv9_mbox_conf.shape, conv9_mbox_loc.shape)

        t = rm.relu(self.conv10_1(t))
        t = rm.relu(self.conv10_2(t))

        conv10_mbox_loc = self.conv10_2_mbox_loc(t)
        conv10_mbox_loc_flat = rm.flatten(conv10_mbox_loc)

        conv10_mbox_conf = self.conv10_2_mbox_conf(t)
        conv10_mbox_conf_flat = rm.flatten(conv10_mbox_conf)

        # print("Conv10", conv10_mbox_conf.shape, conv10_mbox_loc.shape)

        t = rm.relu(self.conv10_1(t))
        t = rm.relu(self.conv10_2(t))

        conv11_mbox_loc = self.conv11_2_mbox_loc(t)
        conv11_mbox_loc_flat = rm.flatten(conv11_mbox_loc)

        conv11_mbox_conf = self.conv11_2_mbox_conf(t)
        conv11_mbox_conf_flat = rm.flatten(conv11_mbox_conf)

        # print("Conv11", conv11_mbox_conf.shape, conv11_mbox_loc.shape)

        mbox_loc = rm.concat([conv4_norm_loc_flat,
                              fc7_mbox_loc_flat,
                              conv8_mbox_loc_flat,
                              conv9_mbox_loc_flat,
                              conv10_mbox_loc_flat,
                              conv11_mbox_loc_flat])

        mbox_conf = rm.concat([conv4_norm_conf_flat,
                               fc7_mbox_conf_flat,
                               conv8_mbox_conf_flat,
                               conv9_mbox_conf_flat,
                               conv10_mbox_conf_flat,
                               conv11_mbox_conf_flat])

        num_boxes = mbox_loc.shape[-1] // 4
        mbox_loc = mbox_loc.reshape((n, num_boxes, 4))
        mbox_conf = mbox_conf.reshape((n, num_boxes, self.num_class))

        # print(mbox_conf.shape, mbox_loc.shape)
        predictions = rm.concat([
            mbox_loc, mbox_conf
        ], axis=2)
        return predictions


class SSD(rm.Model):

    def __init__(self, class_map=[], imsize=(300, 300),
                 overlap_threshold=0.5, load_pretrained_weight=False, train_whole_network=False):

        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)

        self.num_class = len(class_map) + 1
        self.class_map = [c.encode("ascii", "ignore") for c in class_map]

        self._train_whole_network = train_whole_network
        self.prior = PriorBox()
        self.prior_box = self.prior.create()
        self.num_prior = len(self.prior_box)
        self.overlap_threshold = overlap_threshold

        self.imsize = imsize
        vgg = VGG16(class_map, load_pretrained_weight=load_pretrained_weight)
        self._freezed_network = rm.Sequential([vgg._model.block1,
                                               vgg._model.block2])
        self._network = DetectorNetwork(self.num_class, vgg)
        self._opt = rm.Sgd(1e-3, 0.9)


    def regularize(self):
        """Regularize term. You can use this function to add regularize term to
        loss function.

        In SSD, weight decay of 0.0005 will be added.

        Example:
            >>> import numpy as np
            >>> from renom_img.api.detection.ssd import SSD
            >>> x = np.random.rand(1, 3, 300, 300)
            >>> y = np.random.rand(1, 22, 8732)
            >>> model = SSD()
            >>> t = model(x)
            >>> loss = model.loss(t, y)
            >>> reg_loss = loss + model.regularize() # Adding weight decay term.
        """

        reg = 0
        for layer in self.iter_models():
            if hasattr(layer, "params") and hasattr(layer.params, "w"):
                reg += rm.sum(layer.params.w * layer.params.w)
        return 0.00004 * reg / 2


    def preprocess(self, x):
        """Image preprocess for SSD.

        Args:
            x (ndarray):

        Returns:
            (ndarray): Preprocessed data.
        """
        x[:, 0, :, :] -= 123.68  # R
        x[:, 1, :, :] -= 116.779  # G
        x[:, 2, :, :] -= 103.939  # B
        return x

    def build_data(self):
        def builder(img_path_list, annotation_list, augmentation=None, **kwargs):
            """
            Args:
                x: Image path list.
                y: Detection formatted label.
            """
            N = len(img_path_list)
            img_data, label_data = prepare_detection_data(img_path_list,
                                                          annotation_list, self.imsize)
            if augmentation is not None:
                img_data, label_data = augmentation(img_data, label_data, mode="detection")
            targets = []
            for n in range(N):
                bounding_boxes = []
                one_hot_classes = []
                for obj in label_data[n]:
                    one_hot = np.zeros(len(self.class_map))
                    xmin, ymin, xmax, ymax = transform2xy12(obj['box'])

                    # Divide by image size
                    xmin /= self.imsize[0]
                    xmax /= self.imsize[0]
                    ymin /= self.imsize[1]
                    ymax /= self.imsize[1]

                    bounding_box = [xmin, ymin, xmax, ymax]
                    bounding_boxes.append(bounding_box)
                    one_hot[obj['class']] = 1
                    one_hot_classes.append(one_hot)
                bounding_boxes = np.asarray(bounding_boxes)
                one_hot_classes = np.asarray(one_hot_classes)
                boxes = np.hstack((bounding_boxes, one_hot_classes))
                target = self.assign_boxes(boxes)
                targets.append(target)
            # target (N, class, prior box)
            return self.preprocess(img_data), np.array(targets)
        return builder

    def assign_boxes(self, boxes):
        # background is already included in self.num_class.
        assignment = np.zeros((self.num_prior, 5 + self.num_class))
        assignment[:, 4] = 1.0  # background(This means id=0 is background)
        if len(boxes) == 0:
            return assignment
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])

        # (box_num, prior_num, (xmin ymin xmax ymax iou))
        encoded_boxes = encoded_boxes.reshape(-1, self.num_prior, 5)  

        ### DEBUG
        if False:
            img = Image.new('RGB', (300, 300), (255, 255, 255))
            draw = ImageDraw.Draw(img)
            for assigned in encoded_boxes:
                use = assigned[:, 4] > 0
                decoded = self.decode_box(assigned[:, :4])
                for d in decoded[use]:
                    draw.rectangle(tuple(int(b*300) for b in d), outline=(0, 0, 0))
            plt.imshow(img)
            plt.show()
            exit()
        ###

        best_iou = encoded_boxes[:, :, -1].max(axis=0) # get the best fit target for each prior.
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)

        # Cut background
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]

        # Assign conf score
        assignment[:, 0][best_iou_mask] = encoded_boxes[best_iou_idx, best_iou_mask, 4]
        # Assign loc
        assignment[:, 1:5][best_iou_mask] = encoded_boxes[best_iou_idx, best_iou_mask, :4]
        # Assign class
        assignment[:, 5][~best_iou_mask] = 1 # Background.
        assignment[:, 6:][best_iou_mask] = boxes[best_iou_idx, 4:]
        return assignment

    def encode_box(self, box):
        # prior box is point format(xmin, ymin, xmax, ymax).
        # box is center point format(xmin, ymin, xmax, ymax).
        iou = calc_iou(self.prior_box, box)
        encoded_box = np.zeros((self.num_prior, 4 + 1))
        assign_mask = iou > self.overlap_threshold

        if not assign_mask.any():
            assign_mask[iou.argmax()] = True

        encoded_box[:, -1][assign_mask] = iou[assign_mask]

        assigned_priors = self.prior_box[assign_mask]

        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        assigned_priors_center = 0.5 * (assigned_priors[:, :2] + assigned_priors[:, 2:4])
        assigned_priors_wh = assigned_priors[:, 2:4] - assigned_priors[:, :2]

        ### DEBUG
        if False:
            img = Image.new('RGBA', (300, 300), (255, 255, 255, 1))
            draw = ImageDraw.Draw(img)
            print([int(b*300) for b in box], len(assigned_priors))
            draw.rectangle(tuple(int(b*300) for b in box), fill=(0, 0, 0, 255))
            for assigned in assigned_priors:
                draw.rectangle(tuple(int(b*300) for b in assigned), outline=(0, 0, 0, 150))
            plt.imshow(img)
            plt.show()
        ###

        # Encode xy
        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= (assigned_priors_wh*self.prior.variance[0])

        # Encode wh
        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_priors_wh)/self.prior.variance[1]
        return encoded_box.ravel()


    def decode_box(self, loc):
        prior = self.prior_box
        prior_wh = prior[:, 2:] - prior[:, :2]
        prior_xy = prior[:, :2] + prior_wh/2.
        boxes = np.concatenate([
            prior_xy + loc[:, :2]*self.prior.variance[0] * prior_wh,
            prior_wh * np.exp(loc[:, 2:]*self.prior.variance[1]),
        ], axis=1)
        boxes[:, :2] -= boxes[:, 2:] / 2.
        boxes[:, 2:] += boxes[:, :2]
        return boxes


    def forward(self, x):
        self._freezed_network.set_auto_update(self._train_whole_network)
        return self._network(self._freezed_network(x))


    def loss(self, x, y, neg_pos_ratio=3.0, negatives_for_hard=100.0):
        pos_samples = (y[:, :, 0] > 0).astype(np.float32)[..., None]
        N = np.sum(pos_samples)
        pos_Ns = np.sum(pos_samples, axis=1)
        neg_Ns = np.clip(neg_pos_ratio*pos_Ns, 0, y.shape[1])

        # Loc loss
        loc_loss = rm.sum(rm.smoothed_l1(x[..., :4]*pos_samples, y[..., 1:5]*pos_samples, reduce_sum=False))

        # this is for hard negative mining.
        loss_c = np.max(x[..., 4:].as_ndarray(), axis=2)
        loss_c[pos_samples.astype(np.bool)[..., 0]] = 0
        sorted_index = np.argsort(loss_c, axis=1)[:, ::-1]
        index_rank = np.argsort(sorted_index, axis=1)
        neg_samples = index_rank < neg_Ns
        samples = (neg_samples[..., None] + pos_samples).astype(np.bool)
        conf_loss = rm.sum(rm.softmax_cross_entropy((x[..., 4:]*samples).transpose(0, 2, 1), 
                        (y[..., 5:]*samples).transpose(0, 2, 1), reduce_sum=False))

        loss = conf_loss + loc_loss
        return loss/N


    def predict(self, img):
        z = self(img).as_ndarray()
        return self.get_bbox(z)


    def get_bbox(self, z, score_threshold=0.3, nms_threshold=0.4):
        loc, conf = np.split(z, [4], axis=2)
        conf = rm.softmax(conf.transpose(0, 2, 1)).as_ndarray().transpose(0, 2, 1)

        result_bbox = []
        for n, (l, c) in enumerate(zip(loc, conf)):
            decoded = self.decode_box(l)
            keep_flag = (np.min(c, axis=1) > score_threshold) * (np.argmax(c, axis=1) > 0)
            class_score_list = c[keep_flag]
            loc_candidate_list = decoded[keep_flag]
            # To center form
            loc_candidate_list[:, 2:] = loc_candidate_list[:, 2:] - loc_candidate_list[:, :2]
            loc_candidate_list[:, :2] = loc_candidate_list[:, 2:]/2.
            class_score_list = [(np.max(cl), np.argmax(cl)) for cl in class_score_list]

            # NMS 
            sorted_ind = np.argsort([s[0] for s in class_score_list])[::-1]
            keep = np.ones((len(class_score_list),), dtype=np.bool)
            for i, ind1 in enumerate(sorted_ind):
                if not keep[ind1]:
                    continue
                box1 = loc_candidate_list[ind1]
                for j, ind2 in enumerate(sorted_ind[i + 1:], i + 1):
                    box2 = loc_candidate_list[ind2]
                    if keep[ind2] and class_score_list[ind1][1] == class_score_list[ind2][1]:
                        keep[ind2] = calc_iou_xywh(box1, box2) < nms_threshold

            result_bbox.append([{
                "box": loc_candidate_list[i].tolist(),
                "name": self.class_map[class_score_list[i][1] - 1].decode('utf-8'),
                "score": class_score_list[i][0],
                "class": class_score_list[i][1],
            } for i, k in enumerate(keep) if k])
            print(result_bbox[-1])
        return result_bbox



    def get_optimizer(self, current_epoch=None, total_epoch=None, current_batch=None, total_batch=None):
        """Returns an instance of Optimiser for training SSD algorithm.

        Args:
            current_epoch:
            total_epoch:
            current_batch:
            total_epoch:
        """
        if current_epoch == 60:
            self._opt._lr = 1e-4
        elif current_epoch == 100:
            self._opt._lr = 1e-5
        return self._opt



if __name__ == "__main__":
    import os
    from renom.cuda import set_cuda_active
    from renom_img.api.utility.load import parse_xml_detection
    from renom_img.api.utility.misc.display import draw_box

    set_cuda_active(True)
    img_path = "../../../example/VOCdevkit/VOC2007/JPEGImages/"
    label_path = "../../../example/VOCdevkit/VOC2007/Annotations/"
    img_list = [os.path.join(img_path, p) for p in sorted(os.listdir(img_path))[:10]]
    lbl_list = [os.path.join(label_path, p) for p in sorted(os.listdir(label_path))[:10]]
    annotation_list, class_map = parse_xml_detection(lbl_list)

    ssd = SSD(class_map, load_pretrained_weight=True)
    builder = ssd.build_data()
    img, lbl = builder(img_list, annotation_list)
    for i in range(1000):
        ssd.set_models(inference=False)
        with ssd.train():
            loss = ssd.loss(ssd(img), lbl)
        loss.grad().update(ssd.get_optimizer())
        print(i, loss.as_ndarray())
        if i < 990: continue
        ssd.set_models(inference=True)
        plt.imshow(draw_box(img_list[0], ssd.predict(img)[0]))
        plt.show()
