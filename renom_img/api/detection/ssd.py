import os
import numpy as np
import renom as rm
from tqdm import tqdm
from PIL import Image

from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.misc.download import download
from renom_img.api.utility.box import transform2xy12
from renom_img.api.utility.misc.display import draw_box
from renom_img.api.utility.load import prepare_detection_data, load_img
from renom_img.api.classification.vgg import VGG16

img_width, img_height = 300, 300

boxes_paras = []
def create_priors():
    box_configs = [{'layer_width': 38, 'layer_height': 38, 'num_prior': 3, 'min_size':  30.0,
         'max_size': None, 'aspect_ratios': [1.0, 2.0, 1/2.0]},
        {'layer_width': 19, 'layer_height': 19, 'num_prior': 6, 'min_size':  60.0,
         'max_size': 114.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
        {'layer_width': 10, 'layer_height': 10, 'num_prior': 6, 'min_size': 114.0,
         'max_size': 168.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
        {'layer_width':  5, 'layer_height':  5, 'num_prior': 6, 'min_size': 168.0,
         'max_size': 222.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
        {'layer_width':  3, 'layer_height':  3, 'num_prior': 6, 'min_size': 222.0,
         'max_size': 276.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]},
        {'layer_width':  1, 'layer_height':  1, 'num_prior': 6, 'min_size': 276.0,
         'max_size': 330.0, 'aspect_ratios': [1.0, 1.0, 2.0, 1/2.0, 3.0, 1/3.0]}]

    variance = [0.1, 0.1, 0.2, 0.2]
    for config in box_configs:
        layer_width = config['layer_width']
        layer_height = config['layer_height']
        num_priors = config['num_prior']
        min_size = config['min_size']
        max_size = config['max_size']
        aspect_ratios = config['aspect_ratios']

        step_x = float(img_width) / float(layer_width)
        step_y = float(img_height) / float(layer_height)

        linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x, layer_width)
        liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y, layer_height)

        center_x, center_y = np.meshgrid(linx, liny) #38*38
        center_x = center_x.reshape((-1, 1)) #1441
        center_y = center_y.reshape((-1, 1)) #1441

        prior_boxes = np.concatenate((center_x, center_y), axis=1)
        prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors))

        box_widths = []
        box_heights = []
        for ar in aspect_ratios:
            if ar == 1. and len(box_widths) == 0:
                box_widths.append(min_size)
                box_heights.append(min_size)
            elif ar == 1. and len(box_widths) > 0:
                length = np.sqrt(min_size * max_size)
                box_widths.append(length)
                box_heights.append(length)
            elif ar != 1.:
                box_widths.append(min_size*np.sqrt(ar))
                box_heights.append(min_size/np.sqrt(ar))
        box_widths = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)

        prior_boxes[:, ::4] -= box_widths
        prior_boxes[:, 1::4] -= box_heights
        prior_boxes[:, 2::4] += box_widths
        prior_boxes[:, 3::4] += box_heights

        prior_boxes[:, ::2] /= img_width
        prior_boxes[:, 1::2] /= img_height
        prior_boxes = prior_boxes.reshape((-1, 4))

        prior_boxes = np.minimum(np.maximum(prior_boxes, 0.), 1.)
        prior_variance = np.tile(variance, (len(prior_boxes), 1))
        boxes_para = np.concatenate((prior_boxes, prior_variance), axis=1)
        boxes_paras.append(boxes_para)

    return np.concatenate(boxes_paras, axis=0)

def calc_iou(prior, box):
    upleft = np.maximum(prior[:, :2], box[:2])
    bottom_right = np.minimum(prior[:, 2:4], box[2:])
    wh = bottom_right - upleft
    wh = np.maximum(wh, 0)
    inter = wh[:, 0] * wh[:, 1]
    #xmin ymin xmax ymax
    area_pred = (box[2]- box[0]) * (box[3] - box[1])
    area_gt = (prior[:, 2] - prior[:, 0]) * (prior[:, 3] - prior[:, 1])
    union = area_gt + area_pred - inter
    iou = inter / union
    return iou

class PriorBox(object):
    def __init__(self, img_size, min_size, max_size=None, aspect_ratios=None,
                 flip=True, variances=[0.1], clip=True):
        self.img_size = img_size
        self.min_size = min_size
        self.max_size = max_size
        self.aspect_ratios = [1.0]
        if max_size:
            self.aspect_ratios.append(1.0)
        if aspect_ratios:
            for ar in aspect_ratios:
                if ar in self.aspect_ratios:
                    continue
                self.aspect_ratios.append(ar)
                if flip:
                    self.aspect_ratios.append(1.0 / ar)
        self.variances = np.array(variances)
        self.clip = True

    def __call__(self, x, mask=None):
        x = x.as_ndarray()
        input_shape = x.shape
        layer_width = input_shape[3]
        layer_height = input_shape[2]
        img_width = self.img_size[0]
        img_height = self.img_size[1]

        # define prior boxes shapes
        box_widths = []
        box_heights = []

        for ar in self.aspect_ratios:
            # necessary?
            if ar == 1 and len(box_widths) == 0:
                box_widths.append(self.min_size)
                box_heights.append(self.min_size)
            elif ar == 1 and len(box_widths) > 0:
                box_widths.append(np.sqrt(self.min_size * self.max_size))
                box_heights.append(np.sqrt(self.min_size * self.max_size))
            elif ar != 1:
                box_widths.append(self.min_size * np.sqrt(ar))
                box_heights.append(self.min_size / np.sqrt(ar))

        box_widths = 0.5 * np.array(box_widths)
        box_heights = 0.5 * np.array(box_heights)
        # define centers of prior boxes

        step_x = img_width / layer_width
        step_y = img_height / layer_height

        linx = np.linspace(0.5 * step_x, img_width - 0.5 * step_x,
                           layer_width)
        liny = np.linspace(0.5 * step_y, img_height - 0.5 * step_y,
                           layer_height)
        centers_x, centers_y = np.meshgrid(linx, liny)
        centers_x = centers_x.reshape(-1, 1)
        centers_y = centers_y.reshape(-1, 1)
        # define xmin, ymin, xmax, ymax of prior boxes
        num_priors_ = len(self.aspect_ratios)
        prior_boxes = np.concatenate((centers_x, centers_y), axis=1)
        prior_boxes = np.tile(prior_boxes, (1, 2 * num_priors_))
        prior_boxes[:, ::4] -= box_widths
        prior_boxes[:, 1::4] -= box_heights
        prior_boxes[:, 2::4] += box_widths
        prior_boxes[:, 3::4] += box_heights
        prior_boxes[:, ::2] /= img_width
        prior_boxes[:, 1::2] /= img_height
        prior_boxes = prior_boxes.reshape(-1, 4)
        if self.clip:
            prior_boxes = np.minimum(np.maximum(prior_boxes, 0.0), 1.0)
        # define variances
        num_boxes = len(prior_boxes)
        if len(self.variances) == 1:
            variances = np.ones((num_boxes, 4)) * self.variances[0]
        elif len(self.variances) == 4:
            variances = np.tile(self.variances, (num_boxes, 1))
        else:
            raise Exception('Must provide one or four variances')

        prior_boxes = np.concatenate((prior_boxes, variances), axis=1)
        prior_boxes_tensor = np.expand_dims(prior_boxes, 0)
        return prior_boxes_tensor

class DetectorNetwork(rm.Model):
    def __init__(self, num_class, vgg):
        self.num_class = num_class
        block3 = vgg._model.block3
        self.conv3_1 = block3._layers[0]
        self.conv3_2 = block3._layers[2]
        self.conv3_3 = block3._layers[4]
        self.pool3 = rm.MaxPool2d(filter=2, stride=2, padding=1)

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
        self.fc6 = rm.Conv2d(channel=1024, filter=3, padding=6, dilation=6) #relu
        self.fc7 = rm.Conv2d(channel=1024, filter=1, padding=0)

        self.conv8_1 = rm.Conv2d(channel=256, filter=1)
        self.conv8_2 = rm.Conv2d(channel=512, stride=2, filter=3, padding=1)

        self.conv9_1 = rm.Conv2d(channel=128, filter=1)
        self.conv9_2 = rm.Conv2d(channel=256, stride=2, filter=3, padding=1)

        self.conv10_1 = rm.Conv2d(channel=128, filter=1, padding=0)
        self.norm = rm.L2Norm(20)
        self.conv10_2 = rm.Conv2d(channel=256, padding=1, stride=2, filter=3)

        num_priors = 3
        self.conv4_3_mbox_loc = rm.Conv2d(num_priors * 4, padding=1, filter=3)

        self.conv4_3_mbox_conf = rm.Conv2d(num_priors * num_class, padding=1, filter=3)
#        define the PriorBox klass later
        self.conv4_3_priorbox = PriorBox((300, 300), 30.0, aspect_ratios=[2], variances=[0.1, 0.1, 0.2, 0.2])
        #=================================================



        #=================================================
        num_priors = 6
        self.fc7_mbox_loc = rm.Conv2d(num_priors*4, padding=1)
        self.fc7_mbox_conf = rm.Conv2d(num_priors*num_class, padding=1, filter=3)
        self.fc7_priorbox = PriorBox((300, 300), 114.0, max_size=168.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2])
        #=================================================


        #=================================================
        self.conv8_2_mbox_loc = rm.Conv2d(num_priors*4, padding=1, filter=3)
        self.conv8_2_mbox_conf = rm.Conv2d(num_priors*num_class, padding=1, filter=3)
        self.conv8_2_priorbox = PriorBox((300, 300), 114.0, max_size=168.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2])
        #=================================================


        #=================================================
        self.conv9_2_mbox_loc = rm.Conv2d(num_priors*4, padding=1)
        self.conv9_2_mbox_conf = rm.Conv2d(num_priors*num_class, padding=1, filter=3)
        self.conv9_2_priorbox = PriorBox((300, 300), 168.0, max_size=222.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2])
        #=================================================


        #=================================================
        self.conv10_2_mbox_loc = rm.Conv2d(num_priors*4, padding=1)
        self.conv10_2_mbox_conf = rm.Conv2d(num_priors*num_class, padding=1, filter=3)
        self.conv10_2_priorbox = PriorBox((300, 300), 222.0, max_size=276.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2])
        #=================================================

        self.pool11_mbox_loc = rm.Dense(num_priors*4)
        self.pool11_mbox_conf = rm.Dense(num_priors*num_class)

        self.pool11_priorbox = PriorBox((300, 300), 276.0, max_size=330.0, aspect_ratios=[2, 3], variances=[0.1, 0.1, 0.2, 0.2])


    def forward(self, x):
        n = x.shape[0]
        t = x
        t = self.pool3(rm.relu(self.conv3_3(rm.relu(self.conv3_2(rm.relu(self.conv3_1(t)))))))
        t = rm.relu(self.conv4_3(rm.relu(self.conv4_2(rm.relu(self.conv4_1(t))))))

        # Normalize and compute location, confidence and priorbox aspect ratio
        conv4_norm = self.norm(t)
        #conv4_norm = t
        conv4_norm_loc = self.conv4_3_mbox_loc(conv4_norm)
        conv4_norm_loc_flat = rm.flatten(conv4_norm_loc)
        conv4_norm_conf = self.conv4_3_mbox_conf(conv4_norm)
        conv4_norm_conf_flat = rm.flatten(conv4_norm_conf)
        conv4_priorbox = self.conv4_3_priorbox(conv4_norm)

        t = self.pool4(t)

        t = self.pool5(rm.relu(self.conv5_3(rm.relu(self.conv5_2(rm.relu(self.conv5_1(t)))))))

        t = rm.relu(self.fc6(t))
        t = rm.relu(self.fc7(t))

        # Normalize and compute location, confidence and priorbox aspect ratio
        fc7_mbox_loc = self.fc7_mbox_loc(t)
        fc7_mbox_loc_flat = rm.flatten(fc7_mbox_loc)

        fc7_mbox_conf = self.fc7_mbox_conf(t)
        fc7_mbox_conf_flat = rm.flatten(fc7_mbox_conf)
        fc7_priorbox = self.fc7_priorbox(t)

        t = rm.relu(self.conv8_2(rm.relu(self.conv8_1(t))))
        # Normalize and compute location, confidence and priorbox aspect ratio
        conv8_mbox_loc = self.conv8_2_mbox_loc(t)
        conv8_mbox_loc_flat = rm.flatten(conv8_mbox_loc)

        conv8_mbox_conf = self.conv8_2_mbox_conf(t)
        conv8_mbox_conf_flat = rm.flatten(conv8_mbox_conf)
        conv8_priorbox = self.conv8_2_priorbox(t)

        t = rm.relu(self.conv9_2(rm.relu(self.conv9_1(t))))
        # Normalize and compute location, confidence and priorbox aspect ratio
        conv9_mbox_loc = self.conv9_2_mbox_loc(t)
        conv9_mbox_loc_flat = rm.flatten(conv9_mbox_loc)

        conv9_mbox_conf = self.conv9_2_mbox_conf(t)
        conv9_mbox_conf_flat = rm.flatten(conv9_mbox_conf)
        conv9_priorbox = self.conv9_2_priorbox(t)

        t = rm.relu(self.conv10_2(rm.relu(self.conv10_1(t))))
        conv10_mbox_loc = self.conv10_2_mbox_loc(t)
        conv10_mbox_loc_flat = rm.flatten(conv10_mbox_loc)

        conv10_mbox_conf = self.conv10_2_mbox_conf(t)
        conv10_mbox_conf_flat = rm.flatten(conv10_mbox_conf)
        conv10_priorbox = self.conv10_2_priorbox(t)

        t = rm.average_pool2d(t)
        t = rm.flatten(t)

        pool11_mbox_loc_flat = self.pool11_mbox_loc(t)


        pool11_mbox_conf_flat = self.pool11_mbox_conf(t)
        pool11_reshaped = t.reshape((t.shape[0], 256, 1, 1))
        pool11_priorbox = self.pool11_priorbox(pool11_reshaped)

        mbox_loc = rm.concat([conv4_norm_loc_flat,
                              fc7_mbox_loc_flat,
                              conv8_mbox_loc_flat,
                              conv9_mbox_loc_flat,
                              conv10_mbox_loc_flat,
                              pool11_mbox_loc_flat])
        mbox_conf = rm.concat([conv4_norm_conf_flat,
                              fc7_mbox_conf_flat,
                              conv8_mbox_conf_flat,
                              conv9_mbox_conf_flat,
                              conv10_mbox_conf_flat,
                              pool11_mbox_conf_flat])

        mbox_priorbox = np.concatenate([conv4_priorbox,
                              fc7_priorbox,
                              conv8_priorbox,
                              conv9_priorbox,
                              conv10_priorbox,
                              pool11_priorbox], axis=1)




        num_boxes = mbox_loc.shape[-1]//4
        mbox_loc = mbox_loc.reshape((n, 4, num_boxes))
        mbox_conf = mbox_conf.reshape((n, self.num_class, num_boxes))

        predictions = rm.concat([
            mbox_loc, mbox_conf, np.broadcast_to(mbox_priorbox.transpose((0, 2, 1)), (mbox_conf.shape[0], mbox_priorbox.shape[2], mbox_priorbox.shape[1]))
        ])
        return predictions


class SSD(rm.Model):
    """ SSD object detection algorithm.

    Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg
    SSD: Single Shot MultiBox Detector
    https://arxiv.org/abs/1512.02325

    Args:
        num_class (int): Number of class.
        imsize (int, tuple): Image size.
        load_pretrained_weight (bool, str): If true, pretrained weight will be downloaded to current directory.
            If string is given, pretrained weight will be saved as given name.
    """

    SERIALIZED = ("class_map", "num_class", "imsize")
    WEIGHT_URL = "http://docs.renom.jp/downloads/weights/Yolov1.h5"

    def __init__(self, class_map=None, imsize=(300, 300), overlap_threshold=0.5, load_pretrained_weight=False, train_whole_network=False):
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)

        self.num_class = len(class_map) + 1
        self.class_map = class_map
        self._train_whole_network = train_whole_network
        self.prior = create_priors()
        self.num_prior = len(self.prior)
        self.overlap_threshold = overlap_threshold

        self.imsize = imsize
        vgg = VGG16(class_map)
        vgg._model.load('VGG16-2.h5')
        self._freezed_network = rm.Sequential([vgg._model.block1,
                                               vgg._model.block2])
        self._network = DetectorNetwork(self.num_class, vgg)

        self._opt = rm.Sgd(1e-3, 0.9)

        if load_pretrained_weight:
            if isinstance(load_pretrained_weight, bool):
                load_pretrained_weight = self.__class__.__name__ + '.h5'

            if not os.path.exists(load_pretrained_weight):
                download(self.WEIGHT_URL, load_pretrained_weight)

            self.load(load_pretrained_weight)
            for layer in self._network.iter_models():
                layer.params = {}

    @property
    def freezed_network(self):
        return self._freezed_network

    @property
    def network(self):
        return self._network

    def get_optimizer(self, current_epoch=None, total_epoch=None, current_batch=None, total_batch=None):
        """Returns an instance of Optimiser for training Yolov1 algorithm.

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

    def preprocess(self, x):
        """Image preprocess for Yolov1.

        :math:`new_x = x*2/255. - 1`

        Args:
            x (ndarray):

        Returns:
            (ndarray): Preprocessed data.
        """
        x[:, 0, :, :] -= 123.68  # R
        x[:, 1, :, :] -= 116.779  # G
        x[:, 2, :, :] -= 103.939  # B
        return x

    def forward(self, x):
        self.freezed_network.set_auto_update(self._train_whole_network)
        return self.network(self.freezed_network(x))

    def regularize(self):
        """Regularize term. You can use this function to add regularize term to
        loss function.

        In Yolo v1, weight decay of 0.0005 will be added.

        Example:
            >>> import numpy as np
            >>> from renom_img.api.detection.yolo_v1 import Yolov1
            >>> x = np.random.rand(1, 3, 224, 224)
            >>> y = np.random.rand(1, (5*2+20)*7*7)
            >>> model = Yolov1()
            >>> loss = model.loss(x, y)
            >>> reg_loss = loss + model.regularize() # Adding weight decay term.
        """

        reg = 0
        for layer in self.iter_models():
            if hasattr(layer, "params") and hasattr(layer.params, "w"):
                reg += rm.sum(layer.params.w * layer.params.w)
        return 0.0005 * reg

    def get_bbox(self, z, score_threshold=0.3, nms_threshold=0.4, keep_top_k=200):
        """
        Example:
            >>> z = model(x)
            >>> model.get_bbox(z)
            [[{'box': [0.21, 0.44, 0.11, 0.32], 'score':0.823, 'class':1}],
             [{'box': [0.87, 0.38, 0.84, 0.22], 'score':0.423, 'class':0}]]

        Args:
            z (ndarray): Output array of neural network. The shape of array

        Return:
            (list): List of predicted bbox, score and class of each image.
                The format of return value is bellow. Box coordinates and size will be returned as
                ratio to the original image size. Therefore the range of 'box' is [0 ~ 1].

            [
                [ # Prediction of first image.
                    {'box': [x, y, w, h], 'score':(float), 'class':(int)},
                    {'box': [x, y, w, h], 'score':(float), 'class':(int)},
                    ...
                ],
                [ # Prediction of second image.
                    {'box': [x, y, w, h], 'score':(float), 'class':(int)},
                    {'box': [x, y, w, h], 'score':(float), 'class':(int)},
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

        z[:, :, 4:-8] = rm.softmax(z[:, :, 4:-8]).as_ndarray()
        z = z.transpose((0, 2, 1))
        mbox_loc = z[:, :, :4]
        variances = z[:, :, -4:]
        mbox_priorbox = z[:, :, -8:-4]
        mbox_conf = z[:, :, 4:-8]
        results = []
        for i in range(len(mbox_loc)):
            results.append([])
            decoded_bbox = self.decode_boxes(mbox_loc[i], mbox_priorbox[i], variances[i])
            for c in range(self.num_class):
                if c == 0:
                    #background
                    continue
                c_confs = mbox_conf[i, :, c]
                c_confs_m = c_confs > score_threshold
                if len(c_confs[c_confs_m]) > 0:
                    boxes_to_process = decoded_bbox[c_confs_m]
                    confs_to_process = c_confs[c_confs_m]
                    idx = self.nms(boxes_to_process, confs_to_process, nms_threshold)
                    good_boxes = boxes_to_process[idx]
                    confs = confs_to_process[idx][:, None]

                    for j in range(len(confs)):
                        results[-1].append({"class": c-1, "score": confs[j],
                                            "box": good_boxes[j],
                                            'name': self.class_map[c-1]})
            if len(results[-1]) > 0:
                scores = np.array([obj['score'] for obj in results[-1]])
                argsort = np.argsort(scores)[::-1]
                results[-1] = np.array(results[-1])[argsort]
                results[-1] = results[-1][:keep_top_k].tolist()
        return results

    def encode_box(self, box, return_iou=True):
        iou = calc_iou(self.prior, box)
        encoded_box = np.zeros((self.num_prior, 4 + return_iou))
        assign_mask = iou > self.overlap_threshold
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]
        assigned_priors = self.prior[assign_mask]
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        assigned_priors_center = 0.5 * (assigned_priors[:, :2] + assigned_priors[:, 2:4])
        assigned_priors_wh = assigned_priors[:, 2:4] - assigned_priors[:, :2]

        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= assigned_priors_wh
        encoded_box[:, :2][assign_mask] /= assigned_priors[:, -4:-2]
        encoded_box[:, 2:4][assign_mask] = np.log(box_wh/assigned_priors_wh)
        encoded_box[:, 2:4][assign_mask] /= assigned_priors[:, -2:]

        return encoded_box.ravel()

    def assign_boxes(self, boxes):
        assignment = np.zeros((self.num_prior, 4+self.num_class+8))
        assignment[:, 4] = 1.0 #background
        if len(boxes) == 0:
            return assignment
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        encoded_boxes = encoded_boxes.reshape(-1, self.num_prior, 5) #xmin ymin xmax ymax iou
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]
        assign_num = len(best_iou_idx)
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:-8][best_iou_mask] = boxes[best_iou_idx, 4:]
        assignment[:, -8][best_iou_mask] = 1
        return assignment.T

    def decode_boxes(self, mbox_loc, mbox_priorbox, variances):
        prior_width = mbox_priorbox[:, 2] - mbox_priorbox[:, 0]
        prior_height = mbox_priorbox[:, 3] - mbox_priorbox[:, 1]
        prior_center_x = 0.5 * (mbox_priorbox[:, 2] + mbox_priorbox[:, 0])
        prior_center_y = 0.5 * (mbox_priorbox[:, 3] + mbox_priorbox[:, 1])


        decoded_bbox_center_x = mbox_loc[:, 0] * prior_width * variances[:, 0]
        decoded_bbox_center_x += prior_center_x
        decoded_bbox_center_y = mbox_loc[:, 1] * prior_height * variances[:, 1]
        decoded_bbox_center_y += prior_center_y
        decoded_bbox_width = np.exp(mbox_loc[:, 2] * variances[:, 2])
        decoded_bbox_width *= prior_width
        decoded_bbox_height = np.exp(mbox_loc[:, 3] * variances[:, 3])
        decoded_bbox_height *= prior_height

        decoded_bbox = np.concatenate([
            decoded_bbox_center_x[:, None],
            decoded_bbox_center_y[:, None],
            decoded_bbox_width[:, None],
            decoded_bbox_height[:, None]
        ], axis=-1)
        return decoded_bbox

    def nms(self, boxes, score=None, nms_threshold=0.3, limit=None):
        if len(boxes) == 0:
            return []
        selec = []
        x1 = boxes[:,0]
        y1 = boxes[:,1]
        x2 = boxes[:,2]
        y2 = boxes[:,3]

        area = (x2-x1+1) * (y2-y1+1)
        idxs = np.argsort(score)

        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            selec.append(i)
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            overlap = (w * h) / area[idxs[:last]]
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > nms_threshold)[0])))
        return np.array(selec).astype(np.int32)

    def predict(self, img_list):
        """
        This method accepts either ndarray and list of image path.

        Example:
            >>>
            >>> model.predict(['img01.jpg'], [img02.jpg]])
            [[{'box': [0.21, 0.44, 0.11, 0.32], 'score':0.823, 'class':1}],
             [{'box': [0.87, 0.38, 0.84, 0.22], 'score':0.423, 'class':0}]]

        Args:
            img_list (string, list, ndarray):

        Return:
            (list): List of predicted bbox, score and class of each image.
                The format of return value is bellow. Box coordinates and size will be returned as
                ratio to the original image size. Therefore the range of 'box' is [0 ~ 1].

            [
                [ # Prediction of first image.
                    {'box': [x, y, w, h], 'score':(float), 'class':(int)},
                    {'box': [x, y, w, h], 'score':(float), 'class':(int)},
                    ...
                ],
                [ # Prediction of second image.
                    {'box': [x, y, w, h], 'score':(float), 'class':(int)},
                    {'box': [x, y, w, h], 'score':(float), 'class':(int)},
                    ...
                ],
                ...
            ]

        Note:
            Box coordinate and size will be returned as ratio to the original image size.
            Therefore the range of 'box' is [0 ~ 1].

        """
        batch_size = 32
        self.set_models(inference=True)
        if isinstance(img_list, (list, str)):
            if isinstance(img_list, (tuple, list)):
                if len(img_list) >= 32:
                    test_dist = ImageDistributor(img_list)
                    results = []
                    bar = tqdm()
                    bar.total = int(np.ceil(len(test_dist) / batch_size))
                    for i, (x_img_list, _) in enumerate(test_dist.batch(batch_size, shuffle=False)):
                        img_array = np.vstack([load_img(path, self.imsize)[None] for path in x_img_list])
                        img_array = self.preprocess(img_array)
                        results.extend(self.get_bbox(self(img_array).as_ndarray()))
                        bar.update(1)
                    return results
                img_array = np.vstack([load_img(path, self.imsize)[None] for path in img_list])
                img_array = self.preprocess(img_array)
            else:
                img_array = load_img(img_list, self.imsize)[None]
                img_array = self.preprocess(img_array)
                return self.bbox_util.get_bbox(self(img_array).as_ndarray())[0]
        else:
            img_array = img_list
        return self.get_bbox(self(img_array).as_ndarray())

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

            return self.preprocess(img_data), np.array(targets)
        return builder

    def loss(self, x, y, neg_pos_ratio=3.0, negatives_for_hard=100.0):
        batch_size = y.shape[0]
        num_boxes = y.shape[2]
        conf_loss = rm.sum(rm.softmax_cross_entropy(x[:, 4:-8, :], y[:, 4:-8, :], reduce_sum=False), axis=1)
        loc_loss = rm.sum(rm.smoothed_l1(x[:, :4, :], y[:, :4, :], reduce_sum=False), axis=1)

        num_pos = np.sum(y[:, -8, :], axis=1)
        pos_loc_loss = rm.sum(loc_loss*(y[:, -8, :]), axis=1)
        pos_conf_loss = rm.sum(conf_loss * y[:, -8, :], axis=1)

        num_neg = np.minimum(neg_pos_ratio * num_pos, num_boxes - num_pos)
        has_min = num_neg > 0
        has_min = np.any(has_min).astype('float')

        num_neg = np.concatenate([num_neg, [(1-has_min)*negatives_for_hard]])
        num_neg_batch = np.min(num_neg[(num_neg > 0)])

        num_neg_batch = int(num_neg_batch)
        confs_start = 5 # 4+0(background label) + 1
        confs_end = confs_start + self.num_class - 1

        max_confs = np.max(x[:, confs_start:confs_end, :].as_ndarray(), axis=1)
        indices = (max_confs * (1-y[:, -8, :])).argsort()[:, ::-1][:, :num_neg_batch]

        batch_idx = np.expand_dims(range(0, batch_size), 1)
        batch_idx = np.tile(batch_idx, (1, num_neg_batch))
        full_indices = (batch_idx.reshape(-1)*int(num_boxes) + indices.reshape(-1))

        neg_conf_loss = conf_loss.reshape(-1)[full_indices]
        neg_conf_loss = neg_conf_loss.reshape((batch_size, num_neg_batch))
        neg_conf_loss = rm.sum(neg_conf_loss, axis=1)

        total_loss = neg_conf_loss + pos_conf_loss
        total_loss /= (num_pos + float(num_neg_batch))

        num_pos = np.where(np.not_equal(num_pos, 0), num_pos,np.ones_like(num_pos))
        total_loss = total_loss +  (pos_loc_loss/num_pos)
        loss = rm.sum(total_loss)
        return loss

    def fit(self, train_img_path_list=None, train_annotation_list=None,
            valid_img_path_list=None, valid_annotation_list=None,
            epoch=200, batch_size=64, augmentation=None, callback_end_epoch=None):

        train_dist = ImageDistributor(
            train_img_path_list, train_annotation_list, augmentation=augmentation)
        valid_dist = ImageDistributor(valid_img_path_list, valid_annotation_list)

        batch_loop = int(np.ceil(len(train_dist) / batch_size))
        avg_train_loss_list = []
        avg_valid_loss_list = []
        for e in range(epoch):
            bar = tqdm(range(batch_loop))
            display_loss = 0
            for i, (train_x, train_y) in enumerate(train_dist.batch(batch_size, target_builder=self.build_data())):
                self.set_models(inference=False)
                with self.train():
                    loss = self.loss(self(train_x), train_y)
                    reg_loss = loss + self.regularize()
                reg_loss.grad().update(self.get_optimizer(e, epoch, i, batch_loop))
                try:
                    loss = loss.as_ndarray()[0]
                except:
                    loss = loss.as_ndarray()
                display_loss += loss
                bar.set_description("Epoch:{:03d} Train Loss:{:5.3f}".format(e, float(loss)))
                bar.update(1)
            avg_train_loss = display_loss / (i + 1)
            avg_train_loss_list.append(avg_train_loss)

            if valid_dist is not None:
                bar.n = 0
                bar.total = int(np.ceil(len(valid_dist) / batch_size))
                display_loss = 0
                for i, (valid_x, valid_y) in enumerate(valid_dist.batch(batch_size, target_builder=self.build_data())):
                    self.set_models(inference=True)
                    loss = self.loss(self(valid_x), valid_y)
                    try:
                        loss = loss.as_ndarray()[0]
                    except:
                        loss = loss.as_ndarray()
                    display_loss += loss
                    bar.set_description("Epoch:{:03d} Valid Loss:{:5.3f}".format(e, float(loss)))
                    bar.update(1)


                avg_valid_loss = display_loss / (i + 1)
                avg_valid_loss_list.append(avg_train_loss)
                bar.set_description("Epoch:{:03d} Avg Train Loss:{:5.3f} Avg Valid Loss:{:5.3f}".format(
                    e, avg_train_loss, avg_valid_loss))
            else:
                bar.set_description("Epoch:{:03d} Avg Train Loss:{:5.3f}".format(e, avg_train_loss))
            bar.close()
            if callback_end_epoch is not None:
                callback_end_epoch(e, self, avg_train_loss_list, avg_valid_loss_list)

        return avg_train_loss_list, avg_valid_loss_list
