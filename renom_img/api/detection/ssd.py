import time
from itertools import product as product

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from renom_img import __version__
import renom as rm
from renom_img.api import Base, adddoc
from renom_img.api.detection import Detection
from renom_img.api.cnn.ssd import CnnSSD
from renom_img.api.utility.load import prepare_detection_data, resize_detection_data
from renom_img.api.utility.box import transform2xy12, calc_iou_xywh
from renom_img.api.utility.load import parse_xml_detection, load_img
from renom_img.api.utility.nms import nms
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.optimizer import OptimizerSSD
from renom_img.api.utility.exceptions.check_exceptions import *
from renom_img.api.utility.exceptions.exceptions import WeightLoadError

def calc_iou(prior, box):
    """
    Ensure both arguments are point formed boxes.
    """
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

        output = np.array(mean_boxes)
        if self.clip:
            output = np.clip(output, 0, 1)

        # Change output to enter offset format.
        min_xy = output[:, :2] - output[:, 2:] / 2.
        max_xy = output[:, :2] + output[:, 2:] / 2.
        output = np.concatenate([min_xy, max_xy], axis=1)
        return output

class TargetBuilderSSD():
    """
    This function returns a function which creates input data and target data specified for SSD.

    Returns:
        (function): Returns function which creates input data and target data.

    Example:
        >>> builder = model.build_data() 
        >>> x, y = builder(image_path_list, annotation_list)
        >>> z = model(x)
        >>> loss = model.loss(z,y)

    """

    def __init__(self, class_map,imsize,prior,prior_box,num_prior, threshold):
        self.class_map = class_map
        self.num_class = len(self.class_map) + 1
        self.imsize = imsize
        self.prior = prior
        self.prior_box = prior_box
        self.num_prior = num_prior
        self.overlap_threshold = threshold 

    def __call__(self, *args, **kwargs):
        return self.build(*args, **kwargs)

    def preprocess(self, x, reverse=False):
        """Image preprocess for SSD.

        Args:
            x (ndarray): Input Images.
            reverse (bool): Decide whether to add or subtract the ImageNet mean from images.
                If true is given, ImageNet mean is added to the input images, otherwise subtracted. Default is False. 

        Returns:
            (ndarray): Preprocessed data.
        """
        if reverse:
            x[:, 0, :, :] += 123.68  # R
            x[:, 1, :, :] += 116.779  # G
            x[:, 2, :, :] += 103.939  # B
        else:
            x[:, 0, :, :] -= 123.68  # R
            x[:, 1, :, :] -= 116.779  # G
            x[:, 2, :, :] -= 103.939  # B
        return x

    def assign_boxes(self, boxes):
        # background is already included in self.num_class.
        assignment = np.zeros((self.num_prior, 5 + self.num_class))
        # assignment[:, 4] = 1.0  # background(This means id=0 is background)
        if len(boxes) == 0:
            return assignment
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])

        # (box_num, prior_num, (xmin ymin xmax ymax iou))
        encoded_boxes = encoded_boxes.reshape(-1, self.num_prior, 5)

        best_iou = encoded_boxes[:, :, -1].max(axis=0)  # get the best fit target for each prior.
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)

        # Cut background
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]

        # Assign conf score
        assignment[:, 0][best_iou_mask] = encoded_boxes[best_iou_idx, best_iou_mask, 4]
        # Assign loc
        assignment[:, 1:5][best_iou_mask] = encoded_boxes[best_iou_idx, best_iou_mask, :4]
        # Assign class
        assignment[:, 5][~best_iou_mask] = 1  # Background.
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

        # Encode xy
        encoded_box[:, :2][assign_mask] = box_center - assigned_priors_center
        encoded_box[:, :2][assign_mask] /= (assigned_priors_wh * self.prior.variance[0])

        # Encode wh
        encoded_box[:, 2:4][assign_mask] = np.log(
            box_wh / assigned_priors_wh + 1e-8) / self.prior.variance[1]
        return encoded_box.flatten()

    def build(self, img_path_list, annotation_list=None, augmentation=None, **kwargs):
        """
        Args:
            x: Image path list.
            y: Detection formatted label.
        """
        check_missing_param(self.class_map)
        if annotation_list is None:
            img_array = np.vstack([load_img(path,self.imsize)[None]
                                    for path in img_path_list])
            img_array = self.preprocess(img_array)
            return img_array

        N = len(img_path_list)
        img_data, label_data = prepare_detection_data(img_path_list,
                                                      annotation_list)
        if augmentation is not None:
            img_data, label_data = augmentation(img_data, label_data, mode="detection")
        img_data, label_data = resize_detection_data(img_data, label_data, self.imsize)
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
 

class SSD(Detection):
    """ SSD object detection algorithm.

    Args:
        class_map (list, dict): List of class names.
        imsize (int or tuple): Image size. Must be 300x300.
            This can either be specified as an integer ex): 300 or tuple ex): (300,300).
        overlap_threshold (float): Threshold to be used in selecting the best prior box.
            This threshold sould be between 0 and 1. The default is 0.5.
        load_pretrained_weight (bool, string): Whether or not to load pretrained weights for the backbone model.
            If True, pretrained weights will be downloaded to the current directory and loaded into the model.
            If a string is provided, pretrained weightwill be loaded from the specified filename. The default is False.
        train_whole_network (bool): Whether or not to train the whole network.
            If True, all network layers will be trained. If False, the backbone network layers
            will be set to inference mode, and no updates will be performed for the backbone network weights. The default is False.

    Example:
        >>> from renom_img.api.detection.ssd import SSD
        >>> from renom_img.api.utility.load import parse_xml_detection
        >>>
        >>> train_label_path_list = ...  # provide list of paths to training data
        >>> annotation_list, class_map = parse_xml_detection(train_label_path_list)
        >>>
        >>> model = SSD(class_map, imsize=(300,300), load_pretrained_weight=True, train_whole_network=True)

    References:
        | Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu, Alexander C. Berg 
        | **SSD: Single Shot MultiBox Detector**
        | https://arxiv.org/abs/1512.02325
        |

    """
    WEIGHT_URL = CnnSSD.WEIGHT_URL
    SERIALIZED = ("overlap_threshold", *Base.SERIALIZED)

    def __init__(self, class_map=None, imsize=(300, 300),
                 overlap_threshold=0.5, load_pretrained_weight=False, train_whole_network=False):
        # check for exceptions
        check_ssd_init(overlap_threshold,imsize)

        self._model = CnnSSD()
        super(SSD, self).__init__(class_map, imsize,
                                  load_pretrained_weight, train_whole_network, self._model)

        self.num_class = len(self.class_map) + 1
        self._model.set_output_size(self.num_class)
        self._model.set_train_whole(train_whole_network)

        self.overlap_threshold = overlap_threshold
        self.prior = PriorBox()
        self.prior_box = self.prior.create()
        self.num_prior = len(self.prior_box)
        self.default_optimizer = OptimizerSSD()
        self.decay_rate = 0.00004


    def build_data(self):

       return TargetBuilderSSD(self.class_map, self.imsize, self.prior,self.prior_box, self.num_prior, self.overlap_threshold)

    def load(self, filename):
        """Load saved weights to model.

        Args:
            filename (str): File name of saved model.

        Example:
            >>> model = rm.Dense(2)
            >>> model.load("model.hd5")
        """
        import h5py
        f = h5py.File(filename, 'r+')
        values = f['values']
        types = f['types']

        names = sorted(values.keys())

        try:
            self._try_load(names,values,types)
        except AttributeError as e:
            try:
                names,values,types = self._mapping(names,values,types)
                self._try_load(names,values,types)
            except Exception as e:
                raise WeightLoadError('The {} weight file can not be loaded into the {} model.'.format(filename, self.__class__.__name__))

    def _mapping(self,names,values,types):
        for name in names:
            if "._network" in name:
                values[name.replace("._network","._model")] = values.pop(name)
                types[name.replace("._network","._model")] = types.pop(name)
            elif "._freezed_network" in name:
                values[name.replace("._freezed_network","._model._freezed_network")] = values.pop(name)
                types[name.replace("._freezed_network","._model._freezed_network")] = types.pop(name)

        names = [n.replace("._network","._model") for n in names]
        names = [n.replace("._freezed_network","._model._freezed_network") for n in names]

        return sorted(names),values,types

    def _try_load(self,names,values,types):

        def get_attr(root, names):
            names = names.split('.')[1:]
            ret = root
            for name in names:
                ret = getattr(ret, name)
            return ret

        target = self
        for name in names:
            target = get_attr(self, name)

            values_grp = values[name]
            types_grp = types[name]

            for k, v in values_grp.items():
                v = v.value
                if isinstance(v, np.ndarray):
                    type = types_grp.get(k, None)
                    if type:
                        if type.value == 'renom.Variable':
                            auto_update = types_grp[k + '._auto_update'].value
                            v = rm.Variable(v, auto_update=auto_update)
                        else:
                            v = rm.Node(v)

                if k.startswith('__dict__.'):
                    obj = target
                    name = k.split(".", 1)[1]
                else:
                    obj = target.params
                    name = k

                setattr(obj, name, v)


    def decode_box(self, loc):
        prior = self.prior_box
        prior_wh = prior[:, 2:] - prior[:, :2]
        prior_xy = prior[:, :2] + prior_wh / 2.
        boxes = np.concatenate([
            prior_xy + loc[:, :2] * self.prior.variance[0] * prior_wh,
            prior_wh * np.exp(loc[:, 2:] * self.prior.variance[1]),
        ], axis=1)
        boxes[:, :2] -= boxes[:, 2:] / 2.
        boxes[:, 2:] += boxes[:, :2]
        return boxes


    def loss(self, x, y, neg_pos_ratio=3.0):
        """
        Loss function specified for SSD.
        
        Args:
            x(Node, nd_array): Output data of neural network.
            y(Node, nd_array): Target data.
            neg_pos_ratio(float): Positive and Negative ratio to be used for hard negative mining.
                After the matching with true boxes, most of the prior boxes are negative. To eliminate imbalance 
                between positive and negative boxes this ratio is used by the loss function. The default value is 3.0.

        Returns:
          Loss between x and y

        Return type:
          Node

        Example:
            >>> z = model(x)
            >>> loss = model.loss(x,y)
        """
        pos_samples = (y[:, :, 5] == 0)[..., None]
        N = np.sum(pos_samples)
        pos_Ns = np.sum(pos_samples, axis=1)
        neg_Ns = np.clip(neg_pos_ratio * pos_Ns, 0, y.shape[1])

        # Loc loss
        loc_loss = rm.sum(rm.smoothed_l1(x[..., :4], y[..., 1:5], reduce_sum=False) * pos_samples)

        # this is for hard negative mining.
        np_x = x[..., 4:].as_ndarray()
        max_np_x = np.max(np_x)
        loss_c = np.log(np.sum(np.exp(np_x.reshape(-1, self.num_class) - max_np_x),
                               axis=1, keepdims=True) + 1e-8) + max_np_x
        loss_c -= np_x[..., 0].reshape(-1, 1)
        loss_c = loss_c.reshape(len(x), -1)
        loss_c[pos_samples.astype(np.bool)[..., 0]] = np.Inf  # Cut positive samples.

        sorted_index = np.argsort(-1 * loss_c, axis=1)  # Arg sort by dicending order.
        index_rank = np.argsort(sorted_index, axis=1)
        neg_samples = index_rank < neg_Ns
        samples = (neg_samples[..., None] + pos_samples).astype(np.bool)
        conf_loss = rm.sum(rm.softmax_cross_entropy(x[..., 4:].transpose(0, 2, 1),
                                                    y[..., 5:].transpose(0, 2, 1), reduce_sum=False).transpose(0, 2, 1) * samples)

        loss = conf_loss + loc_loss
        return loss / (N / len(x))

    def predict(self, img_list, batch_size=1, score_threshold=0.6, nms_threshold=0.45):
        """
        This method accepts an ndarray of image paths, a list of image paths, or an image path as a string.

        Args:
            img_list (string, list, ndarray): Path to an image, list of path or ndarray.
            score_threshold (float): The threshold for the confidence score.
                                     Predicted boxes which have a lower confidence score than the threshold are discarded.
                                     The default is 0.6.
            nms_threshold (float): The threshold for non-maximum supression. The default is 0.45.

        Return:
            (list): List of predicted bbox, score and class for each image.
            The format of the return value is shown below. Box coordinates and size will be returned as
            ratios to the original image size. Therefore, the values of 'box' are in the range [0 ~ 1].

        .. code-block :: python

            # An example of a return value.
            [
                [ # Prediction for first image.
                    {'box': [x, y, w, h], 'score':(float), 'class':(int), 'name':(str)},
                    {'box': [x, y, w, h], 'score':(float), 'class':(int), 'name':(str)},
                    ...
                ],
                [ # Prediction for second image.
                    {'box': [x, y, w, h], 'score':(float), 'class':(int), 'name':(str)},
                    {'box': [x, y, w, h], 'score':(float), 'class':(int), 'name':(str)},
                    ...
                ],
                ...
            ]

        Example:
            >>>
            >>> model.predict(['img01.jpg', 'img02.jpg']])
            [[{'box': [0.21, 0.44, 0.11, 0.32], 'score':0.823, 'class':1, 'name':'dog'}],
             [{'box': [0.87, 0.38, 0.84, 0.22], 'score':0.423, 'class':0, 'name':'cat'}]]

        Note:
            Box coordinates and size will be returned as ratios to the original image size.
            Therefore, the values of 'box' are in the range [0 ~ 1].

        """
        return super(SSD, self).predict(img_list, batch_size, score_threshold, nms_threshold)


    def get_bbox(self, z, score_threshold=0.6, nms_threshold=0.45):
        """
        Calculates the bounding box location, size and class information for model predictions.

        Example:
            >>> z = model(x)
            >>> model.get_bbox(z)
            [[{'box': [0.21, 0.44, 0.11, 0.32], 'score':0.823, 'class':1, 'name':'dog'}],
             [{'box': [0.87, 0.38, 0.84, 0.22], 'score':0.423, 'class':0, 'name':'cat'}]]

        Args:
            z (ndarray): Output array of neural network.

        Return:
            (list) : List of predicted bbox, score and class for each image.
            The format of the return value is shown below. Box coordinates and size will be returned as
            ratios to the original image size. Therefore, the values of 'box' are in the range [0 ~ 1].

        .. code-block :: python

            # An example of a return value.
            [
                [ # Prediction for first image.
                    {'box': [x, y, w, h], 'score':(float), 'class':(int), 'name':(str)},
                    {'box': [x, y, w, h], 'score':(float), 'class':(int), 'name':(str)},
                    ...
                ],
                [ # Prediction for second image.
                    {'box': [x, y, w, h], 'score':(float), 'class':(int), 'name':(str)},
                    {'box': [x, y, w, h], 'score':(float), 'class':(int), 'name':(str)},
                    ...
                ],
                ...
            ]

        Note:
            Box coordinates and size will be returned as ratios to the original image size.
            Therefore, the values of 'box' are in the range [0 ~ 1].

        """
        N = len(z)
        class_num = len(self.class_map)
        top_k = 100
        if hasattr(z, 'as_ndarray'):
            z = z.as_ndarray()

        loc, conf = np.split(z, [4], axis=2)
        loc = np.concatenate([self.decode_box(loc[n])[None] for n in range(N)], axis=0)
        loc = np.clip(loc, 0, 1)
        loc[:, :, 2:] = loc[:, :, 2:] - loc[:, :, :2]
        loc[:, :, :2] += loc[:, :, 2:] / 2.

        conf = rm.softmax(conf.transpose(0, 2, 1)).as_ndarray().transpose(0, 2, 1)

        result_bbox = []
        conf = conf[:, :, 1:]
        conf[conf < score_threshold] = 0

        # Transpose are required for manipulate tensors as `class major` order.
        # (N, box, class) => (N, class, box)
        sorted_conf_index = np.argsort(-conf, axis=1)  # Arg sort by dicending order.
        keep_index = (np.argsort(sorted_conf_index, axis=1) < top_k).transpose(0, 2, 1)

        conf = conf.transpose(0, 2, 1)
        conf = conf[keep_index].reshape(N, class_num, -1)

        loc = np.concatenate([
            loc[(keep_index[:, c, :].reshape(N, -1, 1) * np.ones_like(loc)).astype(np.bool)]
            .reshape(N, 1, -1, 4)
            for c in range(class_num)], axis=1)

        for n in range(N):
            nth_result = []
            nth_loc = loc[n]
            for ndind in np.ndindex(*conf.shape[1:]):
                if conf[n, ndind[0], ndind[1]] < score_threshold:
                    continue
                nth_result.append({
                    "box": nth_loc[ndind[0], ndind[1]].tolist(),
                    "name": self.class_map[ndind[0]].decode('ascii'),
                    "class": int(ndind[0]),
                    "score": float(conf[n, ndind[0], ndind[1]])
                })
            result_bbox.append(nth_result)
        ret = nms(result_bbox, nms_threshold)
        return ret


if __name__ == "__main__":
    import os
    import random
    from renom.cuda import set_cuda_active
    from renom_img.api.utility.load import parse_xml_detection, load_img
    from renom_img.api.utility.misc.display import draw_box
    from renom_img.api.utility.distributor.distributor import ImageDistributor

    set_cuda_active(True)
    img_path = "../../../example/VOCdevkit/VOC2007/JPEGImages/"
    label_path = "../../../example/VOCdevkit/VOC2007/Annotations/"

    img_list = [os.path.join(img_path, p) for p in sorted(os.listdir(img_path))[:8 * 100]]
    lbl_list = [os.path.join(label_path, p) for p in sorted(os.listdir(label_path))[:8 * 100]]
    valid_img_list = [os.path.join(img_path, p)
                      for p in sorted(os.listdir(img_path))[8 * 300:10 * 300]]
    valid_lbl_list = [os.path.join(label_path, p)
                      for p in sorted(os.listdir(label_path))[8 * 300:10 * 300]]

    annotation_list, class_map = parse_xml_detection(lbl_list)
    valid_annotation_list, _ = parse_xml_detection(valid_lbl_list)

    def callback(epoch, model, train_loss, valid_loss):
        plt.clf()
        path = random.choice(valid_img_list)
        plt.imshow(draw_box(path, model.predict(path)))
        plt.tight_layout()
        plt.savefig("img%04d.png" % (epoch))

    ssd = SSD(class_map, load_pretrained_weight=True, train_whole_network=False)
    ssd.fit(img_list, annotation_list, valid_img_list, valid_annotation_list,
            batch_size=16, callback_end_epoch=callback)
