import os
import numpy as np
import renom as rm
from tqdm import tqdm
from PIL import Image

from renom_img.api.classification.darknet import Darknet
from renom_img.api.utility.distributor.distributor import ImageDistributor
from renom_img.api.utility.misc.download import download
from renom_img.api.utility.box import transform2xy12
from renom_img.api.utility.load import prepare_detection_data, load_img


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
    """ Yolo object detection algorithm.

    Args:
        num_class (int): Number of class. 
        cells (int or tuple): Cell size. 
        boxes (int): Number of boxes.
        imsize (int, tuple): Image size.
        load_pretrained_weight (bool, str): If true, pretrained weight will be 
          downloaded to current directory. If string is given, pretrained weight 
          will be saved as given name.

    References:
        Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi  
        You Only Look Once: Unified, Real-Time Object Detection  
        https://arxiv.org/abs/1506.02640  

    """

    SERIALIZED = ("_cells", "_bbox", "imsize", "class_map", "num_class", "_last_dense_size")
    WEIGHT_URL = "http://docs.renom.jp/downloads/weights/Yolov1.h5"

    def __init__(self, class_map=[], cells=7, bbox=2, imsize=(224, 224), load_pretrained_weight=False, train_whole_network=False):
        num_class = len(class_map)

        if not hasattr(cells, "__getitem__"):
            cells = (cells, cells)
        if not hasattr(imsize, "__getitem__"):
            imsize = (imsize, imsize)

        self.num_class = num_class
        self.class_map = class_map
        self.class_map = [c.encode("ascii", "ignore") for c in self.class_map]
        self._cells = cells
        self._bbox = bbox
        self._last_dense_size = (num_class + 5 * bbox) * cells[0] * cells[1]
        model = Darknet(self._last_dense_size)
        self._train_whole_network = train_whole_network

        self.imsize = imsize
        self._freezed_network = rm.Sequential(model[:-7])
        self._network = rm.Sequential(model[-7:])

        self._opt = rm.Sgd(0.01, 0.9)

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
        """Returns an instance of Optimizer for training Yolov1 algorithm.

        If all argument(current_epoch, total_epoch, current_batch, total_batch) are given,
        an optimizer object which whose learning rate is modified according to the 
        number of training iteration. Otherwise, constant learning rate is set.

        Args:
            current_epoch (int): The number of current epoch.
            total_epoch (int): The number of total epoch.
            current_batch (int): The number of current batch.
            total_epoch (int): The number of total batch.

        Returns:
            (Optimizer): Optimizer object.
        """
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
        """Image preprocess for Yolov1.

        :math:`x_{new} = x*2/255 - 1`

        Args:
            x (ndarray):

        Returns:
            (ndarray): Preprocessed data.
        """
        return x / 255. * 2 - 1

    def forward(self, x):
        """Performs forward propagation.
        This function can be called using ``__call__`` method.
        See following example of method usage.

        Args:
            x (ndarray, Node): Input image as an tensor.

        Returns:
            (Node): Returns raw output of yolo v1.
            You can reform it to bounding box form using the method ``get_bbox``.

        Example:
            >>> import numpy as np
            >>> from renom_img.api.detection.yolo_v1 import Yolov1
            >>>
            >>> x = np.random.rand(1, 3, 224, 224)
            >>> class_map = ["dog", "cat"]
            >>> model = Yolov1(class_map)
            >>> y = model.forward(x) # Forward propagation.
            >>> y = model(x)  # Same as above result.
            >>> 
            >>> bbox = model.get_bbox(y) # The output can be reformed using get_bbox method.

        """
        assert len(self.class_map) > 0, \
            "Class map is empty. Please set the attribute class_map when instantiate model class. " +\
            "Or, please load already trained model using the method 'load()'."
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

    def get_bbox(self, z):
        """
        Example:
            >>> z = model(x)
            >>> model.get_bbox(z)
            [[{'box': [0.21, 0.44, 0.11, 0.32], 'score':0.823, 'class':1, 'name':'dog'}],
             [{'box': [0.87, 0.38, 0.84, 0.22], 'score':0.423, 'class':0, 'name':'cat'}]]

        Args:
            z (ndarray): Output array of neural network. The shape of array 

        Return:
            (list) : List of predicted bbox, score and class of each image.
            The format of return value is bellow. Box coordinates and size will be returned as
            ratio to the original image size. Therefore the range of 'box' is [0 ~ 1].

        .. code-block :: python

            # An example of return value.
            [
                [ # Prediction of first image.
                    {'box': [x, y, w, h], 'score':(float), 'class':(int), 'name':(str)},
                    {'box': [x, y, w, h], 'score':(float), 'class':(int), 'name':(str)},
                    ...
                ],
                [ # Prediction of second image.
                    {'box': [x, y, w, h], 'score':(float), 'class':(int), 'name':(str)},
                    {'box': [x, y, w, h], 'score':(float), 'class':(int), 'name':(str)},
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

        N = len(z)
        cell = self._cells[0]
        bbox = self._bbox
        probs = np.zeros((N, cell, cell, bbox, self.num_class))
        boxes = np.zeros((N, cell, cell, bbox, 4))
        yolo_format_out = z.reshape(
            N, cell, cell, bbox * 5 + self.num_class)
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

        # Clip bounding box.
        w = boxes[:, :, :, :, 2] / 2.
        h = boxes[:, :, :, :, 3] / 2.
        x1 = np.clip(boxes[:, :, :, :, 0] - w, 0, 1)
        y1 = np.clip(boxes[:, :, :, :, 1] - h, 0, 1)
        x2 = np.clip(boxes[:, :, :, :, 0] + w, 0, 1)
        y2 = np.clip(boxes[:, :, :, :, 1] + h, 0, 1)
        boxes[:, :, :, :, 2] = x2 - x1
        boxes[:, :, :, :, 3] = y2 - y1
        boxes[:, :, :, :, 0] = x1 + boxes[:, :, :, :, 2] / 2.
        boxes[:, :, :, :, 1] = y1 + boxes[:, :, :, :, 3] / 2.

        probs = probs.reshape(N, -1, self.num_class)
        boxes = boxes.reshape(N, -1, 4)

        probs[probs < 0.3] = 0
        # Perform NMS

        argsort = np.argsort(probs, axis=1)[:, ::-1]
        for n in range(N):
            for cl in range(self.num_class):
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
                "name": self.class_map[int(max_class[indexes[0][i], indexes[1][i]])].decode("utf-8"),
                "box": boxes[indexes[0][i], indexes[1][i]].astype(np.float64).tolist(),
                "score": float(max_probs[indexes[0][i], indexes[1][i]])
            })
        return result

    def predict(self, img_list):
        """
        This method accepts either ndarray and list of image path.

        Example:
            >>> 
            >>> model.predict(['img01.jpg', 'img02.jpg']])
            [[{'box': [0.21, 0.44, 0.11, 0.32], 'score':0.823, 'class':1, 'name':'dog'}],
             [{'box': [0.87, 0.38, 0.84, 0.22], 'score':0.423, 'class':0, 'name':'cat'}]]

        Args:
            img_list (string, list, ndarray): Path to an image, list of path or ndarray.

        Return:
            (list): List of predicted bbox, score and class of each image.
            The format of return value is bellow. Box coordinates and size will be returned as
            ratio to the original image size. Therefore the range of 'box' is [0 ~ 1].

        .. code-block :: python

            # An example of return value.
            [
                [ # Prediction of first image.
                    {'box': [x, y, w, h], 'score':(float), 'class':(int), 'name':(str)},
                    {'box': [x, y, w, h], 'score':(float), 'class':(int), 'name':(str)},
                    ...
                ],
                [ # Prediction of second image.
                    {'box': [x, y, w, h], 'score':(float), 'class':(int), 'name':(str)},
                    {'box': [x, y, w, h], 'score':(float), 'class':(int), 'name':(str)},
                    ...
                ],
                ...
            ]

        Note:
            Box coordinate and size will be returned as ratio to the original image size.
            Therefore the range of 'box' is [0 ~ 1].

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

    def build_data(self):
        def builder(img_path_list, annotation_list, augmentation=None, **kwargs):
            """
            Args:
                x: Image path list.
                y: Detection formatted label.
            """
            N = len(img_path_list)
            num_bbox = self._bbox
            cell_w, cell_h = self._cells
            target = np.zeros((N, self._cells[0], self._cells[1], 5 * num_bbox + self.num_class))

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
                    one_hot = [0] * obj["class"] + [1] + [0] * (self.num_class - obj["class"] - 1)
                    target[n, int(ty), int(tx)] = \
                        np.concatenate(([1, tx % 1, ty % 1, tw, th] * num_bbox, one_hot))
            return self.preprocess(img_data), target.reshape(N, -1)
        return builder

    def loss(self, x, y):
        N = len(x)
        nd_x = x.as_ndarray()
        num_bbox = self._bbox
        target = y.reshape(N, self._cells[0], self._cells[1], 5 * num_bbox + self.num_class)
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

        diff = (x - y)
        return rm.sum(diff * diff * mask.reshape(N, -1)) / N / 2.

    def fit(self, train_img_path_list, train_annotation_list,
            valid_img_path_list=None, valid_annotation_list=None,
            epoch=136, batch_size=64, augmentation=None, callback_end_epoch=None):

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
                bar.set_description("Epoch:{:03d} Train Loss:{:5.3f}".format(e, loss))
                bar.update(1)
            avg_train_loss = display_loss / (i + 1)
            avg_train_loss_list.append(avg_train_loss)

            if valid_dist is not None:
                bar.n = 0
                bar.total = int(np.ceil(len(valid_dist) / batch_size))
                display_loss = 0
                for i, (valid_x, valid_y) in enumerate(valid_dist.batch(batch_size, target_builder=self.build_data())):
                    self.set_models(inference=True)
                    loss = self.loss(self(train_x), train_y)
                    try:
                        loss = loss.as_ndarray()[0]
                    except:
                        loss = loss.as_ndarray()
                    display_loss += loss
                    bar.set_description("Epoch:{:03d} Valid Loss:{:5.3f}".format(e, loss))
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
