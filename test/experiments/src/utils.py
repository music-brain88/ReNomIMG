import os
import numpy as np
import time
from PIL import Image
from itertools import chain
from xml.etree import ElementTree
from multiprocessing import Pool, Pipe, Process
from renom_img.api.utility.augmentation.augmentation import Augmentation
from renom_img.api.utility.augmentation.process import *


class Yolov2Distributor(object):

    def __init__(self, img_path_list, annotation_list, num_worker=4):
        self._num_worker = num_worker
        self._img_path_list = img_path_list
        self._annotation_list = annotation_list
        self._max_obj_count = np.max([len(a) for a in annotation_list])
        itr = chain.from_iterable([[a['name'] for a in an] for an in annotation_list])
        self._max_class_count = np.max(list(itr)) + 1
        self._autgumentation = Augmentation([
                        Shift(20, 20),
                        Flip(),
                        Rotate(),
                        WhiteNoise()
                    ])

    def __len__(self):
        return len(self._annotation_list)

    def _load_img(self, img_path, imsize):
        img = Image.open(img_path)
        w, h = img.size
        img = img.resize(imsize, Image.BILINEAR)
        img = np.asarray(img).transpose(2, 0, 1).astype(np.float32)
        return img, imsize[0]/float(w), imsize[1]/float(h)

    def _onehot(self, cls_id):
        return [0]*cls_id + [1] + [0]*(self._max_class_count - cls_id - 1)

    def detection_batch(self, batch_size, img_size=(224, 224)):
        feature_size = (img_size[0]//32, img_size[1]//32)

        def build(file_list, annotation_list, send_conn):
            ratio_w = img_size[0]/float(feature_size[0])
            ratio_h = img_size[1]/float(feature_size[1])
            img_list = []
            num_class = self._max_class_count
            channel = num_class + 5
            offset = channel
            label = np.zeros((batch_size, channel, feature_size[0], feature_size[1]))

            for n, (img_path, annotation) in enumerate(zip(file_list, annotation_list)):
                # This returns resized image.
                img, scw, sch = self._load_img(img_path, img_size)

                # Target processing
                boxces = np.array([a['bndbox'] for a in annotation])
                classes = np.array([self._onehot(a['name']) for a in annotation])

                # Change raw scale to resized scale.
                boxces[:, 0] *= scw
                boxces[:, 1] *= sch
                boxces[:, 2] *= scw
                boxces[:, 3] *= sch

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

                img_list.append(img)
            img_list = np.array(img_list)
            # img_list, lb = self._autgumentation(np.array(img_list),
            #                 label[:, 1:6].reshape(batch_size, -1), mode="detection")
            # label[:, 1:6] = lb.reshape(batch_size, 5, feature_size[0], feature_size[1])
            send_conn.send((img_list, label))

        N = len(self._img_path_list)
        batch_loop = int(np.ceil(N/batch_size))
        connection_list = []
        nth_batch = 0
        perm = np.random.permutation(len(self))
        while True:
            for n in range(self._num_worker - len(connection_list)):
                recv, send = Pipe(False)
                batch_perm = perm[nth_batch*batch_size:(nth_batch+1)*batch_size]
                arg = (
                        [self._img_path_list[p] for p in  batch_perm],
                        [self._annotation_list[p] for p in batch_perm],
                        send
                    )
                p = Process(target=build, args=arg)
                p.start()
                connection_list.append(recv)
                nth_batch += 1

            yield connection_list.pop(0).recv()
            if nth_batch >=batch_loop: break

        for r in connection_list:
            yield r.recv()


def create_mapping(label_root_path):
    annotation_path = label_root_path
    mapping = {}
    for filename in sorted(os.listdir(annotation_path)):
        tree = ElementTree.parse(os.path.join(annotation_path, filename))
        root = tree.getroot()
        for object_tree in root.findall('object'):
            class_name = object_tree.find('name').text
            mapping[class_name] = 1
    mapping = {k:i for i, k in enumerate(sorted(mapping.keys()))}
    return mapping


def load_bbox(img_root_path, label_root_path):
    annotation_list = []
    img_path = img_root_path
    annotation_path = label_root_path
    img_path_list = [os.path.join(img_root_path, path) for path in sorted(os.listdir(img_path))]
    mapping = create_mapping(label_root_path)
    for filename in sorted(os.listdir(annotation_path)):
        tree = ElementTree.parse(os.path.join(annotation_path, filename))
        root = tree.getroot()
        size_tree = root.find('size')
        width = float(size_tree.find('width').text)
        height = float(size_tree.find('height').text)
        image_data = []
        for object_tree in root.findall('object'):
            bounding_box = object_tree.find('bndbox')
            xmin = float(bounding_box.find('xmin').text)-1
            ymin = float(bounding_box.find('ymin').text)-1
            xmax = float(bounding_box.find('xmax').text)-1
            ymax = float(bounding_box.find('ymax').text)-1
            w = xmax - xmin
            h = ymax - ymin
            x = xmin + w/2.
            y = ymin + h/2.
            bounding_box = [x, y, w, h]
            class_name = object_tree.find('name').text
            cn = mapping[class_name]
            # onehot = [0]*cn + [1] + [0]*(len(mapping)-cn-1)
            image_data.append({'bndbox': bounding_box, 'name': cn, 'size':[width, height]})
        annotation_list.append(image_data)
    return img_path_list, annotation_list


def calc_iou_xyxy(box1, box2):
    inter_w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    inter_h = min(box1[3], box2[3]) - max(box1[1], box2[1])
    if inter_h <= 0 or inter_w <= 0: return 0
    inter = inter_h * inter_w
    union = (box1[2] - box1[0])*(box1[3] - box1[1]) + (box2[2] - box2[0])*(box2[3] - box2[1]) - inter
    iou = inter/union
    return iou

def calc_iou_xywh(box1, box2):
    box1 = (box1[0] - box1[2]/2.0, box1[1] - box1[3]/2.0, box1[0] + box1[2]/2.0, box1[1] + box1[3]/2.0)
    box2 = (box2[0] - box2[2]/2.0, box2[1] - box2[3]/2.0, box2[0] + box2[2]/2.0, box2[1] + box2[3]/2.0)
    inter_w = min(box1[2], box2[2]) - max(box1[0], box2[0])
    inter_h = min(box1[3], box2[3]) - max(box1[1], box2[1])
    if inter_h <= 0 or inter_w <= 0: return 0
    inter = inter_h * inter_w
    union = (box1[2] - box1[0])*(box1[3] - box1[1]) + (box2[2] - box2[0])*(box2[3] - box2[1]) - inter
    iou = inter/union
    return iou



def create_anchor(box_list, size_list, n_anchor=5, base_size=(416, 416)):
    """
    Requires resized box list.
    Perform k-means clustering using custom metric.
    We want to get only anchor's size so we don't have to consider coordinates.

    Args:
        box_list: 
    """
    convergence = 0.01
    box_list = [(0, 0, box[0]*base_size[0]/size[0],
                box[1]*base_size[1]/size[1]) for box, size in zip(box_list, size_list)]
    centroid_index = np.random.permutation(len(box_list))[:n_anchor]
    centroid = [box_list[i] for i in centroid_index]

    def update(centroid, box_list):
        loss = 0
        group = [[] for _ in range(n_anchor)]
        new_centroid = [[0, 0, 0, 0] for _ in range(n_anchor)]
        metric = lambda x, center: 1 - calc_iou_xywh(x, center)
        for box in box_list:
            minimum_distance = 100
            for c_ind, cent in enumerate(centroid):
                distance = metric(box, cent)
                if distance < minimum_distance:
                    minimum_distance = distance
                    group_index = c_ind
            group[group_index].append(box)
            new_centroid[group_index][2] += box[2] # Sum up for calc mean.
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
    return new_centroids


if __name__ == "__main__":
    img_list, annotation_list = load_bbox("../dataset/VOC2007/JPEGImages/", "../dataset/VOC2007/Annotations/")
    dist = Yolov2Distributor(img_list, annotation_list)

    start_t = time.time()
    for x, y in dist.detection_batch(64):
        print(len(x), time.time() - start_t)
        start_t = time.time()
