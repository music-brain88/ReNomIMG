#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
from PIL import Image


"""Naming Rule.
- build_target_[algorithm name]

This type of function performs following transformation.
  annotation_list => algorithm specific target data

For confirming the format of annotation_list, 
see the function 'parse_xml_detection' written in load.py.
"""

RESIZE_METHOD = Image.BILINEAR


class DataBuilderBase(object):

    def __init__(self, imsize):
        self.imsize = imsize
        self.class_mapping = None

    def __call__(self, img_path_list, annotation_list, augmentation):
        return self.build(img_path_list, annotation_list, augmentation)

    def build(self, img_path_list, annotation_list, augmentation):
        pass

    def reverce_label(self, label_list):
        pass

    def load_img(self, path):
        img = Image.open(path)
        img.load()
        w, h = img.size
        img = img.convert('RGB')
        img = img.resize(self.imsize, RESIZE_METHOD)
        img = np.asarray(img).transpose(2, 0, 1).astype(np.float32)
        return img, self.imsize[0] / float(w), self.imsize[1] / h

    def create_class_mapping(self, annotation_list):
        pass

    @property
    def class_list(self):
        return self.class_mapping


class DataBuilderDefault(DataBuilderBase):
    pass


class DataBuilderClassification(DataBuilderBase):
    """
    Annotation_list must be list of class name.
    """

    def create_class_mapping(self, annotation_list):
        pass

    def __init__(self, imsize, class_mapping):
        super(DataBuilderClassification, self).__init__(imsize)
        self.class_mapping = class_mapping

    def __call__(self, img_path_list, annotation_list, augmentation):
        # Check the class mapping.
        n_class = len(self.class_mapping)

        img_list = []
        label_list = []
        for img_path, an_data in zip(img_path_list, annotation_list):
            one_hot = np.zeros(n_class)
            img, sw, sh = self.load_img(img_path)
            img_list.append(img)
            one_hot[an_data] = 1.
            label_list.append(one_hot)
        if augmentation is not None:
            return augmentation(np.array(img_list), np.array(label_list), mode="classification")
        else:
            return np.array(img_list), np.array(label_list)



class DataBuilderDetection(DataBuilderBase):

    def __init__(self, imsize):
        super(DataBuilderDetection, self).__init__(imsize)

    def build(self, img_path_list, annotation_list, augmentation):
        # Check the class mapping.
        if self.class_mapping is None:
            class_dict = {}
            for annotation in annotation_list:
                for obj in annotation:
                    class_dict[obj['name']] = 1
            self.class_mapping = {k: i for i, k in enumerate(sorted(class_dict.keys()))}

        img_list = []
        new_annotation_list = []
        for i, img_path in enumerate(img_path_list):
            img, sw, sh = self.load_img(img_path)
            img_list.append(img)
            new_annotation_list.append([
                {
                    "box": [
                        an["box"][0] * sw,
                        an["box"][1] * sh,
                        an["box"][2] * sw,
                        an["box"][3] * sh,
                    ],
                    "name": an["name"]
                }
                for an in annotation_list[i]])

        if augmentation is not None:
            img_list, annotation_list = augmentation(
                np.array(img_list), new_annotation_list, mode="detection")
        else:
            img_list, annotation_list = np.array(img_list), new_annotation_list

        # Get max number of objects in one image.
        dlt = 4 + 1
        max_obj_num = np.max([len(annotation) for annotation in annotation_list])
        target = np.zeros((len(annotation_list), max_obj_num * dlt), dtype=np.float32)

        for i, annotation in enumerate(annotation_list):
            for j, obj in enumerate(annotation):
                target[i, j * dlt:(j + 1) * dlt - 1] = [obj['box'][0],
                                                        obj['box'][1], obj['box'][2], obj['box'][3]]
                target[i, (j + 1) * dlt - 1] = self.class_mapping[obj['name']]
        return img_list, target


class DataBuilderYolov1(DataBuilderBase):

    def __init__(self, cells, imsize):
        super(DataBuilderYolov1, self).__init__(imsize)
        self.class_mapping = None
        self._cells = cells

    def create_class_mapping(self, annotation_list):
        if self.class_mapping is None:
            class_dict = {}
            for annotation in annotation_list:
                for obj in annotation:
                    class_dict[obj['name']] = 1
            self.class_mapping = {k: i for i, k in enumerate(sorted(class_dict.keys()))}

    def build(self, img_path_list, annotation_list, augmentation):
        """Use to transform a list of objects per image into a image*cells*cells*(5+classes) matrix.

        Returns:
            (ndarray): Yolo formatted target array.

        This method returns yolo formatted target array which shape is (N, cell*cell*(5 + 1).
        N is batch size. The array consists of following data.

        1. existence flag: A flag which indicates if an object exists in the cell.
        2. Coordinates and size(x, y, w, h): Coordinate and size of each objects.
        3. Class id: The object's class number.

        [
            [existence x y w h 2 existence x y w h 2 ... ],
            [existence x y w h 3 existence x y w h 1 ... ],
        ]
        """
        # Check the class mapping.
        if self.class_mapping is None:
            class_dict = {}
            for annotation in annotation_list:
                for obj in annotation:
                    class_dict[obj['name']] = 1
            self.class_mapping = {k: i for i, k in enumerate(sorted(class_dict.keys()))}

        img_list = []
        new_annotation_list = []
        for i, img_path in enumerate(img_path_list):
            img, sw, sh = self.load_img(img_path)
            img_list.append(img)
            new_annotation_list.append([
                {
                    "box": [
                        an["box"][0] * sw,
                        an["box"][1] * sh,
                        an["box"][2] * sw,
                        an["box"][3] * sh,
                    ],
                    "name": an["name"]
                }
                for an in annotation_list[i]])

        if augmentation is not None:
            img_list, annotation_list = augmentation(
                np.array(img_list), new_annotation_list, mode="detection")
        else:
            img_list, annotation_list = np.array(img_list), new_annotation_list

        # Cell can be tuple, list or int.
        if isinstance(self._cells, (tuple, list)):
            cell_w = self._cells[0]
            cell_h = self._cells[1]
        elif isinstance(self._cells, int):
            cell_w = self._cells
            cell_h = self._cells
        num_class = len(self.class_mapping)
        img_w, img_h = self.imsize
        target = np.zeros((len(annotation_list), cell_h, cell_w, 5 + num_class))
        for ind_img in range(len(target)):
            annotation = annotation_list[ind_img]
            for ind_obj in range(len(annotation)):
                obj = annotation[ind_obj]
                c = self.class_mapping[obj['name']]
                onehot_class = [0] * c + [1] + [0] * (num_class - c - 1)
                truth_x = np.clip(obj['box'][0], 0, img_w)
                truth_y = np.clip(obj['box'][1], 0, img_h)
                truth_w = np.clip(obj['box'][2], 0, img_w)
                truth_h = np.clip(obj['box'][3], 0, img_h)
                norm_x = truth_x * .99 * cell_w / img_w
                norm_y = truth_y * .99 * cell_h / img_h
                norm_w = truth_w / img_w
                norm_h = truth_h / img_h
                target[ind_img, int(norm_y), int(norm_x)] = \
                    np.concatenate(([1, norm_x % 1, norm_y % 1, norm_w, norm_h], onehot_class))
        target = target.reshape(len(annotation_list), -1)
        return img_list, target

    def reverce_label(self, label_list):
        pass
